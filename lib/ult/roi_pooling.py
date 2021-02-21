import torch.nn as nn
from torch.nn.modules.utils import _pair
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.nn.functional as F
from collections import namedtuple
import cupy
from string import Template

CUDA_NUM_THREADS = 1024
Stream = namedtuple('Stream', ['ptr'])

def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'
    elif isinstance(t, torch.cuda.IntTensor):
        return 'int'
    else:
        raise ValueError('WIP. Check pyinn-issue-#10')

@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)
'''

def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_roipooling_kernel = kernel_loop + '''
extern "C"
__global__ void roi_pooling2d_forward_kernel(
    const ${Dtype}* bottom_data, const ${Dtype}* bottom_rois,
    ${Dtype}* top_data, ${Dtype_ind}* argmax_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    // pos in output filter
    int pw = index % ${pooled_width};
    int ph = (index / ${pooled_width}) % ${pooled_height};
    int c = (index / ${pooled_width} / ${pooled_height}) % ${channels};
    int num = index / ${pooled_width} / ${pooled_height} / ${channels};

    int roi_batch_ind = bottom_rois[num * 5 + 0];
    int roi_start_w = round(bottom_rois[num * 5 + 1] * ${spatial_scale});
    int roi_start_h = round(bottom_rois[num * 5 + 2] * ${spatial_scale});
    int roi_end_w = round(bottom_rois[num * 5 + 3] * ${spatial_scale});
    int roi_end_h = round(bottom_rois[num * 5 + 4] * ${spatial_scale});

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    float bin_size_h = static_cast<float>(roi_height)
                   / static_cast<float>(${pooled_height});
    float bin_size_w = static_cast<float>(roi_width)
                   / static_cast<float>(${pooled_width});

    int hstart = static_cast<int>(floor(static_cast<float>(ph)
                                  * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<float>(pw)
                                  * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                                * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                                * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), ${height});
    hend = min(max(hend + roi_start_h, 0), ${height});
    wstart = min(max(wstart + roi_start_w, 0), ${width});
    wend = min(max(wend + roi_start_w, 0), ${width});
    bool is_empty = (hend <= hstart) || (wend <= wstart);
    // Define an empty pooling region to be zero
    float maxval = is_empty ? 0 : -1E+37;
    // If nothing is pooled, argmax=-1 causes nothing to be backprop'd
    int maxidx = -1;
    int data_offset = (roi_batch_ind * ${channels} + c) * ${height} * ${width};
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * ${width} + w;
        if (bottom_data[data_offset + bottom_index] > maxval) {
          maxval = bottom_data[data_offset + bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}
'''


_roipooling_kernel_backward_grad_input = kernel_loop + '''
extern "C"
__global__ void roi_pooling2d_backward_grad_input_kernel(
    const ${Dtype}* const top_diff, const ${Dtype_ind}* const argmax_data,
    const ${Dtype}* bottom_rois, ${Dtype}* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    int w = index % ${width};
    int h = (index / ${width}) % ${height};
    int c = (index / (${width} * ${height})) % ${channels};
    int num = index / (${width} * ${height} * ${channels});
    float gradient = 0;

    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < ${num_rois}; ++roi_n) {
        // Skip if ROI's batch index doesn't match num
        if (num != static_cast<int>(bottom_rois[roi_n * 5])) {
            continue;
        }
        int roi_start_w = round(bottom_rois[roi_n * 5 + 1]
                                * ${spatial_scale});
        int roi_start_h = round(bottom_rois[roi_n * 5 + 2]
                                * ${spatial_scale});
        int roi_end_w = round(bottom_rois[roi_n * 5 + 3]
                              * ${spatial_scale});
        int roi_end_h = round(bottom_rois[roi_n * 5 + 4]
                              * ${spatial_scale});
        // Skip if ROI doesn't include (h, w)
        const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                             h >= roi_start_h && h <= roi_end_h);
        if (!in_roi) {
            continue;
        }
        int offset = (roi_n * ${channels} + c) * ${pooled_height}
                     * ${pooled_height};
        // Compute feasible set of pooled units that could have pooled
        // this bottom unit
        // Force malformed ROIs to be 1x1
        int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        float bin_size_h = static_cast<float>(roi_height)
                       / static_cast<float>(${pooled_height});
        float bin_size_w = static_cast<float>(roi_width)
                       / static_cast<float>(${pooled_height});
        int phstart = floor(static_cast<float>(h - roi_start_h)
                            / bin_size_h);
        int phend = ceil(static_cast<float>(h - roi_start_h + 1)
                         / bin_size_h);
        int pwstart = floor(static_cast<float>(w - roi_start_w)
                            / bin_size_w);
        int pwend = ceil(static_cast<float>(w - roi_start_w + 1)
                         / bin_size_w);
        phstart = min(max(phstart, 0), ${pooled_height});
        phend = min(max(phend, 0), ${pooled_height});
        pwstart = min(max(pwstart, 0), ${pooled_height});
        pwend = min(max(pwend, 0), ${pooled_height});
        for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
                int index_ = ph * ${pooled_height} + pw + offset;
                if (argmax_data[index_] == (h * ${width} + w)) {
                    gradient += top_diff[index_];
                }
            }
        }
    }
    bottom_diff[index] = gradient;
  }
}
'''


class ROIPooling2d(Function):
    """Spatial Region of Interest (ROI) pooling function.

    This function acts similarly to :class:`~pytorch.nn.MaxPool2d`, but
    it computes the maximum of input spatial patch for each channel
    with the region of interest.

    See the original paper proposing ROIPooling:
    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_.

    Args:
        x (~pytorch.autograd.Variable): Input variable. The shape is expected
            to be 4 dimentional: (n: batch, c: channel, h, height, w: width).
        rois (~pytorch.autograd.Variable): Input roi variable. The shape is
            expected to be (m: num-rois, 5), and each roi is set as below:
            (batch_index, x_min, y_min, x_max, y_max).
        output_size (int or tuple): the target output size of the image of the
            form H x W. Can be a tuple (H, W) or a single number H for a square
            image H x H.
        spatial_scale (float): scale of the rois if resized.
    Returns:
        `~pytorch.autograd.Variable`: Output variable.
    """

    @staticmethod
    def forward(self, input, rois, output_size=(7, 7), spatial_scale=1.0):
        self.output_h, self.output_w = output_size
        self.spatial_scale = spatial_scale
        assert input.dim() == 4 and input.is_cuda
        assert rois.dim() == 2 and rois.size(1) == 5 and rois.is_cuda
        _, channels, height, width = input.size()
        num_rois, _ = rois.size()

        output = input.new(num_rois, channels, self.output_h, self.output_w)
        n = output.numel()
        with torch.cuda.device_of(input):
            argmax_data = torch.cuda.IntTensor(
                num_rois, channels, self.output_h, self.output_w)
            f = load_kernel(
                'roi_pooling2d_forward_kernel',
                _roipooling_kernel,
                Dtype=Dtype(input), Dtype_ind=Dtype(argmax_data), nthreads=n,
                spatial_scale=self.spatial_scale,
                channels=channels, height=height, width=width,
                pooled_height=self.output_h, pooled_width=self.output_w)
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(n), 1, 1),
              args=[input.data_ptr(), rois.data_ptr(),
                    output.data_ptr(), argmax_data.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        self.save_for_backward(rois)
        self.argmax_data = argmax_data
        self.input_size = input.size()
        return output

    @staticmethod
    @once_differentiable
    def backward(self, grad_output):
        assert grad_output.is_cuda
        rois, = self.saved_tensors
        argmax_data = self.argmax_data
        num_rois = rois.size()[0]
        batch_size, channels, height, width = self.input_size

        grad_input = grad_rois = None
        grad_input = grad_output.new(batch_size, channels, height, width)
        n = grad_input.numel()
        with torch.cuda.device_of(grad_output):
            f = load_kernel('roi_pooling2d_backward_grad_input_kernel',
                            _roipooling_kernel_backward_grad_input,
                            Dtype=Dtype(grad_output),
                            Dtype_ind=Dtype(argmax_data),
                            nthreads=n,
                            num_rois=num_rois,
                            spatial_scale=self.spatial_scale,
                            channels=channels, height=height, width=width,
                            pooled_height=self.output_h,
                            pooled_width=self.output_w)
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(n), 1, 1),
              args=[grad_output.data_ptr(), argmax_data.data_ptr(),
                    rois.data_ptr(), grad_input.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return grad_input, grad_rois, None, None


def roi_pooling_2d(input, rois, output_size=(7, 7), spatial_scale=1.0):
    r"""Applies a 2D ROI pooling over an input signal composed of several input
    planes and a set of regions parametrized as indices and bounding boxes

    See :class:`~ROIPooling2d` for details and output shape.

    Args:
        output_size (int or tuple): the target output size of the image of the
            form H x W. Can be a tuple (H, W) or a single number H for a square
            image H x H.
        spatial_scale (float): scale of the rois if resized.
    """
    return ROIPooling2d.apply(input, rois, output_size, spatial_scale)


def roi_pooling_2d_pytorch(input, rois, output_size=(7, 7), spatial_scale=1.0):
    """Spatial Region of Interest (ROI) pooling function in pure pytorch/python

    This function acts similarly to `~roi_pooling_2d`, but performs a python
    loop over ROI. Note that this is not a direct replacement of
    `~roi_pooling_2d` (viceversa).

    See :function:`~roi_pooling_2d` for details and output shape.

    Args:
        output_size (int or tuple): the target output size of the image of the
            form H x W. Can be a tuple (H, W) or a single number H for a square
            image H x H.
        spatial_scale (float): scale of the rois if resized.
    """
    assert rois.dim() == 2
    assert rois.size(1) == 5
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)

    rois[:, 1:].mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        im = input.narrow(0, im_idx, 1)[...,
                                        roi[2]:(roi[4]+1),
                                        roi[1]:(roi[3]+1)]
        output.append(F.adaptive_max_pool2d(im, output_size))

    return torch.cat(output, 0)

class ROIPooling2d(nn.Module):
    """Spatial Region of Interest (ROI) pooling.

    This function acts similarly to :class:`~pytorch.nn.MaxPool2d`, but
    it computes the maximum of input spatial patch for each channel
    with the region of interest. This module only works with CUDA tensors.
    Take a look at the :class:`~ROIPooling2dPytorch` for an architecture
    agnostic implementation.

    See the original paper proposing ROIPooling:
    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_.

    Args:
        x (~pytorch.autograd.Variable): Input variable. The shape is expected
            to be 4 dimentional: (n: batch, c: channel, h, height, w: width).
        rois (~pytorch.autograd.Variable): Input roi variable. The shape is
            expected to be (m: num-rois, 5), and each roi is set as below:
            (batch_index, x_min, y_min, x_max, y_max).
        output_size (int or tuple): the target output size of the image of the
            form H x W. Can be a tuple (H, W) or a single number H for a square
            image H x H.
        spatial_scale (float): scale of the rois if resized.
    Returns:
        `~pytorch.autograd.Variable`: Output variable.
    """

    def __init__(self, output_size, spatial_scale=1.0):
        super(ROIPooling2d, self).__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):
        return roi_pooling_2d(input, rois, self.output_size,
                              self.spatial_scale)

    def __repr__(self):
        return ('{}(output_size={}, spatial_scale={:.6f})'.format(
            self.__class__.__name__, str(self.output_size),
            str(self.spatial_scale)))


class ROIPooling2dPytorch(nn.Module):
    """Spatial Region of Interest (ROI) pooling.

    This function acts similarly to :class:`~ROIPooling2d`, but performs a
    python loop over ROI. Note that this is not a direct replacement of that
    operation and viceversa.

    See the original paper proposing ROIPooling:
    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_.

    Args:
        x (~pytorch.autograd.Variable): Input variable. The shape is expected
            to be 4 dimentional: (n: batch, c: channel, h, height, w: width).
        rois (~pytorch.autograd.Variable): Input roi variable. The shape is
            expected to be (m: num-rois, 5), and each roi is set as below:
            (batch_index, x_min, y_min, x_max, y_max).
        output_size (int or tuple): the target output size of the image of the
            form H x W. Can be a tuple (H, W) or a single number H for a square
            image H x H.
        spatial_scale (float): scale of the rois if resized.
    Returns:
        `~pytorch.autograd.Variable`: Output variable.
    """

    def __init__(self, output_size, spatial_scale=1.0):
        super(ROIPooling2dPytorch, self).__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = spatial_scale

    def forward(self, input, rois, debug=False):
        if debug:
            print("==> [in ROIPooling2dPytorch] rois.shape:", rois.shape)  
        return roi_pooling_2d_pytorch(input, rois, self.output_size,
                                      self.spatial_scale)

    def __repr__(self):
        return ('{}(output_size={}, spatial_scale={:.6f})'.format(
            self.__class__.__name__, str(self.output_size),
            str(self.spatial_scale)))
