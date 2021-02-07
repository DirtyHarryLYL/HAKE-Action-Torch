import platform
import os
import subprocess
import time

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def make_cython_ext(name, module, sources):
    extra_compile_args = None
    if platform.system() != 'Windows':
        extra_compile_args = {
            'cxx': ['-Wno-unused-function', '-Wno-write-strings']
        }

    extension = Extension(
        '{}.{}'.format(module, name),
        [os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=[np.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args)
    extension, = cythonize(extension)
    return extension

def make_cuda_ext(name, module, sources):

    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })

def get_ext_modules():
    # only windows visual studio 2013+ support compile c/cuda extensions
    # If you force to compile extension on Windows and ensure appropriate visual studio
    # is intalled, you can try to use these ext_modules.
    ext_modules = [make_cython_ext(
            name='soft_nms_cpu',
            module='tools.inference_tools.detector.nms',
            sources=['src/soft_nms_cpu.pyx']),
        make_cuda_ext(
            name='nms_cpu',
            module='tools.inference_tools.detector.nms',
            sources=['src/nms_cpu.cpp']),
        make_cuda_ext(
            name='nms_cuda',
            module='tools.inference_tools.detector.nms',
            sources=['src/nms_cuda.cpp', 'src/nms_kernel.cu'])]
    return ext_modules

setup(
    name="activity2vec",
    version="1.0.0",
    author="MVIG",
    url="https://github.com/DirtyHarryLYL/HAKE-Action-Torch",
    description="Activity2Vec model for action understanding",
    install_requires=["torch==1.4.0",
    "torchvision==0.5.0",
    "lmdb",
    "tqdm",
    "easydict",
    "cupy-cuda100==7.8.0",
    "opencv-python",
    "imageio",
    "natsort",
    "cython",
    "pyyaml",
    "matplotlib",
    "munkres",
    "tensorboardx",
    "terminaltables",
    "timm",
    "visdom",
    "imutils",
    "numpy",
    "scipy",
    "anycurve",
    "yacs",
    "moviepy",
    "h5py",
    "PyTurboJPEG"
    ],
    packages=find_packages(),
    ext_modules=get_ext_modules(),
    cmdclass={'build_ext': BuildExtension}
)
