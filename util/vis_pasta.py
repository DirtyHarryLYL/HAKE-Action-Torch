import os
import os.path as osp
import numpy as np
import cv2


def vis_image_pasta(im_path,
                    results,
                    thres=0.3,
                    out_dir='./vis',
                    flag='dt',
                    colors=None):
    """
    results: list of (probs, names)
    """

    if not osp.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    img_name = osp.basename(im_path)
    out_name = osp.join(out_dir, flag + '_' + img_name)

    x, y = 10, 20

    def put_text(im, text, color, x, y):
        if isinstance(color, str):
            color = {'red': (0, 0, 255), 'blue': (255, 0, 0)}[color]
        vis_im = cv2.putText(im, text, (x, y), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=color, thickness=2)
        y += 20
        return vis_im, x, y

    vis_im = cv2.imread(im_path)

    if isinstance(colors, list):
        assert len(colors) == len(results)
    else:
        colors = ['red' for _ in range(len(results))]

    for (probs, names), color in zip(results, colors):
        inds = np.nonzero(probs > thres)[0]
        for ind in inds:
            name = names[ind]
            prob = probs[ind]
            text = f'{name}: {prob:.2f}'
            vis_im, x, y = put_text(vis_im, text, color, x, y)

    cv2.imwrite(out_name, vis_im)
