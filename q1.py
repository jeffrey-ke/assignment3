import pdb

import numpy as np
import cv2

import utils

def q1():
    q1 = utils.load_q1()
    F = utils.create_F(q1.corresp)
    pred_values = np.einsum('mi,bij,mj->bm',
            q1.corresp[1],
            F,
            q1.corresp[0]
    )
    assert np.allclose(pred_values, np.zeros_like(pred_values))
    img = utils.draw_epipolar(q1.corresp[0], F, q1.img2)
    cv2.imshow('img', img)
    cv2.waitKey(0)       # waits indefinitely for a keypress
    cv2.destroyAllWindows()

if __name__ == "__main__":
    q1()
