import pdb
import os

import numpy as np
import cv2

import utils

def save_display(question, name, img):
    cv2.imshow('img', img)
    cv2.waitKey(0)       # waits indefinitely for a keypress
    cv2.destroyAllWindows()
    cv2.imwrite(f"output/{question}/{name}.jpg", img)

def q1(scene="bench"):
    os.makedirs("output/q1", exist_ok=True)
    q1 = utils.load_q1(scene)
    F = utils.create_F(q1.corresp)
    pred_values = np.einsum('mi,bij,mj->bm',
            q1.corresp[1],
            F,
            q1.corresp[0]
    )
    assert np.allclose(pred_values, np.zeros_like(pred_values))
    img = utils.draw_epipolar(q1.corresp[0], F, q1.img2)
    save_display("q1", scene, img) 

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    q1("bench")
    q1("remote")
