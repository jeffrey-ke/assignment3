import pdb
import os

import numpy as np
import cv2

import utils
from q2 import error_all

def q1a(scene="bench"):
    os.makedirs("output/q1", exist_ok=True)
    q1 = utils.load_q1(scene)
    F = utils.create_F(q1.corresp)
    pred_values = np.einsum('mi,bij,mj->bm',
            q1.corresp[1],
            F,
            q1.corresp[0]
    )
    assert np.allclose(pred_values, np.zeros_like(pred_values))
    img = utils.draw_epipolar(q1, F)
    utils.save_display("q1", scene, img) 

def q1b(scene="hydrant"):
    os.makedirs("output/q1", exist_ok=True)
    q1 = utils.load_q1(scene)
    ## hypothesis: will some solutions fit better than others? I.e. will the pred values be lower?
    Fs = utils.create_F(q1.corresp)
    pred_values = np.einsum('mi,bij,mj->bm',
            q1.corresp[1],
            Fs,
            q1.corresp[0]
    )
    assert np.allclose(pred_values, np.zeros_like(pred_values))

    utils.draw_annos(q1, vis=True)

    img1 = utils.draw_epipolar(q1, Fs[0:1])
    # img2 = utils.draw_epipolar(q1, Fs[1:2])
    # img3 = utils.draw_epipolar(q1, Fs[2:3])

    utils.save_display("q1", scene, img1)
    # utils.save_display("q1", scene, img2)
    # utils.save_display("q1", scene, img3)

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    # q1a("bench")
    # q1a("remote")
    # q1b("hydrant")
    # q1b("ball")
    q1 = utils.load_q1("bench")
    F = utils.create_F(q1.corresp)
    error = error_all(F, q1.corresp)
    pdb.set_trace()
