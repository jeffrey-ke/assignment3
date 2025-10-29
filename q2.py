import os
import pdb
from math import log

import numpy as np
import cv2
from scipy.optimize import least_squares

import utils
from tqdm import tqdm

"""
    Implements the Sampson approximation for the error function
"""
def error_all(F, all_corresps):
    assert len(F.shape) == 3
    xs, xs_prime = all_corresps
    assert xs.shape[-1] == 3
    numerator = (
            np.einsum(
                    'mi,bij,mj->bm',
                    xs_prime,
                    F,
                    xs
            )
            **2
    )
    Fx = np.einsum('bij,mj->bmi', F, xs)
    assert Fx.shape == (F.shape[0], xs.shape[0], 3)
    Fx_prime = np.einsum('bij,mj->bmi', F.transpose(0, 2, 1), xs_prime)
    denominator = (Fx[..., :-1]**2).sum(axis=-1) + (Fx_prime[..., :-1]**2).sum(axis=-1)
    assert numerator.shape == (F.shape[0], xs.shape[0])
    assert denominator.shape == (F.shape[0], xs.shape[0])
    return numerator / denominator

def lm_F(F_init, best_inlier_set):
    assert len(best_inlier_set.shape) == 3
    def residual(F):
        F = F.reshape(1, 3, 3)
        error = error_all(F, best_inlier_set)
        return error.squeeze()
    result = least_squares(residual, F_init.flatten())
    assert result.success
    return result.x.reshape(3,3)

def ransac_F(noisy_corresps, nc, max_err=0.05, inlier_ratio=0.55, thresh=1.25):
    best_F = None
    best_inlier_set = np.empty((0, 2, 3))
    best_error = float('inf')

    iters = 10000#int(log(max_err)/log(1 - inlier_ratio**nc))
    corresps = noisy_corresps.transpose(1, 0, 2) # M pairs

    rng = np.random.default_rng(0)
    for _ in tqdm(range(iters)):
        samples = rng.choice(corresps, nc, replace=False)
        assert samples.shape == (nc, 2, 3) # nc pairs
        F = utils.create_F(samples.transpose(1, 0, 2))
        assert len(F.shape) == 3
        errors = error_all(F, corresps.transpose(1, 0, 2))
        assert errors.shape == (F.shape[0], corresps.shape[0])
        best_F_idx = np.argmin(errors.sum(axis=-1))
        inlier_mask = errors[best_F_idx] < thresh
        assert inlier_mask.shape == (corresps.shape[0],)
        inliers = corresps[inlier_mask]
        assert inliers.shape[1:] == (2,3)

        if len(inliers) > len(best_inlier_set):
            best_inlier_set = inliers
            best_F = F[best_F_idx]
            best_error = errors[best_F_idx].mean()

    final_F = lm_F(best_F, best_inlier_set.transpose(1, 0, 2))
    return final_F

if __name__ == "__main__":
    q1a = utils.load_q1("bench")
    lm_f = ransac_F(q1a.corresp_noisy, 8)
    img = utils.draw_epipolar(q1a, lm_f[None, ...])
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
