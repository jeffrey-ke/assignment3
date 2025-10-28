from math import sqrt
import pdb
from dataclasses import dataclass
import os

import numpy as np
import cv2

@dataclass
class Q1:
    corresp: np.ndarray
    corresp_noisy: np.ndarray
    img1: np.ndarray
    img2: np.ndarray
    K1: np.ndarray
    K2: np.ndarray
    gt: np.lib.npyio.NpzFile
_root = "data"
def load_q1(dataset="bench"):
    q1a = f"q1a/{dataset}"

    pts1, pts2 = np.load(os.path.join(_root, q1a, "corresp.npz")).values()
    pts1_noisy, pts2_noisy = np.load(os.path.join(_root, q1a, "corresp_noisy.npz"))

    corresp = homogenize(
        np.stack((pts1, pts2), axis=0)
    )
    corresp_noisy = homogenize(
        np.stack((pts1_noisy, pts2_noisy), axis=0)
    )

    image1 = cv2.imread(os.path.join(_root, q1a, "image1.jpg"))
    image2 = cv2.imread(os.path.join(_root, q1a, "image2.jpg"))
    assert image1 is not None and image2 is not None

    gt = np.load(os.path.join(_root, q1a, "gt.npz"))

    K1,K2 = np.load(os.path.join(_root, q1a, "intrinsics.npz")).values()
    return Q1(corresp, corresp_noisy, image1, image2, K1, K2, gt)


def draw_epipolar(points, F, img2):
    assert F.shape[0] == 1
    F = F.squeeze()
    height,width = img2.shape[:-1]
    lines = points @ F.T # np.einsum('bi,ji->bj')
    rng = np.random.default_rng(0)
    for line in lines:
        a,b,c = line
        if abs(b) > abs(a):
            pt1 = (0, int(-c/b))
            pt2 = (width, int((-c - a * width)/b))
        else:
            pt1 = (int(-c/a), 0)
            pt2 = (int((-c - b * height)/a), height)
        color = tuple(rng.integers(0, 255, 3).tolist())
        img2 = cv2.line(img2, pt1, pt2, color=color, thickness=4)
    return img2

def create_F(correspondences):
    n, nc, dim = correspondences.shape
    assert n == 2 and dim == 3 # again assuming that points are homogeneous
    xs, xs_prime = correspondences
    T, T_prime = get_T(xs, sqrt(2)), get_T(xs_prime, sqrt(2))

    correspondences = correspondences.copy()
    correspondences[0, ...] = correspondences[0, ...] @ T.T
    correspondences[1, ...] = correspondences[1, ...] @ T_prime.T

    def create_row(correspondence):
        x, x_prime = correspondence
        return np.outer(x_prime, x).flatten()

    A = (
            np.stack(
                [create_row(correspondence) for correspondence in correspondences.transpose(1,0,2)],
                axis=0
            )
            .reshape(-1, 9)
    )
    _, E, Vh = np.linalg.svd(A)
    if nc == 7:
        F1, F2 = Vh[-2:].reshape(2,3,3)
        ls = [-1, 0, 1, 2]
        F = lambda l : l * F1 + (1 - l) * F2
        coeffs = (
            np.polyfit(
                ls,
                [np.linalg.det(F(l)) for l in ls],
                3
            )
            [::-1]
        )
        zeros = [root for root in np.roots(coeffs) if not isinstance(root, complex)]
        Fs = np.array(
            [l * F1 + (1 - l) * F2 for l in zeros]
        )
        return np.linalg.inv(T_prime) @ Fs @ T

    elif nc == 8:
        F = Vh[-1].reshape(1,3,3)
        U, E, Vh = np.linalg.svd(F)
        F = T_prime.T @ U @ np.diag([*E[0,:-1], 0]) @ Vh @ T
        return F
    else:
        raise ValueError(f"Not the right number of correspondences. You passed {nc} correspondences but there only should be 7 or 8")

def create_E(F, K1, K2):
    return K2.T @ F @ K1

def get_T(coords, target_scale):
    assert coords.shape[-1] > 2
    coords /= coords[:, -1:]
    mu = np.mean(coords[:, :-1], axis=0)
    cur_scale = np.mean(
            np.linalg.norm(coords[:, :-1] - mu, axis=-1),
            axis=0
        )
    scale = target_scale / cur_scale
    T = np.diag(
        [
            *[scale] * (coords.shape[-1] - 1),
            1
        ]
    )
    T[:-1, -1:] = -scale * mu.reshape(-1, 1)
    return T

def homogenize(points):
    return np.concatenate(
        (points, np.ones_like(points[..., -1:])), 
        axis=-1
    )

def unhomogenize(points):
    points /= points[..., -1:]
    return points[..., :-1]
