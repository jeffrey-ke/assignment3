from dataclasses import dataclass
import pdb
import os

from math import isclose, sqrt
from scipy.optimize import least_squares
import numpy as np
import cv2

import utils

def algebraic_triangulate(correspondences, cameras):
    P, P_prime = cameras
    def create_row(correspondence):
        x, x_prime = correspondence
        a, b, c = x
        a_prime, b_prime, c_prime = x_prime

        assert isclose(c, 1) and isclose(c_prime, 1)

        A_i = np.zeros((4,4), dtype=np.float32)
        A_i[0] = a * P[2] - c * P[0]
        A_i[1] = b * P[2] - c * P[1]
        A_i[2] = a_prime * P_prime[2] - c_prime * P_prime[0]
        A_i[3] = b_prime * P_prime[2] - c_prime * P_prime[1]

        return A_i

    A = np.concatenate(
        [create_row(correspondence) for correspondence in correspondences.transpose(1, 0, 2)],
        axis=0
    )
    *_, Vh = np.linalg.svd(A)

    return utils.unhomogenize(Vh[-1].reshape(-1, 4), keepdims=True)

def triangulate(correspondences, cameras):
    P, P_prime = cameras
    all_x, all_x_prime = correspondences
    T, T_prime = utils.get_T(all_x, sqrt(2)), utils.get_T(all_x_prime, sqrt(2))
    all_x = all_x @ T.T
    all_x_prime = all_x_prime @ T_prime.T
    correspondences = np.stack((all_x, all_x_prime), axis=0)

    P = T @ P
    P_prime = T_prime @ P_prime
    cameras = np.stack((P, P_prime), axis=0)
    init_3d_points = algebraic_triangulate(correspondences, cameras).flatten()

    def residual(points_3d):
        nonlocal all_x, all_x_prime
        points_3d = points_3d.reshape(-1,4)
        projections_P, projections_P_prime = utils.unhomogenize(
            np.einsum('cij,bj->cbi', cameras, points_3d)
        )
        all_x, all_x_prime = utils.unhomogenize(all_x), utils.unhomogenize(all_x_prime)
        distance = np.linalg.norm(projections_P - all_x, axis=-1) + np.linalg.norm(projections_P_prime - all_x_prime)
        assert len(distance.shape) == 1
        return distance

    result = least_squares(residual, init_3d_points)
    assert result.success
    return result.x.reshape(-1, 3)

@dataclass
class Q3:
    img1: np.ndarray
    img2: np.ndarray
    corresp: np.ndarray
    cameras: np.ndarray

def load_q3():
    root = "data/q3"
    img1 = cv2.imread(os.path.join(root, "img1.jpg"))
    img2 = cv2.imread(os.path.join(root, "img2.jpg"))
    assert img1 is not None and img2 is not None
    pts1 = np.load(os.path.join(root, "pts1.npy")).astype(np.float32)
    pts2 = np.load(os.path.join(root, "pts2.npy")).astype(np.float32)
    P = np.load(os.path.join(root, "P1.npy"))
    P_prime = np.load(os.path.join(root, "P2.npy"))
    corresps = utils.homogenize(np.stack((pts1, pts2), axis=0))
    cameras = np.stack((P, P_prime), axis=0)
    return Q3(img1, img2, corresps, cameras)

if __name__ == "__main__":
    q3 = load_q3()
    points_3d = triangulate(q3.corresp, q3.cameras)
    pdb.set_trace()
