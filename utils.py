from math import sqrt, isclose
import pdb
from dataclasses import dataclass
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

class xy_plotter:

    def __init__(self, output_path, name):
        self.output_path = output_path
        self.name = name
        self.fig, self.ax = plt.subplots()
        self.xs1 = []
        self.ys1 = []
        self.xs2 = []
        self.ys2 = []
        self.output_path = output_path
        self.name = name

    def append_xy1(self, x, y):
        self.xs1.append(x)
        self.ys1.append(y)

    def append_xy2(self, x, y):
        self.xs2.append(x)
        self.ys2.append(y)

    def plot(self):
        seven_pt, = self.ax.plot(self.xs1, self.ys1, label="7 point")
        eight_pt,= self.ax.plot(self.xs2, self.ys2, label="8 point")
        self.ax.legend(handles=[seven_pt, eight_pt])

    def savefig(self):
        self.fig.savefig(os.path.join(self.output_path, f"{self.name}.png"))

    def __del__(self):
        plt.close(self.fig)



@dataclass
class Q1:
    corresp: np.ndarray
    corresp_noisy: np.ndarray
    img1: np.ndarray
    img2: np.ndarray
    K1: np.ndarray
    K2: np.ndarray
    gt: np.lib.npyio.NpzFile


def draw_annos(q1, noisy=False, vis=False):
    img1, img2 = q1.img1, q1.img2
    corresps = (
            unhomogenize(q1.corresp_noisy.transpose(1, 0, 2) if noisy else q1.corresp.transpose(1, 0, 2))
            .astype(int)
    )
    rng = np.random.default_rng(0)
    for (p_1, p_2) in corresps:
        color = tuple(rng.integers(0, 255, 3).tolist())
        img1 = cv2.circle(img1, p_1, radius=5, color=color, thickness=5)
        img2 = cv2.circle(img2, p_2, radius=5, color=color, thickness=5)

    if vis:
        cv2.imshow("img1", img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("img2", img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img1, img2

"""
base = None
for node, dirs, files in os.walk(os.getcwd()):
    if dataset in dirs:
        base = node
        break
assert base is not None
"""
_root = "data"
def load_q1(dataset="bench"):
    if dataset in ["bench", "remote"]:
        base = f"q1a/{dataset}"
    else:
        base = f"q1b/{dataset}"

    pts1, pts2 = np.load(os.path.join(_root, base, "corresp.npz")).values()
    pts1_noisy, pts2_noisy = np.load(os.path.join(_root, base, "corresp_noisy.npz")).values()

    corresp = homogenize(
        np.stack((pts1, pts2), axis=0)
    )
    corresp_noisy = homogenize(
        np.stack((pts1_noisy, pts2_noisy), axis=0)
    )

    image1 = cv2.imread(os.path.join(_root, base, "image1.jpg"))
    image2 = cv2.imread(os.path.join(_root, base, "image2.jpg"))
    assert image1 is not None and image2 is not None

    gt = np.load(os.path.join(_root, base, "gt.npz"))

    K1,K2 = np.load(os.path.join(_root, base, "intrinsics.npz")).values()
    return Q1(corresp, corresp_noisy, image1, image2, K1, K2, gt)


def draw_epipolar(q1, F):
    img2 = q1.img2.copy()
    points = q1.corresp[0]
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
        zeros = [root.real if isinstance(root, complex) else root for root in np.roots(coeffs) if not isinstance(root, complex) or isclose(root.imag, 0)]
        Fs = np.array(
            [l * F1 + (1 - l) * F2 for l in zeros]
        )
        return T_prime.T @ Fs @ T

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
    coords = coords.copy()
    coords = coords / coords[:, -1:]
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

def unhomogenize(points, keepdims=False):
    points = points.copy()
    points = points / points[..., -1:]
    return points if keepdims else points[..., :-1]

def save_display(question, name, img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f"output/{question}/{name}.jpg", img)
