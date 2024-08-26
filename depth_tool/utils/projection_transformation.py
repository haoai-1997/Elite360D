import math
from copy import copy
from functools import lru_cache
from functools import partial

import igl
import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import norm
from scipy.ndimage.interpolation import map_coordinates


@lru_cache(maxsize=12)
def get_icosahedron(level=0):
    if level == 0:
        v, f = get_base_icosahedron()
        return v, f
    # require subdivision
    v, f = get_icosahedron(level - 1)
    v, f = subdivision(v, f, 1)
    return v, f
def subdivision(v, f, level=1):
    for _ in range(level):
        # subdivision
        v, f = igl.upsample(v, f)
        # normalize
        v /= np.linalg.norm(v, axis=1)[:, np.newaxis]
    return v, f

def get_base_icosahedron():
    t = (1.0 + 5.0 ** .5) / 2.0
    vertices = [-1, t, 0, 1, t, 0, 0, 1, t, -t, 0, 1, -t, 0, -1, 0, 1, -t, t, 0, -1, t, 0,
                1, 0, -1, t, -1, -t, 0, 0, -1, -t, 1, -t, 0]
    faces = [0, 2, 1, 0, 3, 2, 0, 4, 3, 0, 5, 4, 0, 1, 5,
             1, 7, 6, 1, 2, 7, 2, 8, 7, 2, 3, 8, 3, 9, 8, 3, 4, 9, 4, 10, 9, 4, 5, 10, 5, 6, 10, 5, 1, 6,
             6, 7, 11, 7, 8, 11, 8, 9, 11, 9, 10, 11, 10, 6, 11]

    # make every vertex have radius 1.0
    vertices = np.reshape(vertices, (-1, 3)) / (np.sin(2 * np.pi / 5) * 2)
    faces = np.reshape(faces, (-1, 3))

    # Rotate vertices so that v[0] = (0, -1, 0), v[1] is on yz-plane
    ry = -vertices[0]
    rx = np.cross(ry, vertices[1])
    rx /= np.linalg.norm(rx)
    rz = np.cross(rx, ry)
    R = np.stack([rx, ry, rz])
    vertices = vertices.dot(R.T)
    return np.float32(vertices), np.int32(faces)

def erp2sphere(erp_img, V, method="linear"):
    """

    Parameters
    ----------
    erp_img: equirectangular projection image
    V: array of spherical coordinates of shape (n_vertex, 3)
    method: interpolation method. "linear" or "nearest"
    """
    uv = xyz2uv(V)
    img_idx = uv2img_idx(uv, erp_img)
    x = remap(erp_img, img_idx, method=method)
    return x

def xyz2uv(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    u = np.arctan2(x, z)
    c = np.sqrt(x * x + z * z)
    v = np.arctan2(y, c)
    return np.stack([u, v], axis=-1)

def uv2xyz(uv):
    sin_u = np.sin(uv[..., 0])
    cos_u = np.cos(uv[..., 0])
    sin_v = np.sin(uv[..., 1])
    cos_v = np.cos(uv[..., 1])
    return np.stack([
        cos_v * sin_u,
        sin_v,
        cos_v * cos_u,
    ], axis=-1)

def coords2uv(coords, w, h):
    #output uv size w*h*2
    uv = np.zeros_like(coords, dtype = np.float32)
    middleX = w/2 + 0.5
    middleY = h/2 + 0.5
    uv[..., 0] = (coords[...,0] - middleX) / w * 2 * np.pi
    uv[..., 1] = (coords[...,1] - middleY) / h * np.pi
    return uv

def uv2img_idx(uv, erp_img):
    h, w = erp_img.shape[:2]
    delta_w = 2 * np.pi / w
    delta_h = np.pi / h
    x = uv[..., 0] / delta_w + w / 2 - 0.5
    y = uv[..., 1] / delta_h + h / 2 - 0.5
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    return np.stack([y, x], axis=0)

def remap(img, img_idx, cval=[0, 0, 0], method="linear"):
    # interpolation method
    if method == "linear":
        order = 1
    else:
        # nearest
        order = 0

    # remap image
    if img.ndim == 2:
        # grayscale
        x = map_coordinates(img, img_idx, order=order, cval=cval[0])
    elif img.ndim == 3:
        # color
        x = np.zeros([*img_idx.shape[1:], img.shape[2]], dtype=img.dtype)
        for i in range(img.shape[2]):
            x[..., i] = map_coordinates(img[..., i], img_idx, order=order, cval=cval[i])
    else:
        assert False, 'img.ndim should be 2 (grayscale) or 3 (color)'

    return x
