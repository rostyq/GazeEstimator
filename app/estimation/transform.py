import numpy as np
from cv2 import Rodrigues


def gaze3Dto2D(vectors, stack=True):
    """
    theta = asin(-y) -- pitch
    phi = atan2(-x, -z) -- yaw
    """
    if vectors.ndim == 2:
        assert vectors.shape[1] == 3
        x, y, z = (vectors[:, i] for i in range(3))
    elif vectors.ndim == 1:
        assert vectors.shape[0] == 3
        x, y, z = (vectors[i] for i in range(3))

    pitch = np.arcsin(-y)  # pitch
    yaw = np.arctan2(-x, -z)  # yaw

    if not stack:
        return yaw, pitch
    elif stack:
        return np.column_stack((yaw, pitch))


def gaze2Dto3D(angles):
    """
    x = (-1) * cos(theta) * sin(phi)
    y = (-1) * sin(theta)
    z = (-1) * cos(theta) * cos(phi)
    """
    if angles.ndim == 2:
        assert angles.shape[1] == 2
        yaw, pitch = (angles[:, i] for i in range(2))
    elif angles.ndim == 1:
        assert angles.shape[0] == 2
        yaw, pitch = (angles[i] for i in range(2))

    x = (-1) * np.cos(pitch) * np.sin(yaw)
    y = (-1) * np.sin(pitch)
    z = (-1) * np.cos(pitch) * np.cos(yaw)

    vectors = np.column_stack((x, y, z))
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors / norm


def pose3Dto2D(vectors):
    """
    M = Rodrigues((x,y,z))
    Zv = (the third column of M)
    theta = asin(Zv[1])
    phi = atan2(Zv[0], Zv[2])
    """

    Zv = np.apply_along_axis(
        lambda vector: Rodrigues(vector)[0][:, 2],
        axis=1,
        arr=vectors.astype(np.float32)
    ).T

    pitch = np.arcsin(Zv[1])  # pitch
    yaw = np.arctan2(Zv[0], Zv[2])  # yaw

    return np.column_stack((yaw, pitch))
