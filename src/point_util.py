import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

import numpy as np
from numpy.linalg import inv

chess_size = 1  # mm
extrinsic_param = os.path.join(PARENT_DIR, 'params/E.npz')

with np.load(extrinsic_param) as X:
    mtx, dist, Mat, tvecs = [X[i] for i in ('mtx', 'dist', 'Matrix', 'tvec')]

tvec = tvecs * chess_size
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]


def yolo2point(yolo_pose_results):
    """Extract the yolo pose estimation information

    Parameters
    ----------
    yolo_pose_results : ultralytics return
        yolo pose result

    Returns
    -------
    points: ndarray
        the second and third keypoints in homogeneous format
    visual: ndarray
        the second and third keypoints visualbility
    """
    confidence_thresdhold = 0.8
    confidence = yolo_pose_results[0].keypoints.conf.detach().cpu().numpy()
    conf_mean = np.mean(confidence, axis=-1)
    indices = np.where(conf_mean > confidence_thresdhold)
    keypoints = yolo_pose_results[0].keypoints.data.detach().cpu().numpy()
    num_object = len(indices[0])
    points = keypoints[indices[0], :, :2]
    visual = keypoints[indices[0], :, 2]

    one_vector = np.ones(points.shape[:2])
    one_vector = np.expand_dims(one_vector, axis=2)
    points = np.concatenate((points, one_vector), axis=2)
    return points, visual


def depth_acquisit(points, depth_img, depth_acquisit_scale=[1, 2, 3]):
    dim = points.shape[-1]
    pts = points.reshape((-1, dim))
    depths = np.zeros(pts.shape[0])

    # depth_img = depth_img
    depth_trans = depth_transform(depth_img)
    rows, cols = depth_trans.shape

    for i, point in enumerate(pts):
        y, x = point[:2].astype(int)
        for scale in depth_acquisit_scale:

            row_start = max(0, x - scale)
            row_end = min(rows, x + scale + 1)
            col_start = max(0, y - scale)
            col_end = min(cols, y + scale + 1)

            neighborhood = depth_trans[row_start:row_end, col_start:col_end]

            non_zero_values = neighborhood[neighborhood != 0]
            if non_zero_values.size > 0:
                depth = np.mean(non_zero_values)  # Use mean of non-zero values
                break
            else:
                depth = 0
                continue
        depths[i] = depth
    return depths, depth_trans


def depth_transform(depth_image):
    h, w = depth_image.shape
    grid = np.mgrid[0:h, 0:w]
    v, u = grid[0], grid[1]
    x = (u - cx) * depth_image / fx
    y = (v - cy) * depth_image / fy
    pc_mesh = np.stack((x, y, depth_image), axis=2)
    pc_mesh_t = pc_mesh.transpose((2, 0, 1))
    test = pc_mesh_t.reshape((3, -1))
    test = inv(Mat) @ (test - tvecs)
    test = test.reshape((3, h, w))
    depth_trans = test[-1, :, :]
    # depth_trans[np.abs(depth_trans) > 0.5] = 0
    depth_trans[np.abs(depth_trans) > 200] = 0
    depth_trans = np.nan_to_num(depth_trans, nan=0)
    # depth_trans = depth_trans * 1000
    # depth_trans = depth_trans * -1
    return depth_trans


def projection(points, mtx, Mat, tvecs, depth=0):
    """camera back projection

    Parameters
    ----------
    points : ndarray
        keypoints for camera back projection
    mtx : ndarray
        camera intrinsic parameter
    Mat : ndarray
        camera extrinsic parameter: rotation matrix
    tvecs : ndarray
        camera extrinsic parameter: translation vector
    depth : int, optional
        depth of the point in the world coordinate frame, by default 0

    Returns
    -------
    result: ndarray
        point in the world coordinate frame
    """
    num_object = points.shape[0]
    results = np.zeros_like(points)
    for i in range(num_object):
        point = points[i, :].reshape(3, 1)
        point2 = inv(mtx) @ point
        predefined_z = depth[i // 2]
        vec_z = Mat[:, [2]] * predefined_z
        Mat2 = np.copy(Mat)
        Mat2[:, [2]] = -1 * point2
        vec_o = -1 * (vec_z + tvecs)
        result = inv(Mat2) @ vec_o
        results[i] = result.squeeze()
    return results


def img2world(point, depth):
    """project the keypoint from camera frame to world coordinate frame

    Parameters
    ----------
    point : ndarray
        THe second and third keypoints
    depth : ndarray
        Depth in the world cooridnate frame

    Returns
    -------
    grap_point: ndarray
        grasp point in the world coordinate frame 
    orientation: ndarray
        green onion orientation in the world coordinate frame
    """
    point = point.reshape((-1, 3))
    # depth = depth / 1000
    depth = np.round(depth, 2)
    result = projection(point, mtx, Mat, tvec, depth)
    result = np.round(result, 2)
    result = result.reshape((-1, 3, 3))
    grasp_point = result[:, 1, :]
    p1 = result[:, 0, :]
    p2 = result[:, 2, :]
    dy = p2[:, 1] - p1[:, 1]
    dx = p2[:, 0] - p1[:, 0]
    orientation = np.arctan2(dy, dx) * 180 / np.pi
    grasp_point[:, -1] = depth[1::3]
    return grasp_point, orientation
