import cv2
import numpy as np


def kpts_visiual(img, points, pro_points):
    """keypoint visualization in the 2D image

    Parameters
    ----------
    img : ndarray
        RGB image array
    points : ndarray [num_object, 3, 3]
        points pixel in the image frame
    pro_points : ndarray [num_object, 3]
        mid point in the world coordiante frame
    """
    grasp_points = points[:,1,:2].astype(int)
    for i, grasp_point in enumerate(grasp_points):
        pro_point = np.round(pro_points[i])
        text = f"({pro_point[0]},{pro_point[1]},{pro_point[2]})"
        cv2.circle(img, grasp_point, radius=5, color=(0, 255, 0), thickness=-1)
        cv2.putText(img, text, (grasp_point[0] + 10, grasp_point[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 255), thickness=1)
        
    cv2.imshow("image", img)
    k = cv2.waitKey(5)
    return k


def optimal_kpts_visual(img, grasp_point, optimal_item):
    """optimal keypoint visualization

    Parameters
    ----------
    img : ndarray
        _description_
    grasp_point : grasp point in the image frame
    optimal_item : optimal point in the world frame

    Returns
    -------
        visual input
    """
    text = f"({optimal_item[0]},{optimal_item[1]},{optimal_item[2]})"
    cv2.circle(img, grasp_point, radius=5, color=(0, 255, 0), thickness=-1)
    cv2.putText(img, text, (grasp_point[0] + 10, grasp_point[1] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 255), thickness=1)

    cv2.imshow("image", img)
    k = cv2.waitKey(5)
    return k
