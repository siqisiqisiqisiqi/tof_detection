import cv2


def coordinates_draw(img, corners, imgpts):
    corner = tuple(corners[0].astype(int).ravel())
    cv2.line(img, corner, tuple(
        imgpts[0].astype(int).ravel()), (255, 0, 0), 5)
    cv2.line(img, corner, tuple(
        imgpts[1].astype(int).ravel()), (0, 255, 0), 5)
    cv2.line(img, corner, tuple(
        imgpts[2].astype(int).ravel()), (0, 0, 255), 5)
    return img
