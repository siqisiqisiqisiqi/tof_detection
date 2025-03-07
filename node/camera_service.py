#!/home/grail/.virtualenvs/yolo_keypoint/bin/python
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)

import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32MultiArray
from ultralytics import YOLO

from src.point_util import yolo2point, depth_acquisit, img2world, priority_evaluate, point_trans, object_filter, point_offset
from src.visual import kpts_visiual, optimal_kpts_visual, img_visual
from tof_detection.srv import EthObjectDetection, EthObjectDetectionResponse


class Camera:
    def __init__(self):
        rospy.init_node("camera", anonymous=True)

        # Init variable
        self.rate = rospy.Rate(10)
        self.bridge = CvBridge()
        self.cv_image = None
        self.cv_depth = None
        self.image_t = None
        self.depth_t = None
        self.save_index = 0

        # YOLO init
        self.model = YOLO(f"{PARENT_DIR}/weights/bunched_rgb_kpts.pt")

        # Warm-up the model
        results = self.model(f"{PARENT_DIR}/test/test.jpg", conf=0.2)

        # Init subscribers
        rospy.Subscriber("camera/color/image_raw", Image, self.get_image)
        rospy.Subscriber("camera/depth/image_raw", Image, self.get_depth)

        # Create the service
        self.service = rospy.Service(
            "eth_detect_object", EthObjectDetection, self.eth_detect_objects)

    def get_image(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.image_t = 1e-9 * data.header.stamp.nsecs + \
                data.header.stamp.secs
        except CvBridgeError as e:
            print(e)

    def get_depth(self, data):
        try:
            self.cv_depth = self.bridge.imgmsg_to_cv2(data, "16UC1")
            self.depth_t = 1e-9 * data.header.stamp.nsecs + \
                data.header.stamp.secs
        except CvBridgeError as e:
            print(e)

    def eth_detect_objects(self, seq):
        rospy.loginfo("Received object detection request...")
        img = np.copy(self.cv_image)
        depth_img = np.array(self.cv_depth, dtype=np.float32)
        results = self.model(img, conf=0.2)
        try:
            points, _, num_object = yolo2point(results)

            # points = point_offset(points)
            depths, _ = depth_acquisit(points, depth_img)
            pro_points, angles, pro_result = img2world(points, depths)

            # priority calculation
            yaws = angles * np.pi / 180
            ave_depth = np.mean(depths.reshape(-1, 3), axis=1)
            index = object_filter(pro_points)
            optimal_obj, optimal_yaw, optimal_index = priority_evaluate(
                ave_depth[index], pro_points[index], yaws[index], pro_result[index])

            # send the optimal obj with orientation [x(mm), y(mm), z(mm), yaw (radian)]
            optimal_obj_data = point_trans(optimal_obj, optimal_yaw)
            x,y,z,yaw = optimal_obj_data
            return EthObjectDetectionResponse(x, y, z, yaw)
        except:
            rospy.loginfo("Problem happens for the service.")
            return EthObjectDetectionResponse(0, 0, 0, 0)

    def run(self):

        while not rospy.is_shutdown():
            if self.cv_image is None or self.cv_depth is None:
                rospy.loginfo("Waiting for camera initialization...")
                rospy.sleep(1)
                continue

            img = np.copy(self.cv_image)
            depth_img = np.array(self.cv_depth, dtype=np.float32)
            cv2.imshow("image", img)
            k = cv2.waitKey(5)
            if k == ord('q'):
                break
            if k == ord('s'):
                cv2.imwrite(
                    f'{PARENT_DIR}/test/test{self.save_index}.jpg', img)
                np.save(
                    f'{PARENT_DIR}/test/test{self.save_index}.npy', depth_img)
                self.save_index = self.save_index + 1

            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = Camera()
        node.run()
    except rospy.ROSInterruptException:
        pass

    cv2.destroyAllWindows()
    rospy.loginfo("Finish the code!")
