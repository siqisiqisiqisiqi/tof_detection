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

        # Init subscribers
        rospy.Subscriber("camera/color/image_raw", Image, self.get_image)

        # Init subscribers
        rospy.Subscriber("camera/depth/image_raw", Image, self.get_depth)

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

    def run(self):

        while not rospy.is_shutdown():
            if self.cv_image is None or self.cv_depth is None:
                rospy.loginfo("Waiting for camera initialization...")
                rospy.sleep(1)
                continue

            img = np.copy(self.cv_image)
            depth = np.array(self.cv_depth, dtype=np.float32)
            rospy.loginfo(f"depth shape: {depth.shape}")
            
            cv2.imshow('rgb_image', img)
            k = cv2.waitKey(5)

            if k == ord('s'):
                cv2.imwrite(f'{PARENT_DIR}/test/test.jpg', img)
                np.save(f'{PARENT_DIR}/test/test.npy', depth)

            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = Camera()
        node.run()
    except rospy.ROSInterruptException:
        pass

    cv2.destroyAllWindows()
    rospy.loginfo("Finish the code!")
