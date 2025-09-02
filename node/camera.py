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

        # Init subscribers
        rospy.Subscriber("camera/color/image_raw", Image, self.get_image)
        rospy.Subscriber("camera/depth/image_raw", Image, self.get_depth)

        # Create a publisher object
        self.pub = rospy.Publisher(
            'grasp_pose', Float32MultiArray, queue_size=10)

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
            depth_img = np.array(self.cv_depth, dtype=np.float32)
            results = self.model(img, conf=0.2)

            if results[0].keypoints.conf is None:
                # img_visual(img)
                rospy.loginfo(f"detect confidence is not enough")
                continue

            try:
                points, _, num_object = yolo2point(results)
                if num_object == 0:
                    # img_visual(img)
                    rospy.loginfo(f"key point confidence is not enough")
                    continue
                # points = point_offset(points)
                depths, _ = depth_acquisit(points, depth_img)
                pro_points, angles, pro_result = img2world(points, depths)

                # priority calculation
                yaws = angles * np.pi / 180
                ave_depth = np.mean(depths.reshape(-1, 3), axis=1)
                index = object_filter(pro_points)
                if sum(index) == 0:
                    # img_visual(img)
                    continue

                optimal_obj, optimal_yaw, optimal_index = priority_evaluate(
                    ave_depth[index], pro_points[index], yaws[index], pro_result[index])

                # point transformation

                # send the optimal obj with orientation [x(mm), y(mm), z(mm), yaw (radian)]
                optimal_obj_data = point_trans(optimal_obj, optimal_yaw)
                optimal_obj_msg = Float32MultiArray()
                optimal_obj_msg.data = optimal_obj_data
                self.pub.publish(optimal_obj_msg)

                # # optimal visualization
                points = points[index]
                grasp_point = points[optimal_index, 1, :2].astype(int)

                k = optimal_kpts_visual(img, grasp_point, optimal_obj, optimal_yaw)
                if k == ord('q'):
                    break
                if k == ord('s'):
                    cv2.imwrite(
                        f'{PARENT_DIR}/test/test{self.save_index}.jpg', self.cv_image)
                    np.save(
                        f'{PARENT_DIR}/test/test{self.save_index}.npy', depth_img)
                    self.save_index = self.save_index + 1

                # visualization all the projected results
                # k = kpts_visiual(img, points, pro_points)
                # if k == ord('q'):
                #     break

                # visualize the yolo keypoint result
                # for r in results:
                #     im_array = r.plot()
                #     cv2.imshow("image", im_array)
                #     k = cv2.waitKey(5)

            except Exception as e:
                rospy.loginfo(e)
                cv2.imwrite(
                    f'{PARENT_DIR}/test/test{self.save_index}.jpg', img)
                np.save(
                    f'{PARENT_DIR}/test/test{self.save_index}.npy', depth_img)
                self.save_index = self.save_index + 1
                break

            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = Camera()
        node.run()
    except rospy.ROSInterruptException:
        pass

    cv2.destroyAllWindows()
    rospy.loginfo("Finish the code!")
