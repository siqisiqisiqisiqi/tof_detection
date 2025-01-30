#!/home/grail/.virtualenvs/yolo_keypoint/bin/python
import rospy
from std_srvs.srv import SetBool
from orbbec_camera.srv import SetInt32


def set_camera_exposure(exposure_value):
    # Wait for the service to be available
    rospy.wait_for_service('/camera/set_color_exposure')
    try:
        # Create a service proxy
        set_exposure = rospy.ServiceProxy(
            '/camera/set_color_exposure', SetInt32)
        # Call the service with the desired exposure value
        response = set_exposure(exposure_value)
        print(f"Service call successful: {response}")
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")


def set_camera_gain(gain_value):
    # Wait for the service to be available
    rospy.wait_for_service('/camera/set_color_gain')
    try:
        # Create a service proxy
        set_gain = rospy.ServiceProxy(
            '/camera/set_color_gain', SetInt32)
        # Call the service with the desired exposure value
        response = set_gain(gain_value)
        print(f"Service call successful: {response}")
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")


def set_auto_exposure(auto_exposure):
    # Wait for the service to be available
    rospy.wait_for_service('/camera/set_color_auto_exposure')
    try:
        # Create a service proxy
        set_auto_exposure_service = rospy.ServiceProxy(
            '/camera/set_color_auto_exposure', SetBool)
        # Call the service with the desired auto exposure value
        response = set_auto_exposure_service(auto_exposure)
        rospy.loginfo(f"Service call successful: {response}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")


if __name__ == "__main__":
    # Initialize the ROS node
    rospy.init_node('camera_setting')

    # Desired exposure value with light on
    # auto_exposure = False
    # exposure_value = 90
    # set_auto_exposure(auto_exposure)
    # set_camera_exposure(exposure_value)

    # Desired exposure value with light scenario 7
    auto_exposure = False
    exposure_value = 250
    gain_value = 65
    set_auto_exposure(auto_exposure)
    set_camera_exposure(exposure_value)
    set_camera_gain(gain_value)
