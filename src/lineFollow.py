#!/usr/bin/env python

import rospy
import cv_bridge
import cv2
import numpy as np
import imutils
from sensor_msgs.msg import Joy, LaserScan, Image
from std_msgs.msg import Bool, Int32, String
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from math import pi

start = False
finished = False
shape_count = {"circle": 0, "triangle": 0, "square": 0}


def image_callback(msg):
    global bridge, start, finished
    global shape_count, end_time, shape_pub, count_pub

    if start:
        image = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

        lower_green = np.array([50, 100, 100])
        upper_green = np.array([120, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        count = countRed(mask)
        if count > 0:
            shape_pub.publish(Bool(True))
            print("Found green")
        else:
            shape_pub.publish(Bool(False))


def countRed(image):
    global shape_count
    # from https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    areas = [c for c in cnts if cv2.contourArea(c) > 400]
    return len(areas)


def ready_callback(msg):
    global start
    print("start finding green")
    if msg.data:
        start = True
    else:
        start = False


def odom_callback(msg):
    global shape_pub, start

    if start:
        pose = msg.pose.pose.orientation
        euler = euler_from_quaternion((pose.x, pose.y, pose.z, pose.w))

        upper_range = -70*pi/180
        lower_range = -110*pi/180
        print(euler[2], "euler")
        if euler[2] > lower_range and euler[2] < upper_range:
            shape_pub.publish(Bool(True))
            print("target angel hit")
        else:
            shape_pub.publish(Bool(False))


rospy.init_node("findshape2")
bridge = cv_bridge.CvBridge()
# image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, image_callback)
start_sub = rospy.Subscriber('/green_start', Bool, ready_callback)
odom_sub = rospy.Subscriber('odom', Odometry, callback=odom_callback)
start_time = rospy.Time.now()
end_time = rospy.Time.now() + rospy.Duration(10)

shape_pub = rospy.Publisher("hasShape2", Bool, queue_size=1)


rospy.spin()
