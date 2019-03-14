#!/usr/bin/env python

import rospy
import cv_bridge
import cv2
import numpy as np
import imutils
from sensor_msgs.msg import Joy, LaserScan, Image
from std_msgs.msg import Bool, Int32, String

start = False
finished = False
shape_count = {"circle": 0, "triangle": 0, "square": 0}


def image_callback(msg):
    global bridge, start, finished
    global shape_count, end_time, shape_pub, count_pub
    if start:
        image = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

        lower_red = np.array([150, 150, 80])
        upper_red = np.array([360, 256, 225])

        h, w, d = image.shape
        mask = cv2.inRange(hsv, lower_red, upper_red)
        search_bot = 1*h/5

        # mask[0:search_top, 0:w] = 0
        mask[search_bot:h, 0:w] = 0

        count = countRed(mask)
        if count > 0:
            shape_pub.publish(Bool(True))
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
    areas = [c for c in cnts if cv2.contourArea(c) > 100]
    return len(areas)


def ready_callback(msg):
    global start
    if msg.data:
        start = True
    else:
        start = False


rospy.init_node("findshape4")
bridge = cv_bridge.CvBridge()
image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, image_callback)
start_sub = rospy.Subscriber('/startShape4', Bool, ready_callback)
start_time = rospy.Time.now()
end_time = rospy.Time.now() + rospy.Duration(10)

shape_pub = rospy.Publisher("hasShape4", Bool, queue_size=1)


rospy.spin()
