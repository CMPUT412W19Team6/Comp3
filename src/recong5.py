#!/usr/bin/env python

import rospy
import cv_bridge
import cv2
import numpy as np
import imutils
from sensor_msgs.msg import Joy, LaserScan, Image
from std_msgs.msg import Bool, Int32, String

start = False
shape_count = {"circle": 0, "triangle": 0, "square": 0}


def image_callback(msg):
    global bridge, start
    global shape_count, end_time, shape_pub
    if start:
        # if True:
        image = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

        lower_red = np.array([150, 150, 80])
        upper_red = np.array([360, 256, 225])

        mask = cv2.inRange(hsv, lower_red, upper_red)

        h, w, d = image.shape
        search_top = 1*h/6
        search_bot = 3*h/4

        mask[0:search_top, 0:w] = 0
        mask[search_bot:h, 0:w] = 0
        M = cv2.moments(mask)

        getShape(mask)
        cv2.imshow("window", mask)
        cv2.waitKey(3)

        if end_time < rospy.Time.now():
            shape = max(shape_count, key=lambda k: shape_count[k])
            print("Found SHAPE=", shape)

            shape_count["triangle"] = 0
            shape_count["square"] = 0
            shape_count["circle"] = 0
            start = False

            shape_pub.publish(String(shape))


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


def getShape(image):
    global shape_count
    # from https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    print("Start get shape")
    if len(cnts) > 0:
        index = np.argmax(np.array([cv2.contourArea(c) for c in cnts]))
        c = cnts[index]
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) <= 3:
            shape_count["triangle"] += 1
        if 4 <= len(approx):
            shape_count["square"] += 1
        if len(approx) >= 5:
            shape_count["circle"] += 1
        # print(shape, len(cnts), cv2.contourArea(c))
        print(len(approx), cv2.contourArea(c))
    print("End get shape")


def ready_callback(msg):
    global start
    global start_time, end_time
    if msg.data:
        start = True
        start_time = rospy.Time.now()
        end_time = rospy.Time.now() + rospy.Duration(6)


rospy.init_node("recong5")
bridge = cv_bridge.CvBridge()
image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, image_callback)
start_sub = rospy.Subscriber('/recong5_start', Bool, ready_callback)
shape_pub = rospy.Publisher("/shape5", String, queue_size=10)
start_time = rospy.Time.now()
end_time = rospy.Time.now() + rospy.Duration(6)

rospy.spin()
