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
        lower_green = np.array([50, 100, 100])
        upper_green = np.array([120, 255, 255])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        maskGreen = cv2.inRange(hsv, lower_green, upper_green)

        h, w, d = image.shape
        search_top = 1*h/2
        search_bot = 3*h/4 + 20

        # mask[0:search_top, 0:w] = 0
        mask[search_bot:h, 0:w] = 0
        maskGreen[search_bot:h, 0:w] = 0

        getShape(maskGreen)
        count = countRed(mask)
        # cv2.imshow("window", mask)
        cv2.waitKey(3)

        if end_time < rospy.Time.now():
            shape = max(shape_count, key=lambda k: shape_count[k])
            print("FINAL SHAPE=", shape)
            shape_pub.publish(String(shape))
            print("TOTAL_COUNT=", count+1)
            count_pub.publish(Int32(count+1))
            print("end timer")
            start = False
            finished = True


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

    if len(cnts) > 0:
        index = np.argmax(np.array([cv2.contourArea(c) for c in cnts]))
        c = cnts[index]
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) <= 3:
            shape_count["triangle"] += 1
        if 4 <= len(approx) and len(approx) <= 5:
            shape_count["square"] += 1
        if len(approx) > 5:
            shape_count["circle"] += 1
        # print(shape, len(cnts), cv2.contourArea(c))
        print(len(approx), cv2.contourArea(c))


def ready_callback(msg):
    global start, finished
    global start_time, end_time
    print("ready recieved", msg, msg.data)
    if msg.data and not finished:
        start = True
        start_time = rospy.Time.now()
        end_time = rospy.Time.now() + rospy.Duration(10)


rospy.init_node("count2")
bridge = cv_bridge.CvBridge()
image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, image_callback)
start_sub = rospy.Subscriber('/start2', Bool, ready_callback)
start_time = rospy.Time.now()
end_time = rospy.Time.now() + rospy.Duration(10)

shape_pub = rospy.Publisher("shape2", String, queue_size=1)
count_pub = rospy.Publisher("count2", Int32, queue_size=1)

rospy.spin()
