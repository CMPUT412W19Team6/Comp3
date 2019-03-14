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
    global start, bridge, count_pub
    if start:
        rospy.sleep(rospy.Duration(2))
        image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red = np.array([150, 150, 80])
        upper_red = np.array([360, 256, 225])

        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        object_count = 0

        # cv2.imshow("window", mask_red)
        gray = mask_red
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        areas = [c for c in cnts if cv2.contourArea(c) > 300]
        object_count = len(areas)

        count_pub.publish(Int32(object_count))


def ready_callback(msg):
    global start
    print("ready recieved for count1", msg, msg.data)
    if msg.data:
        start = True
    else:
        start = False


rospy.init_node("count1")
bridge = cv_bridge.CvBridge()
image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, image_callback)
start_sub = rospy.Subscriber('/start1', Bool, ready_callback)
start_time = rospy.Time.now()
end_time = rospy.Time.now() + rospy.Duration(10)

count_pub = rospy.Publisher("count1", Int32, queue_size=1)

rospy.spin()
