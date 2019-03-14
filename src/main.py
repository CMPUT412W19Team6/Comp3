#!/usr/bin/env python

import rospy
import cv2
import cv_bridge
from smach import State, StateMachine
import smach_ros
from dynamic_reconfigure.server import Server
from comp3.cfg import Comp3Config
from ar_track_alvar_msgs.msg import AlvarMarkers
from geometry_msgs.msg import Twist, Pose, PoseStamped, PointStamped
from nav_msgs.msg import Odometry
from kobuki_msgs.msg import BumperEvent, Sound, Led
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import decompose_matrix, compose_matrix, quaternion_from_euler
from ros_numpy import numpify
import actionlib
from sensor_msgs.msg import Joy, LaserScan, Image
import numpy as np
import angles as angles_lib
import math
import random
from std_msgs.msg import Bool, String, Int32
import imutils
from copy import deepcopy

START = False    
FORWARD_CURRENT = 0
TURN_CURRENT = 0
POSE = [0, 0, 0, 0]
turn_direction = 1
PHASE = None
SHAPE = "circle"
SHAPE_MATCHED = False
NUM_SHAPES = 0
CURRENT_CHECKPOINT = 0
UNKNOWN_CHECKPOINT = 0
PHASE4_TASK_COMPLETED = 0

# TODO: update POSE from callback


class WaitForButton(State):
    def __init__(self):
        State.__init__(self, outcomes=["pressed", "exit"])
        self.rate = rospy.Rate(10)

    def execute(self, userdata):
        global START
        while not START and not rospy.is_shutdown():
            self.rate.sleep()
        if rospy.is_shutdown():
            return "exit"
        else:
            return "pressed"


class FollowLine(State):
    """
    should follow the white line, until see full red
    """

    def __init__(self, phase="1.0"):
        State.__init__(self, outcomes=[
                       "see_red", "exit", "failure", "see_nothing", "see_long_red"])
        self.image_sub = rospy.Subscriber(
            '/usb_cam/image_raw', Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.bridge = cv_bridge.CvBridge()
        self.phase = phase
        self.green_start_pub = rospy.Publisher(
            'green_start', Bool, queue_size=1)
        self.green_sub = rospy.Subscriber('hasShape2', Bool, self.green_callback)
        self.find_green = False
        self.reset()

    def reset(self):
        self.twist = Twist()
        self.found_object = False
        self.start_timeout = False
        self.temporary_stop = False
        self.image_received = False
        self.white_line_ended = False
        self.image = None
        self.object_area = 0
        self.dt = 1.0 / 20.0

    def green_callback(self, msg):
        self.find_green = msg.data

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        

    def execute(self, userdata):
        global RED_VISIBLE, PHASE, FORWARD_CURRENT, TURN_CURRENT, linear_vel, red_timeout, Kp, Kd, Ki
        FORWARD_CURRENT = 0.0
        TURN_CURRENT = 0.0
        previous_error = 0
        integral = 0
        sleep_duration = rospy.Duration(self.dt, 0)
        PHASE = self.phase

        self.reset()
        start_time = None

        while not rospy.is_shutdown() and START:
            if self.image is None:
                continue

            image = deepcopy(self.image)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

            lower_white = np.array([white_min_h, white_min_s, white_min_v])
            upper_white = np.array([white_max_h, white_max_s, white_max_v])

            mask = cv2.inRange(hsv, lower_white, upper_white)

            hsv2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            lower_red = np.array([red_min_h,  red_min_s,  red_min_v])
            upper_red = np.array([red_max_h, red_max_s, red_max_v])

            mask_red = cv2.inRange(hsv2, lower_red, upper_red)

            lower_green = np.array([108, 68, 100])
            upper_green = np.array([200, 255, 255])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)

            cx = cy = w = None
            h, w, d = image.shape
            search_top = 3*h/4
            search_bot = 3*h/4 + 20

            mask[0:search_top, 0:w] = 0
            mask[search_bot:h, 0:w] = 0
            M = cv2.moments(mask)

            if M['m00'] > 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.circle(image, (cx, cy), 20, (0, 0, 255), -1)

            if self.phase == "4.2" and M['m00'] == 0:  # no more white line ahead
                self.start_timeout = True
            elif self.phase == "2.1":
                if self.find_green and M['m00'] == 0:
                    self.start_timeout = True
            elif self.phase != "4.2":
                max_area = 0
                # if self.phase == "2.1":  # calculate sum of area of contours of red and green shapes
                #     mask_red[h/2:h, 0:w] = 0

                #     im2, contours, hierarchy = cv2.findContours(
                #         mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #     # total_area += sum([cv2.contourArea(x) for x in contours])

                # else:  # calculate sum of area of contours of red tapes
                # mask_red[0:search_top, 0:w] = 0
                if self.phase == "3.1":
                    mask_red[0:h/2, 0:w] = 0

                im2, contours, hierarchy = cv2.findContours(
                    mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # total_area = sum([cv2.contourArea(x) for x in contours])
                contours = [x for x in contours if cv2.contourArea(x) > 10]

                if len(contours) > 0:  # Keep the maximum sum of area of red/green objects since first detection
                    self.found_object = True
                    max_area = max([cv2.contourArea(x) for x in contours])
                    self.object_area = max(self.object_area, max_area)

                # red objects no more visible since last detection
                if len(contours) == 0 and self.found_object:
                    self.found_object = False
                    thresh = 1000
                    if self.phase=="2.2":
                        thresh = 1000
                    print(self.object_area)
                    if self.object_area < red_area_threshold and self.object_area > thresh:  # valid small red object in front
                        if self.phase == "4.1":
                            self.temporary_stop = True
                        else:
                            self.start_timeout = True
                    elif self.object_area > red_area_threshold:  # valid large red object in front
                        self.temporary_stop = True
                    self.object_area = 0
                # elif PHASE == "2.1" and self.found_object:
                #     print(self.object_area)
                #     self.found_object = False
                #     if self.object_area > 800:
                #         self.start_timeout = True
                #     self.object_area = 0

            cv2.imshow("window", mask_red)
            cv2.waitKey(3)

            if self.phase == "2.1":
                self.green_start_pub.publish(Bool(True))
            # if self.phase == "2.1":
            #     Kp = 1.0 / 350.0
            #     Kd = 1.0 / 700.0
            # else:
            #     Kp = 1.0 / 400.0
            #     Kd = 1.0 / 700.0


            if self.temporary_stop:
                rospy.sleep(rospy.Duration(2.0))
                self.temporary_stop = False
                print("temp stoped")
                if self.phase=="4.1":
                    self.start_timeout = True

            if self.start_timeout and start_time is None:
                print("?????")
                start_time = rospy.Time.now()

            r_timeout = red_timeout
            if self.phase=="2.2":
                r_timeout = 1.5
            if self.start_timeout and start_time + r_timeout < rospy.Time.now():
                start_time = None
                self.start_timeout = False
                # image_sub.unregister()

                if self.phase == "4.1":
                    return "see_long_red"
                elif self.phase == "4.2":
                    return "see_nothing"
                elif self.phase == "2.1":
                    self.green_start_pub.publish(Bool(False))
                    return "see_red"
                else:
                    return "see_red"

            # BEGIN CONTROL
            if cx is not None and w is not None:
                error = float(cx - w/2.0)
                integral += error * self.dt
                derivative = (error - previous_error) / self.dt

                # FORWARD_CURRENT = linear_vel
                # TURN_CURRENT = - (Kp * error + Kd * derivative + Ki * integral)
                # move(FORWARD_CURRENT, TURN_CURRENT, self.cmd_vel_pub, 0.3)

                self.twist.linear.x = linear_vel
                self.twist.angular.z = - \
                    (Kp * float(error) + Kd * derivative + Ki * integral)
                self.cmd_vel_pub.publish(self.twist)

                previous_error = error

                rospy.sleep(sleep_duration)
            # END CONTROL

        # image_sub.unregister()
        if not START:
            return "exit"


def calc_delta_vector(start_heading, distance):
    dx = distance * np.cos(start_heading)
    dy = distance * np.sin(start_heading)
    return np.array([dx, dy])


def check_forward_distance(forward_vec, start_pos, current_pos):
    current = current_pos - start_pos
    # vector projection (project current onto forward_vec)
    delta = np.dot(current, forward_vec) / \
        np.dot(forward_vec, forward_vec) * forward_vec
    dist = np.sqrt(delta.dot(delta))
    return dist


class MoveBaseGo(State):
    def __init__(self, distance = 0, yaw = 0):
        State.__init__(self, outcomes=["success", "exit", 'failure'])
        self.distance = distance
        self.yaw = yaw
        self.move_base_client = actionlib.SimpleActionClient(
            "move_base", MoveBaseAction)

    def execute(self, userdata):
        if START and not rospy.is_shutdown():

            quaternion = quaternion_from_euler(0, 0, self.yaw)

            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "base_footprint"
            goal.target_pose.pose.position.x = self.distance
            goal.target_pose.pose.position.y = 0
            goal.target_pose.pose.orientation.x = quaternion[0]
            goal.target_pose.pose.orientation.y = quaternion[1]
            goal.target_pose.pose.orientation.z = quaternion[2]
            goal.target_pose.pose.orientation.w = quaternion[3]
            
            self.move_base_client.send_goal_and_wait(goal)

            return "success"


class Translate(State):
    def __init__(self, distance=0.15, linear=-0.2):
        State.__init__(self, outcomes=["success", "exit", 'failure'])
        self.tb_position = None
        self.tb_rot = [0,0,0,0]
        self.distance = distance
        self.COLLISION = False
        self.linear = linear

        # pub / sub
        self.cmd_pub = rospy.Publisher(
            "cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("odom", Odometry, callback=self.odom_callback)

    def odom_callback(self, msg):
        tb_pose = msg.pose.pose
        __, __, angles, position, __ = decompose_matrix(numpify(tb_pose))
        self.tb_position = position[0:2]
        self.tb_rot = angles

    def execute(self, userdata):
        global turn_direction
        global START
        if not START:
            return 'quit'
        self.COLLISION = False
        start_heading = self.tb_rot[2]
        start_pos = self.tb_position
        forward_vec = calc_delta_vector(start_heading, self.distance)
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            dist = check_forward_distance(
                forward_vec, start_pos, self.tb_position)
            if dist > self.distance:
                return "success"
            if self.linear > 0 and self.COLLISION:
                msg = Twist()
                msg.linear.x = 0.0
                self.cmd_pub.publish(msg)
                self.COLLISION = False
                # hit something while driving perpendicular so swap turn direction
                turn_direction = -1 * turn_direction
                return "collision"

            msg = Twist()
            msg.linear.x = self.linear
            self.cmd_pub.publish(msg)
            rate.sleep()


class Turn(State):
    """
    Turning a specific angle, based on Sean's example code from demo2
    """

    def __init__(self, angle=90):
        State.__init__(self, outcomes=["success", "exit", 'failure'])
        self.tb_position = None
        self.tb_rot = None
        # angle defines angle target relative to goal direction
        self.angle = angle

        # pub / sub
        rospy.Subscriber("odom", Odometry, callback=self.odom_callback)
        self.cmd_pub = rospy.Publisher(
            "cmd_vel", Twist, queue_size=1)

    def odom_callback(self, msg):
        tb_pose = msg.pose.pose
        __, __, angles, position, __ = decompose_matrix(numpify(tb_pose))
        self.tb_position = position[0:2]
        self.tb_rot = angles

    def execute(self, userdata):
        global turn_direction
        global START
        global POSE

        if not START:
            return 'exit'
        start_pose = POSE
        if self.angle == 0:  # target is goal + 0
            goal = start_pose[1]
        elif self.angle == 90:  # target is goal + turn_direction * 90
            goal = start_pose[1] + np.pi/2 * turn_direction
        elif self.angle == 180:  # target is goal + turn_direction * 180
            goal = start_pose[1] + np.pi * turn_direction
        elif self.angle == -90:  # target is goal + turn_direction * 270
            goal = start_pose[1] - np.pi/2 * turn_direction
        elif self.angle == -100:  # target is goal + turn_direction * 270
            goal = start_pose[1] - 5*np.pi/9 * turn_direction
        elif self.angle == 120:
            goal = start_pose[1] + 2*np.pi/3 *turn_direction
        elif self.angle == 135:
            goal = start_pose[1] + 150*np.pi/180 * turn_direction

        goal = angles_lib.normalize_angle(goal)

        cur = np.abs(angles_lib.normalize_angle(self.tb_rot[2]) - goal)
        speed = 0.55
        rate = rospy.Rate(30)

        direction = turn_direction

        if 2 * np.pi - angles_lib.normalize_angle_positive(goal) < angles_lib.normalize_angle_positive(goal) or self.angle == 0:
            direction = turn_direction * -1

        while not rospy.is_shutdown():
            cur = np.abs(angles_lib.normalize_angle(self.tb_rot[2]) - goal)

            # slow down turning as we get closer to the target heading
            if cur < 0.1:
                speed = 0.15
            if cur < 0.0571:
                break
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = direction * speed
            self.cmd_pub.publish(msg)
            rate.sleep()

        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

        return 'success'


class DepthCount(State):
    """
    Count the number of objects based on the depth image
    """

    def __init__(self):
        State.__init__(self, outcomes=["success", "exit", "failure"],
                       output_keys=["object_count"])

        self.bridge = cv_bridge.CvBridge()

        self.count_start = False
        self.object_count = 0
        self.count_finished = False
        self.count1_start_pub = rospy.Publisher('start1', Bool, queue_size=1)

    def execute(self, userdata):
        global START

        self.count_start == False
        self.object_count = 0
        self.count_finished = False

        self.count1_start_pub.publish(Bool(True))
        count1_sub = rospy.Subscriber('count1', Int32, self.count_callback)

        while not rospy.is_shutdown() and START and not self.count_finished:
            pass

        userdata.object_count = self.object_count
        return "success"

        if not START:
            return "exit"

    def count_callback(self, msg):
        self.object_count = msg.data
        self.count1_start_pub.publish(Bool(False))
        self.count_finished = True

    # def image_callback(self, msg):
    #     global RED_VISIBLE, red_area_threshold, white_max_h, white_max_s, white_max_v, white_min_h, white_min_s, white_min_v, red_max_h, red_max_s, red_max_v, red_min_h, red_min_s, red_min_v

    #     if self.count_start:
    #         rospy.sleep(rospy.Duration(2))
    #         image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #         lower_red = np.array([red_min_h,  red_min_s,  red_min_v])
    #         upper_red = np.array([red_max_h, red_max_s, red_max_v])

    #         mask_red = cv2.inRange(hsv, lower_red, upper_red)
    #         self.object_count = 0

    #         # cv2.imshow("window", mask_red)
    #         gray = mask_red
    #         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #         thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    #         cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    #                                 cv2.CHAIN_APPROX_SIMPLE)
    #         cnts = imutils.grab_contours(cnts)
    #         areas = [c for c in cnts if cv2.contourArea(c) > 140]
    #         self.object_count = len(areas)
    #         if len(areas) > 3:
    #             self.object_count = 3
    #         if len(areas) < 1:
    #             self.object_count = 1

    #         self.count_finished = True


class Signal1(State):
    """
    Make sound and LED lights based on count for phase 1
    """

    def __init__(self):
        State.__init__(self, outcomes=["success", "exit", "failure"],
                       input_keys=["object_count"])
        self.led1_pub = rospy.Publisher(
            "/mobile_base/commands/led1", Led, queue_size=1)
        self.led2_pub = rospy.Publisher(
            "/mobile_base/commands/led2", Led, queue_size=1)
        self.sound_pub = rospy.Publisher(
            '/mobile_base/commands/sound', Sound, queue_size=1)

    def execute(self, userdata):
        global START

        print userdata.object_count

        if START and not rospy.is_shutdown():
            if userdata.object_count == 1:
                self.led1_pub.publish(Led(1))
            elif userdata.object_count == 2:
                self.led2_pub.publish(Led(1))
            elif userdata.object_count == 3:
                self.led1_pub.publish(Led(1))
                self.led2_pub.publish(Led(1))
        for _ in range(userdata.object_count):
            self.sound_pub.publish(Sound(0))
            rospy.sleep(rospy.Duration(0.5))
        return "success"
        if not START:
            return "exit"


class LookForSide(State):
    """
    Looking for the half red line while following the white line
    """

    def __init__(self):
        State.__init__(self, outcomes=["see_half_red", "exit", "failure"])

    def execute(self, userdata):
        global START

        if not START:
            return "exit"


class FindGreen(State):
    """
    Recognize the Green image shape based on NN?
    """

    def __init__(self):
        State.__init__(self, outcomes=["success", "exit", "failure"])
        self.shape2_sub = rospy.Subscriber(
            "/shape2", String, self.shapeCallback)
        self.count2_sub = rospy.Subscriber(
            "/count2", Int32, self.countCallback)
        self.done_shape = False
        self.done_count = False

        self.count2_pub = rospy.Publisher("start2", Bool, queue_size=1)
        self.led1_pub = rospy.Publisher(
            "/mobile_base/commands/led1", Led, queue_size=1)
        self.led2_pub = rospy.Publisher(
            "/mobile_base/commands/led2", Led, queue_size=1)

        self.sound_pub = rospy.Publisher(
            "/mobile_base/commands/sound", Sound, queue_size=1)

    def shapeCallback(self, msg):
        global SHAPE
        print("GOT SHAPE", msg)
        SHAPE = msg.data
        self.done_shape = True

    def countCallback(self, msg):
        print("GOT COUNT", msg)
        if msg.data == 1:
            self.led1_pub.publish(Led(1))
        elif msg.data == 2:
            self.led2_pub.publish(Led(1))
        elif msg.data == 3:
            self.led1_pub.publish(Led(1))
            self.led2_pub.publish(Led(1))

        for _ in range(msg.data):
            self.sound_pub.publish(Sound(0))
            rospy.sleep(rospy.Duration(0.5))

        self.done_count = True

    def execute(self, userdata):
        global START, SHAPE

        self.led1_pub.publish(Led(0))
        self.led2_pub.publish(Led(0))

        self.count2_pub.publish(Bool(data=True))
        print("published")

        while not rospy.is_shutdown():
            if self.done_shape and self.done_count:
                break
            rospy.Rate(10).sleep()

        if not START:
            return "exit"
        return "success"


class MoveForward(State):
    """
    Move forward a bit until can't see the red
    """

    def __init__(self):
        State.__init__(self, outcomes=["no_more_red", "exit", "failure"])

    def execute(self, userdata):
        global START

        if not START:
            return "exit"


class CheckShape(State):
    """
    determine if the RED pattern is the shape we see in phase 3
    """

    def __init__(self):
        State.__init__(self, outcomes=["matched", "exit", "failure"])
        self.start3_pub = rospy.Publisher("start3", Bool, queue_size=1)
        self.shape3_sub = rospy.Subscriber(
            "shape3", String, self.shapeCallback)

        self.shape = None
        self.got_shape = False

    def shapeCallback(self, msg):

        self.got_shape = True
        print("GOT SHAPE=", msg.data)
        self.shape = msg.data

    def execute(self, userdata):
        global START, SHAPE_MATCHED, SHAPE

        self.start3_pub.publish(Bool(True))
        self.got_shape = False
        while not rospy.is_shutdown():
            if self.got_shape:
                break
            rospy.Rate(10).sleep()

        if self.shape == SHAPE:
            SHAPE_MATCHED = True
            return "matched"
        elif NUM_SHAPES >= 3:
            return "matched"

        else:
            print("failed?", self.shape)
            return "failure"
        if not START:
            return "exit"

class CheckShape2(State):
    """
    determine if the RED pattern is the shape we see in phase 3
    """

    def __init__(self):
        State.__init__(self, outcomes=["matched", "exit", "failure"])
        self.start3_pub = rospy.Publisher("start5", Bool, queue_size=1)
        self.shape3_sub = rospy.Subscriber(
            "shape5", String, self.shapeCallback)

        self.shape = None
        self.got_shape = False

    def shapeCallback(self, msg):

        self.got_shape = True
        print("GOT SHAPE=", msg.data)
        self.shape = msg.data

    def execute(self, userdata):
        global START, SHAPE_MATCHED, SHAPE

        self.start3_pub.publish(Bool(True))
        self.got_shape = False
        while not rospy.is_shutdown():
            if self.got_shape:
                break
            rospy.Rate(10).sleep()

        if self.shape == SHAPE:
            SHAPE_MATCHED = True
            return "matched"
        elif NUM_SHAPES >= 3:
            return "matched"

        else:
            print("failed?", self.shape)
            return "failure"
        if not START:
            return "exit"

class Signal3(State):
    """
    Make a sound for phase 3
    """

    def __init__(self):
        State.__init__(self, outcomes=["success", "exit", "failure"])
        self.sound_pub = rospy.Publisher(
            "mobile_base/commands/sound", Sound, queue_size=1)

    def execute(self, userdata):
        global START, SHAPE_MATCHED
        if SHAPE_MATCHED:
            self.sound_pub.publish(Sound(0))
        return "success"

        if not START:
            return "exit"


def joy_callback(msg):
    global START, UNKNOWN_CHECKPOINT

    if msg.buttons[4] == 1: # LB
        if msg.buttons[0]:
            UNKNOWN_CHECKPOINT = 0
        elif msg.buttons[1]:
            UNKNOWN_CHECKPOINT = 1
        elif msg.buttons[2]:
            UNKNOWN_CHECKPOINT = 2
        elif msg.buttons[3]:
            UNKNOWN_CHECKPOINT = 3


    elif msg.buttons[5] == 1: #RB
        if msg.buttons[0]:
            UNKNOWN_CHECKPOINT = 4
        elif msg.buttons[1]:
            UNKNOWN_CHECKPOINT = 5
        elif msg.buttons[2]:
            UNKNOWN_CHECKPOINT = 6
        elif msg.buttons[3]:
            UNKNOWN_CHECKPOINT = 7

    elif msg.buttons[0] == 1:  # button A
        START = True
    elif msg.buttons[1] == 1:  # button B
        START = False


def dr_callback(config, level):
    global Kp, Kd, Ki, red_area_threshold, red_timeout, linear_vel, white_max_h, white_max_s, white_max_v, white_min_h, white_min_s, white_min_v, red_max_h, red_max_s, red_max_v, red_min_h, red_min_s, red_min_v

    # Kp = config["Kp"]
    # Kd = config["Kd"]
    # Ki = config["Ki"]
    # linear_vel = config["linear_vel"]

    # white_max_h = config["white_max_h"]
    # white_max_s = config["white_max_s"]
    # white_max_v = config["white_max_v"]

    # white_min_h = config["white_min_h"]
    # white_min_s = config["white_min_s"]
    # white_min_v = config["white_min_v"]

    red_max_h = config["red_max_h"]
    red_max_s = config["red_max_s"]
    red_max_v = config["red_max_v"]

    red_min_h = config["red_min_h"]
    red_min_s = config["red_min_s"]
    red_min_v = config["red_min_v"]

    # red_area_threshold = config["red_area_threshold"]
    # red_timeout = rospy.Duration(config["red_timeout"])

    return config


def move(forward_target, turn_target, pub, ramp_rate=0.5):
    """
    modified version of move(forward, turn) from https://github.com/erichsueh/LifePoints-412-Comp1/blob/2e9fc4701c3cdc8e4ab8b04ca1da8581cfdf0c5b/robber_bot.py#L25
    """
    global FORWARD_CURRENT
    global TURN_CURRENT

    twist = Twist()
    new_forward = ramped_vel(FORWARD_CURRENT, forward_target, ramp_rate)
    new_turn = ramped_vel(TURN_CURRENT, turn_target, ramp_rate)
    twist.linear.x = new_forward
    twist.angular.z = new_turn
    pub.publish(twist)

    FORWARD_CURRENT = new_forward
    TURN_CURRENT = new_turn


def ramped_vel(v_prev, v_target, ramp_rate):
    """
    get the ramped velocity
    from rom https://github.com/MandyMeindersma/Robotics/blob/master/Competitions/Comp1/Evasion.py
    """
    if abs(v_prev) > abs(v_target):
        ramp_rate *= 2
    step = ramp_rate * 0.1
    sign = 1.0 if (v_target > v_prev) else -1.0
    error = math.fabs(v_target - v_prev)
    if error < step:  # we can get there in this time so we are done
        return v_target
    else:
        return v_prev + sign*step


class StartParking(State):
    def __init__(self):
        State.__init__(self, outcomes=["ready"])

    def execute(self, userdata):
        pass


class ParkNext(State):
    def __init__(self):
        State.__init__(self, outcomes=[
                       "see_shape", "see_AR", "close_to_random", "find_nothing"])

        self.checkpoint_list = [MoveBaseGoal(), MoveBaseGoal(), MoveBaseGoal(
        ), MoveBaseGoal(), MoveBaseGoal(), MoveBaseGoal(), MoveBaseGoal(), MoveBaseGoal()]
        self.move_base_client = actionlib.SimpleActionClient(
            "move_base", MoveBaseAction)
        self.shape_start_pub = rospy.Publisher(
            'startShape4', Bool, queue_size=1)

    def reset(self):
        self.marker_data_received = False
        self.found_marker = False

    def marker_callback(self, msg):
        if not self.marker_data_received:
            self.marker_data_received = True

        if len(msg.markers) > 0:
            self.found_marker = True

    def shape_callback(self, msg):
        self.found_shape = msg.data

    def execute(self, userdata):
        global START, CURRENT_CHECKPOINT, UNKNOWN_CHECKPOINT, PHASE4_TASK_COMPLETED

        self.reset()
        marker_sub = rospy.Subscriber(
            'ar_pose_marker_base', AlvarMarkers, self.marker_callback)
        shape_sub = rospy.Subscriber('hasShape4', Bool, self.shape_callback)

        self.shape_start_pub.publish(Bool(True))

        if not rospy.is_shutdown() and START:
            while not self.marker_data_received:
                continue
            # result = self.move_base_client.send_goal_and_wait(
            #     self.checkpoint_list[CURRENT_CHECKPOINT])

            if CURRENT_CHECKPOINT == UNKNOWN_CHECKPOINT:
                marker_sub.unregister()
                return "close_to_random"
            else:
                self.found_marker = False

                rospy.sleep(rospy.Duration(2))

                if self.found_marker:
                    marker_sub.unregister()
                    self.shape_start_pub.publish(Bool(False))
                    return "see_AR"
                # elif found marker
                elif self.found_shape:
                    self.shape_start_pub.publish(Bool(False))
                    return "see_shape"
                else:
                    CURRENT_CHECKPOINT += 1
                    marker_sub.unregister()
                    self.shape_start_pub.publish(Bool(False))
                    return "find_nothing"

        marker_sub.unregister()


class Signal4(State):
    def __init__(self, led1, led1color, led2=False, led2color=None, playsound=True):
        State.__init__(self, outcomes=["done"])
        self.led1 = led1
        self.led1color = led1color
        self.led2 = led2
        self.led2color = led2color
        self.playsound = playsound

        self.led1_pub = rospy.Publisher(
            "/mobile_base/commands/led1", Led, queue_size=1)
        self.led2_pub = rospy.Publisher(
            "/mobile_base/commands/led2", Led, queue_size=1)
        self.sound_pub = rospy.Publisher(
            '/mobile_base/commands/sound', Sound, queue_size=1)

        

    def execute(self, userdata):
        pass

        if self.playsound:
            self.sound_pub.publish(Sound(0))
        if self.led1:
            self.led1_pub.publish(self.led1color)
        if self.led2:
            self.led2_pub.publish(self.led2color)

        rospy.sleep(rospy.Duration(3))
        if self.led1:
            self.led1_pub.publish(0)
        if self.led2:
            self.led2_pub.publish(0)

        return "done"

class CheckCompletion(State):
    def __init__(self, backup=True):
        State.__init__(self, outcomes=["completed", "not_completed", "next"])
        self.backup = backup

    def execute(self, userdata):
        global PHASE4_TASK_COMPLETED, CURRENT_CHECKPOINT

        if START and not rospy.is_shutdown():
            PHASE4_TASK_COMPLETED += 1

            if CURRENT_CHECKPOINT == 8:
                return "completed"
            else:
                CURRENT_CHECKPOINT += 1

                if self.backup:
                    return "not_completed"
                else:
                    return "next"


class ParkAtExit(State):
    def __init__(self):
        State.__init__(self, outcomes=["done"])

        self.end_goal = MoveBaseGoal()
        self.move_base_client = actionlib.SimpleActionClient(
            "move_base", MoveBaseAction)

    def execute(self, userdata):
        result = None
        while not rospy.is_shutdown() and START and result != 3:
            result = self.move_base_client.send_goal_and_wait(self.end_goal)

            if result == 3:
                return "done"


if __name__ == "__main__":
    rospy.init_node('comp3')

    Kp = rospy.get_param("~Kp", 1.0 / 300.0)
    Kd = rospy.get_param("~Kd", 1.0 / 700.0)
    Ki = rospy.get_param("~Ki", 0)
    linear_vel = rospy.get_param("~linear_vel", 0.3)

    white_max_h = rospy.get_param("~white_max_h", 255)
    white_max_s = rospy.get_param("~white_max_s", 72)
    white_max_v = rospy.get_param("~white_max_v", 256)

    white_min_h = rospy.get_param("~white_min_h", 0)
    white_min_s = rospy.get_param("~white_min_s", 0)
    white_min_v = rospy.get_param("~white_min_v", 230)

    red_max_h = rospy.get_param("~red_max_h", 226.8)
    red_max_s = rospy.get_param("~red_max_s", 295.2)
    red_max_v = rospy.get_param("~red_max_v", 256)

    red_min_h = rospy.get_param("~red_min_h", 0)
    red_min_s = rospy.get_param("~red_min_s", 64.8)
    red_min_v = rospy.get_param("~red_min_v", 194)

    red_timeout = rospy.Duration(rospy.get_param("~red_timeout", 1.0))

    red_area_threshold = rospy.get_param("~red_area_threshold", 25000)

    rospy.Subscriber("/joy", Joy, callback=joy_callback)
    srv = Server(Comp3Config, dr_callback)

    sm = StateMachine(outcomes=['success', 'failure'])
    with sm:
        StateMachine.add("Wait", WaitForButton(),
            transitions={'pressed': 'Phase1', 'exit': 'failure'})
            # transitions={'pressed': 'Phase1', 'exit': 'failure'})
                         

        StateMachine.add("Ending", FollowLine(),
                         transitions={"see_red": "Wait", "failure": "failure", "exit": "Wait", "see_nothing": "failure", "see_long_red": "failure"})

        # # Phase 1 sub state
        phase1_sm = StateMachine(outcomes=['success', 'failure', 'exit'])
        with phase1_sm:
            StateMachine.add("Finding1", FollowLine(), transitions={
                             "see_red": "Turn11", "failure": "failure", "exit": "exit", "see_nothing": "failure", "see_long_red": "failure"})
            StateMachine.add("Turn11", Turn(90), transitions={
                             "success": "Count1", "failure": "failure", "exit": "exit"})  # turn left 90 degrees
            StateMachine.add("Count1", DepthCount(), transitions={
                             "success": "MakeSignal1", "failure": "failure", "exit": "exit"})
            StateMachine.add("MakeSignal1", Signal1(), transitions={
                             "success": "Turn12", "failure": "failure", "exit": "exit"})
            StateMachine.add("Turn12", Turn(0), transitions={
                "success": "success", "failure": "failure", "exit": "exit"})  # turn right 90 degrees
        StateMachine.add("Phase1", phase1_sm, transitions={
                         'success': 'Phase2', 'failure': 'failure', 'exit': 'Wait'})

        # Phase 2 sub state
        phase2_sm = StateMachine(outcomes=['success', 'failure', 'exit'])
        with phase2_sm:
            StateMachine.add("Finding2", FollowLine("2.0"), transitions={
                "see_red": "Turn21", "failure": "failure", "exit": "exit", "see_nothing": "failure", "see_long_red": "failure"})
            StateMachine.add("Turn21", Turn(180), transitions={
                             "success": "FollowToEnd", "failure": "failure", "exit": "exit"})
            StateMachine.add("FollowToEnd", FollowLine("2.1"), transitions={
                             "see_red": "TurnLeftAbit", 'failure': "FindGreenShape", "exit": "exit", "see_nothing": "failure", "see_long_red": "failure"})
            StateMachine.add("TurnLeftAbit", Turn(-100), transitions={
                             "success": "Backup", "failure": "failure", "exit": "exit"})
            StateMachine.add("Backup", Translate(
                distance=0.05, linear=-0.2), transitions={"success": "FindGreenShape","failure": "failure", "exit": "exit"})
            StateMachine.add("FindGreenShape", FindGreen(), transitions={
                "success": "TurnBack",  "failure": "failure", "exit": "exit"})
            StateMachine.add("TurnBack", Turn(90), transitions={
                "success": "MoveForward", "failure": "failure", "exit": "exit"})
            StateMachine.add("MoveForward", FollowLine("2.2"), transitions={
                "see_red": "Turn22", "failure": "failure", "exit": "exit", "see_nothing": "failure", "see_long_red": "failure"})
            StateMachine.add("Turn22", Turn(90), transitions={
                "success": "success", "failure": "failure", "exit": "exit"})  # turn left 90
        StateMachine.add("Phase2", phase2_sm, transitions={
            'success': 'Phase4', 'failure': 'failure', 'exit': 'Wait'})

        # Phase 3 sub state
        phase3_sm = StateMachine(outcomes=['success', 'failure', 'exit'])
        with phase3_sm:
            
            StateMachine.add("Finding3", FollowLine("3.1"), transitions={
                "see_red": "Turn31", "failure": "failure", "exit": "exit", "see_nothing": "failure", "see_long_red": "failure"})
            StateMachine.add("Turn31", Turn(0), transitions={
                             "success": "CheckShape", "failure": "failure", "exit": "exit"})  # turn left 90
            StateMachine.add("CheckShape", CheckShape2(), transitions={
                             "matched": "Signal3", "failure": "TurnRight", "exit": "exit"})
            StateMachine.add("Signal3", Signal3(), transitions={
                             "success": "TurnRight", "failure": "failure", "exit": "exit"})
            StateMachine.add("TurnRight", Turn(-90), transitions={
                             "success": "Finding3", "failure": "failure", "exit": "exit"})

        StateMachine.add("Phase3", phase3_sm, transitions={
            'success': 'Ending', 'failure': 'failure', 'exit': 'Wait'})

        # Phase 4 sub state
        # phase4_sm = StateMachine(outcomes=['success', 'failure', 'exit'])
        # with phase4_sm:
        #     StateMachine.add("Finding4", FollowLine("4.1"), transitions={
        #         "see_long_red": "MoveForward", "see_nothing": "failure", "see_red": "failure", "failure": "failure", "exit": "exit"
        #     })

        #     StateMachine.add("MoveForward", Translate(), transitions={
        #         "success": "Turn41"
        #     })

        #     StateMachine.add("Turn41", Turn(-30), transitions={
        #         "success": "FollowRamp", "failure": "failure", "exit": "exit"
        #     })

        #     StateMachine.add("FollowRamp", FollowLine("4.2"), transitions={
        #         "see_nothing": "StartParking", "see_long_red": "failure", "see_red": "failure", "failure": "failure", "exit": "exit"
        #     })

        #     StateMachine.add("StartParking", StartParking(), transitions={
        #         "ready": "ParkNext"
        #     })

        #     StateMachine.add("ParkNext", ParkNext(), transitions={
        #         "see_shape": "MatchShape", "see_AR": "SignalAR", "close_to_random": "SignalRandom", "find_nothing": "ParkNext"
        #     })

        #     StateMachine.add("MatchShape", CheckShape(), transitions={
        #                      "matched": "SignalShape", "failure": "ParkNext", "exit": "exit"})

        #     StateMachine.add("SignalAR", Signal4(True, 1), transitions={
        #                      "done": "CheckCompletion"})

        #     StateMachine.add("SignalShape", Signal4(True, 2), transitions={
        #                      "done": "CheckCompletion"})

        #     StateMachine.add("SignalRandom", Signal4(True, 3), transitions={
        #                      "done": "CheckCompletion"})

        #     StateMachine.add("CheckCompletion", CheckCompletion(), transitions={
        #                      "completed": "PartAtExit", "not_completed": "ParkNext"})

        #     StateMachine.add("PartAtExit", ParkAtExit(), transitions={
        #                      "done": "ForwardUntilWhite"})

        #     StateMachine.add("ForwardUntilWhite", Translate(),
        #                      transitions={"success": "success"})

        phase4_sm = StateMachine(outcomes=['success', 'failure', 'exit'])

        move_list = {
            "point8": [Turn(90), MoveBaseGo(1.1), Turn(0)],
            "point5": [MoveBaseGo(0.25), Turn(90), MoveBaseGo(0.2)],
            "point4": [Turn(180), MoveBaseGo(0.75), Turn(90)],
            "point7": [Turn(180), MoveBaseGo(0.45), Turn(-90)], 
            "point6": [Turn(180), MoveBaseGo(0.7), Turn(-90)],
            "point3": [Turn(0), MoveBaseGo(0.4), Turn(90)],
            "point2": [Turn(180), MoveBaseGo(0.8), Turn(90) ],
            "point1": [Turn(180), MoveBaseGo(0.8), Turn(90)],
            "exit":   [Turn(-90), MoveBaseGo(1.6), Turn(-90)],
        }

        park_distance =       [0.3,       0.5,       0.5,     0.5 ,       0.5,       0.5,   0.5,      0.5,      0.5]

        checkpoint_sequence = ["point8", "point5", "point4", "point7", "point6", "point3", "point2", "point1", "exit"]

        with phase4_sm:
            i = 0

            StateMachine.add("Finding4", FollowLine("4.1"), transitions={	  
                "see_long_red": "MoveForward", "see_nothing": "failure", "see_red": "failure", "failure": "failure", "exit": "exit"	
            })
            StateMachine.add("MoveForward", Translate(distance=0.65, linear=0.2), transitions={
                "success": "Turn41",  "failure": "failure", "exit": "exit"
            })

            StateMachine.add("Turn41", Turn(135), transitions={
                "success": "FollowRamp", "failure": "failure", "exit": "exit"
            })

            StateMachine.add("FollowRamp", FollowLine("4.2"), transitions={
                "see_nothing": checkpoint_sequence[0] + "-0", "see_long_red": "failure", "see_red": "failure", "failure": "failure", "exit": "exit"
            })

            for i in xrange(len(checkpoint_sequence)):
                moves_to_point = move_list[checkpoint_sequence[i]]
                for j in xrange(len(moves_to_point)):
                    
                    name = checkpoint_sequence[i] + "-" + str(j)
                    next_state_name = None

                    if j == len(moves_to_point) - 1:
                        if i < len(checkpoint_sequence) - 1:
                            next_state_name = checkpoint_sequence[i + 1] + "-" + str(0)

                            StateMachine.add(name, moves_to_point[j], transitions={
                                "success": checkpoint_sequence[i] + "-" + "ParkNext", "failure": "failure", "exit": "exit"
                            })

                            StateMachine.add(checkpoint_sequence[i] + "-" + "ParkNext", ParkNext(), transitions={
                                "see_shape": checkpoint_sequence[i] + "-" + "MatchShape", "see_AR": checkpoint_sequence[i] + "-" + "ParkAR", "close_to_random": checkpoint_sequence[i] + "-" + "ParkRandom", "find_nothing": checkpoint_sequence[i] + "-" + "CheckCompletionNoBackup"
                            })

                            StateMachine.add(checkpoint_sequence[i] + "-" + "ParkAR", MoveBaseGo(park_distance[i]), transitions={
                                "success": checkpoint_sequence[i] + "-" + "SignalAR", "failure": "failure", "exit": "exit"
                            })
                            StateMachine.add(checkpoint_sequence[i] + "-" + "ParkRandom", MoveBaseGo(park_distance[i]), transitions={
                                "success": checkpoint_sequence[i] + "-" + "SignalRandom", "failure": "failure", "exit": "exit"
                            })
                            StateMachine.add(checkpoint_sequence[i] + "-" + "ParkShape", MoveBaseGo(park_distance[i]), transitions={
                                "success": checkpoint_sequence[i] + "-" + "SignalShape", "failure": "failure", "exit": "exit"
                            })
                            StateMachine.add(checkpoint_sequence[i] + "-" + "Moveback", Translate(park_distance[i],-0.2), transitions={
                                "success": next_state_name, "failure": "failure", "exit": "exit"
                            })
                            StateMachine.add(checkpoint_sequence[i] + "-" + "MatchShape", CheckShape(), transitions={
                                            "matched": checkpoint_sequence[i] + "-" + "ParkShape", "failure": checkpoint_sequence[i] + "-" + "CheckCompletionNoBackup", "exit": "exit"})

                            StateMachine.add(checkpoint_sequence[i] + "-" + "SignalAR", Signal4(True, 1), transitions={
                                            "done": checkpoint_sequence[i] + "-" + "CheckCompletion"})

                            StateMachine.add(checkpoint_sequence[i] + "-" + "SignalShape", Signal4(True, 2), transitions={
                                            "done": checkpoint_sequence[i] + "-" + "CheckCompletion"})

                            StateMachine.add(checkpoint_sequence[i] + "-" + "SignalRandom", Signal4(True, 3), transitions={
                                            "done": checkpoint_sequence[i] + "-" + "CheckCompletion"})

                            StateMachine.add(checkpoint_sequence[i] + "-" + "CheckCompletion", CheckCompletion(), transitions={
                                            "completed": next_state_name, "not_completed": checkpoint_sequence[i] + "-" + "Moveback", "next": next_state_name})

                            StateMachine.add(checkpoint_sequence[i] + "-" + "CheckCompletionNoBackup", CheckCompletion(False), transitions={
                                            "completed": next_state_name, "not_completed": checkpoint_sequence[i] + "-" + "Moveback", "next": next_state_name})
                        elif i == len(checkpoint_sequence) -1: # last move of last point
                            StateMachine.add(name, moves_to_point[j], transitions={
                                "success": "success", "failure": "failure", "exit": "exit"
                            })
                    elif j < len(moves_to_point) - 1:
                        next_state_name = checkpoint_sequence[i] + "-" + str(j + 1)

                        StateMachine.add(name, moves_to_point[j], transitions={
                            "success": next_state_name, "failure": "failure", "exit": "exit"
                        })
            StateMachine.add("ForwardUntilWhite", Translate(),
                                        transitions={"success": "success"}) 

              

                


        # with phase4_sm:
            

        #     StateMachine.add("ParkNext", ParkNext(), transitions={
        #         "see_shape": "MatchShape", "see_AR": "SignalAR", "close_to_random": "SignalRandom", "find_nothing": "ParkNext"
        #     })

        #     StateMachine.add("MatchShape", CheckShape(), transitions={
        #                      "matched": "SignalShape", "failure": "ParkNext", "exit": "exit"})

        #     StateMachine.add("SignalAR", Signal4(True, 1), transitions={
        #                      "done": "CheckCompletion"})

        #     StateMachine.add("SignalShape", Signal4(True, 2), transitions={
        #                      "done": "CheckCompletion"})

        #     StateMachine.add("SignalRandom", Signal4(True, 3), transitions={
        #                      "done": "CheckCompletion"})

        #     StateMachine.add("CheckCompletion", CheckCompletion(), transitions={
        #                      "completed": "PartAtExit", "not_completed": "ParkNext"})

        #     StateMachine.add("PartAtExit", ParkAtExit(), transitions={
        #                      "done": "ForwardUntilWhite"})

        #     StateMachine.add("ForwardUntilWhite", Translate(),
        #                      transitions={"success": "success"})

        # StateMachine.add("Phase4", phase4_sm, transitions={
        #                  'success': 'Phase3', 'failure': 'failure', 'exit': 'Wait'})

        StateMachine.add("Phase4", phase4_sm, transitions={
                         'success': 'Phase3', 'failure': 'failure', 'exit': 'Wait'})

        

    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    sis.start()
    outcome = sm.execute()
    sis.stop()
