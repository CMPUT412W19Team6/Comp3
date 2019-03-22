# Competition 3: Parking with GMapping, AMCL and ARTags

**Description:** In this competition, the robot is supposed to follow a line, count objects, match similar shaped objects, find AR tags and park at certain points.

## 1. Purpose

For this competition, we want the robot to follow a pre-defined course and do different vision and navigation tasks. 

### The Course Map

- The course map in simulation:

    <img src="https://github.com/CMPUT412W19Team6/Competition2/blob/master/course_sim.png?s=200" width="200">

- The actual course's picture:

    <img src="https://github.com/CMPUT412W19Team6/Competition2/blob/master/course_pic.png?s=200" width="200">
<!-- ![pick link](https://github.com/CMPUT412W19Team6/Competition2/blob/master/course_pic.png?s=200) -->

- Explanation:
     - The robot is supposed to follow the white lines on the floor, and stop for a bit when reaching a red mark on the floor.
     - There are 3 locations (1,2 and 4) where vision tasks are required in addition to another location (3) that requires both vision and navigation tasks:
        
        a. Location 1: 
        
        Detect how many cylinders are on the left , (Maximum 3), and signal the result with LED lights and sound. 
        Example picture:         
        <img src="https://github.com/CMPUT412W19Team6/Competition2/blob/master/location1.png?s=200" width="200">
        <!-- ![location 1 picture](https://github.com/CMPUT412W19Team6/Competition2/blob/master/location1.png?s=200) -->
        
        b. Location 2: 
        
        Detect how many geometric shapes are there (Maximum 3), signal the result with LED lights and sound,and recognize what the green shape is (one of the three shapes: triangle, square, circle). 
        Example picture:         
        <img src="https://github.com/CMPUT412W19Team6/Competition2/blob/master/location2.png?s=200" width="200">
        <!-- Example picture: ![location 2 picture](https://github.com/CMPUT412W19Team6/Competition2/blob/master/location2.png?s=200) -->
        
        c. Location 3: 
        
        Park at one of the parking spots specified at the start of the run, park at the spot where the AR tag is located, park at the spot that has the object with the same shape as the green shape from Location 2.
        Example picture:         
        <img src="https://github.com/CMPUT412W19Team6/Competition2/blob/master/location3.png?s=200" width="200">
        <!-- Example picture: ![location 3 picture](https://github.com/CMPUT412W19Team6/Competition2/blob/master/location3.png?s=200) -->
        
        d. Location 4: 
        
        Recognize the red shapes on the left one by one, and signal with a sound when finding one that's matching the green shape discovered at Location 2. 
        Example picture:         
        <img src="https://github.com/CMPUT412W19Team6/Competition2/blob/master/location3.png?s=200" width="200">
        <!-- Example picture: ![location 3 picture](https://github.com/CMPUT412W19Team6/Competition2/blob/master/location3.png?s=200) -->


## 2. Pre-requisites

## 2.1 Hardware requirement

- A kobuki turtlebot base
- An Asus Xtion Pro
- A controller (prefered logitech)
- A device with minimum 3 usb ports
- The course set up 

### 2.2 Software requirement

- ROS kinetic (which includes opencv2 ros package) ([Guide here](http://wiki.ros.org/kinetic/Installation/Ubuntu))

- Turtlebot packages ([Guide here](http://wiki.ros.org/action/show/Robots/TurtleBot?action=show&redirect=TurtleBot))

  ```bash
  sudo apt-get install ros-kinetic-turtlebot
  sudo apt-get install ros-kinetic-turtlebot-apps
  sudo apt-get install ros-kinetic-turtlebot-interactions
  sudo apt-get install ros-kinetic-turtlebot-simulator
  ```

- Kobuki ROS packages ([Guide here](https://wiki.ros.org/kobuki/Tutorials/Installation))

  ```bash
  sudo apt-get install ros-kinetic-kobuki
  sudo apt-get install ros-kinetic-kobuki-core
  ```

- Upgrade camera packages

  ```bash
  sudo apt-get install ros-kinetic-openni2-camera
  sudo apt-get install ros-kinetic-openni2-launch
  ```

- ROS smach ([Guide here](http://wiki.ros.org/smach))

- Python2.7 ([Guide here](https://www.python.org/))

- Python package `imutils` ([Repo here](https://github.com/jrosebr1/imutils))
  ```
  pip install imutils
  ```
## 3. Execution

### 3.1 Quickstart

1. Clone this repo into the source directory of your catkin workspace (e.g. catkin_ws/src)

   ```bash
   # under catkin_ws/src folder
   mkdir comp2
   git clone https://github.com/CMPUT412W19Team6/Competition2.git
   ```

2. Run catkin_make and source the setup.bash

   ```bash
   cd ..
   catkin_make
   source ./devel/setup.bash
   ```

3. Connect your your kobuki base, Asus Xtion Pro and controller.

4. Power up the kobuki base and put it on the start position

5. Start the library

   ```bash
   roslaunch comp3 comp3.launch
   ```
5. Specify the parking spot of location 3 by doing one of the following button combo on the joypad
   ```
   1st spot: Left Bar + A
   2nd spot: Left Bar + B
   3rd spot: Left Bar + X
   4th spot: Left Bar + Y
   5th spot: Right Bar + A
   6th spot: Right Bar + B
   7th spot: Right Bar + X
   8th spot: Right Bar + Y
   ```
6. Start the turtlebot by pressing A on the controller


## 4. Concepts & Code

### Overview

- state machine:
    ![statemachine](https://github.com/CMPUT412W19Team6/Competition2/blob/master/SM_Full.png?s=200)

    > Note1: There are 3 phases and a wait start in the state machine. Everytime the robot saw a long red strip, it will end the current phase.
    
    > Note2: The waite state is the starting state. When `button A` is pressed, it will enter Phase 1. And if `button B` is pressed during any phase, it will return to wait state.

### Counting objects

_Concept_:

    1. Filter out the image based on HSV (Get red color for location 1 and 3, anything but white for location 2).

    2. Count the number of contours for what's left.


### Recognizing object shapes

_Concept_:

    1. Filter out the image based on HSV. (Get green color for location 2 and red color for location 3)

    2. Found the contour with largest area.

    3. Try to fit a polygon to the contour.

    4. Determine the shape
    
        4.1 If the number of conner is 3, it's triangle.
        4.1 If the number of conner is 4, it's square.
        4.1 If the number of conner is greater than 4, it's circle.

### Following the white line

_Concept_:

    1. Filter out all the colors of the video stream except white and convert the images to be binary.

    2. Keep 20 rows of the images from the video stream starting at 3/4th the height of the images and discard all the other rows

    3. Calculate the moments and find the center of the white line. Since the images from the video have been converted to be binary, the white pixels of the white line can be considered mass whose zeroth moment represents the total mass and the first moment divided by the zeroth moment (total mass) is the center of the white line (center of mass).

    4. Adjust robot's angular velocity so that if the centroid is not at center, change robot's angular velocity to turn towards the centroid.
    
### Detecting red spots and differentiating between large and small spots

_Concept_:

    1. Filter out all the colors of the video stream except red and convert the images to be binary.

    2. Keep bottom 50% of rows of the images from the video stream.

    3. Calculate the the sum of the area of all the contours.

    4. Differentiate between large and small spots using a threshold on the sum of area.

### Going off ramp

_Concept_:

    1. After completing everything from location 2, get back on the main track.
    2. When the first red marker is seen:
        2.1 Stop for a little while
        2.2 Go forward a little
        2.3 Turn towards the off ramp white line
        2.4 Follow the white line that is visible in the middle to its end

### Completing tasks at location 3

_Concept_:

    1. For each parking spot, do the following:
        
      
## 5. Lesson Learned

### Do it the ROS way: Separate tasks into seprate nodes
    
    During our competition, we had counting objects for location1 together with our main node (which 
    is using image to follow white lines). Because image processing for counting the objects took too 
    many computing resources, the camera was not responding correctly for line following, so the robot
    always went off track after location1. However, after we took the image process part out and put 
    it into another node, it works perfectly. 
    
