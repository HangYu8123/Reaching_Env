#!/usr/bin/env python

import os, time
import re
import sys, select
import cv2, os, time, math
import rospy
import numpy as np
import argparse
import imutils
from geometry_msgs.msg import Pose
import csv
from kinova_msgs.srv import Start, Stop
from moveit_msgs.msg import PositionIKRequest
from moveit_msgs.msg import RobotState
import actionlib
from kinova_msgs.msg import ArmJointAnglesGoal, ArmJointAnglesAction
from kinova_msgs.msg import JointAngles, FingerPosition
import thread
from robotiq_85_msgs.msg import GripperStat
from geometry_msgs.msg import Pose, PoseStamped, Vector3, Point, Quaternion
import tf
from numpy.lib.shape_base import column_stack 
import rospy, tf
from scipy.spatial import distance as dist
import numpy as np
import argparse
from continuous_cartesian import go_to_relative
from hlpr_manipulation_utils.arm_moveit2 import ArmMoveIt
from hlpr_manipulation_utils.manipulator import Gripper
import hlpr_manipulation_utils.transformations as Transform
from hlpr_manipulation_utils.msg import RLTransitionPlusReward
from hlpr_manipulation_utils.srv import GetActionFromObs
from geometry_msgs.msg import Pose, PoseStamped, Vector3, Point, Quaternion
import actionlib
from kinova_msgs.msg import ArmJointAnglesGoal, ArmJointAnglesAction
import kf_agent
from inputimeout import inputimeout, TimeoutOccurred
import signal, readchar
from pynput.keyboard import Key, Controller
TIMEOUT = 3 # number of seconds your want for timeout
arm_frame = "/j2s7s300_link_base"





def interrupted(signum, frame):
    "called when read times out"
    #print 'interrupted!'
    keyboard = Controller()
    key = "a"

    keyboard.press(key)
    keyboard.release(key)


def input():
    try:
            #print 'You have 5 seconds to type in your stuff...'
            foo = readchar.readkey()
            #print(foo)
            return foo
    except:
            # timeout
            return


class  Reaching():
    def __init__(self, max_action=1, n_actions=6, reset_pose=None, episode_time=50, stack_size=4, max_action_true=.05,
                    sparse_rewards=False, success_threshold=.08):
        try:
            rospy.init_node("Reaching", disable_signals=True)
        except:
            pass
        self.arm = ArmMoveIt("j2s7s300_link_base")
        self.grip = Gripper()
        self.reset_pose = reset_pose   

        #self.target = [ 0.05328277, -0.78893369,  0.10388978]
        self.target = [-0.1261, -0.4263, 0.0858]
        self.episode_time = episode_time
        #self.camera.start()
        self.n_actions = 6
        self.action_space = 1
        self.sparse_rewards = sparse_rewards
        self.grip.close()
        self.cnt = 0
        self.done = False
        self.state  = self.get_obs()
        self.next_state = self.state 
# [ 0.1748 -0.593
# [ 0.207  -0.4242
# [-0.1686 -0.5774
# [-0.1261 -0.4263
# z: 0.0896669626236
        self.ARM_RELEASE_SERVICE = '/j2s7s300_driver/in/start_force_control'
        self.ARM_LOCK_SERVICE = '/j2s7s300_driver/in/stop_force_control'
        self.arm_release_srv = rospy.ServiceProxy(self.ARM_RELEASE_SERVICE, Start)
        self.arm_lock_srv = rospy.ServiceProxy(self.ARM_LOCK_SERVICE, Stop)
        self.arm = ArmMoveIt("j2s7s300_link_base")
        self.tf_listener = tf.TransformListener()
        self.is_moving = False
        self.topic_name = '/j2s7s300_driver/joints_action/joint_angles'
        self.client = actionlib.SimpleActionClient(self.topic_name, ArmJointAnglesAction)
        self.object_pos = None
        # self.right_back = [113.23162841796875, 127.58155822753906, 186.45321655273438, 76.94646453857422, 170.41021728515625, 150.9701690673828, 260.50738525390625, 1.0]
        # self.left_back = [77.96541595458984, 133.20977783203125, 176.51698303222656, 70.0947036743164, 175.44898986816406, 156.97613525390625, 260.50738525390625, 1.0]
        # self.left_front = [78.92186737060547, 117.53868865966797, 181.82089233398438, 108.19294738769531, 175.4025421142578, 134.1125030517578, 260.50738525390625, 1.0]
        # self.right_front = [107.40848541259766, 117.48986053466797, 189.65536499023438, 107.9133529663086, 175.3997039794922, 134.21804809570312, 260.50738525390625, 1.0]
        self.right_back = [94.70604705810547, 141.18145751953125, 198.02059936523438, 77.33027648925781, 165.373291015625, 138.90811157226562, 305.0677490234375, 1.0]
        self.left_back = [67.33387756347656, 139.68556213378906, 178.99354553222656, 78.54619598388672, 176.41107177734375, 145.59689331054688, 305.0667419433594, 1.0]
        self.left_front = [67.67176818847656, 124.49005889892578, 193.93260192871094, 112.3369140625, 165.373291015625, 128.25228881835938, 305.0677490234375, 1.0]
        self.right_front = [96.81249237060547, 124.48957061767578, 197.97752380371094, 109.4166030883789, 165.37342834472656, 128.1443328857422, 305.0677490234375, 1.0]
        self.x_max = 0.18
        self.y_max = -0.42
        self.x_min = -0.15
        self.y_min = -0.55
        self.z = 0.11
        #self.ini_frame = [98.8447265625, 170.96554565429688, 170.94142150878906, 91.81941986083984, 177.91305541992188, 99.4183349609375, 273.16748046875, 1.0]
        #self.end_frame = [123.39920806884766, 110.06729888916016, 180.34024047851562, 153.36834716796875, 177.7704315185547, 98.41349792480469, 273.1673278808594, 1.0]        
        #self.move_to_frame(self.ini_frame)
        #self.move_to_frame(self.end_frame)
        #go_to_relative([0.1, 0, 0, 0, 0, 0])
        os.system('rosservice call /j2s7s300_driver/in/home_arm')


    def move_to_frame(self, frame):
        while(self.is_moving):
            pass
        print("moving time!")
        self.is_moving = True
        self.client.wait_for_server()
        goal = ArmJointAnglesGoal()

        goal.angles.joint1 = frame[0]
        goal.angles.joint2 = frame[1]
        goal.angles.joint3 = frame[2]
        goal.angles.joint4 = frame[3]
        goal.angles.joint5 = frame[4]
        goal.angles.joint6 = frame[5]
        goal.angles.joint7 = frame[6]
        #print(goal.angles)
        self.client.send_goal(goal)
        self.client.wait_for_result()
        if frame[7] == 0:
            self.grip.open()
        else:
            self.grip.close()
        
        self.is_moving = False 

    def reset(self):
        # self.move_to_frame(self.left_front)
        # keyboard = Controller()
        # key = "a"

        # keyboard.press(key)
        # keyboard.release(key)
        self.grip.close()
        self.cnt = 0
        self.done = False
        self.move_to_frame(self.left_front)
        print("reset done!")
        act = [ 0, 0, 0.03, 0, 0, 0]
        go_to_relative(act, collision_check=True, complete_action=True)
        #os.system('rosservice call /j2s7s300_driver/in/home_arm')
        self.state  = self.get_obs()
        # print(self.state)

        # self.move_to_frame(self.left_back)
        # self.state  = self.get_obs()
        # print(self.state)

        # self.move_to_frame(self.right_front)
        # self.state  = self.get_obs()
        # print(self.state)

        # self.move_to_frame(self.right_back)
        # self.state  = self.get_obs()
        # print(self.state)


        self.next_state = self.state[:] 
        return np.array(self.state)
    def get_height(self):
        curr_ee_pose = rospy.wait_for_message('/j2s7s300_driver/out/tool_pose', PoseStamped)
        return curr_ee_pose.pose.position.z
    def get_obs(self):

        acc = 4
        curr_ee_pose = rospy.wait_for_message('/j2s7s300_driver/out/tool_pose', PoseStamped)
        print("CURR POSE:", curr_ee_pose.pose)
        #curr_ee_pose = [curr_ee_pose.pose.position.x, curr_ee_pose.pose.position.y, curr_ee_pose.pose.position.z]
        curr_ee_pose = [round(curr_ee_pose.pose.position.x,acc), round(curr_ee_pose.pose.position.y,acc), #round(curr_ee_pose.pose.position.z,acc),
        round(self.target[0]- curr_ee_pose.pose.position.x,acc), round(self.target[1]-curr_ee_pose.pose.position.y,acc)] #round(self.target[2]-curr_ee_pose.pose.position.z,acc)]

        return np.array(curr_ee_pose)

    def step(self, action, enrich= 0, complete_action=False, action_duration=0.3, check_collisions=True):
        step_len = 0.05
        reward = 0
        if action == 0 :
            act = [step_len, 0, 0, 0, 0, 0]
            if self.state[0] + step_len < self.x_max:
                go_to_relative(act, collision_check=check_collisions, complete_action=True)
            else:
                reward -= 0.001
        if action == 1 :
            act = [0, step_len, 0, 0, 0, 0]
            
            if self.state[1] + step_len < self.y_max:
                go_to_relative(act, collision_check=check_collisions, complete_action=True)
            else:
                reward -= 0.001
        if action == 2 :
            act = [-1*step_len, 0, 0, 0, 0, 0]
            if self.state[0] - step_len > self.x_min:
                go_to_relative(act, collision_check=check_collisions, complete_action=True)
            else:
                reward -= 0.001
        if action == 3:
            act = [ 0, -1* step_len, 0, 0, 0, 0]
            if self.state[1] - step_len > self.y_min:
                go_to_relative(act, collision_check=check_collisions, complete_action=True)
            else:
                reward -= 0.001    
        if action  == 4:
            act = [ 0, 0, -0.03, 0, 0, 0]
            go_to_relative(act, collision_check=check_collisions, complete_action=True)
            #time.sleep(2)
            signal.signal(signal.SIGALRM, interrupted)
            signal.alarm(TIMEOUT)
            s = input()
            
            #signal.alarm(0)
            print(s)
            
            if s == 'b':
                act = [ 0, 0, 0.03, 0, 0, 0]
                go_to_relative(act, collision_check=check_collisions, complete_action=True)
                self.done = True
                reward+= 10 
            else:
                #self.done = Truea
                reward -= 1
                z = self.get_height()
                act = [ 0, 0, self.z-z, 0, 0, 0]
                go_to_relative(act, collision_check=True, complete_action=True)
            #if self.done == True:
            # done_check = True
            # for i in range (2):
            #     if abs(self.state[i] - self.target[i]) > 0.005:
            #         done_check = False
            # if done_check :
            #     reward += 100
            #     self.done = True
            # else:
            #     reward -= 0
            #     act = [ 0, 0, 0.04, 0, 0, 0]
            #     go_to_relative(act, collision_check=True, complete_action=True)
                



        
        self.cnt +=1
        self.next_state = self.get_obs()
        
        reward  +=  10*(self.get_reward() -0.001)
     
        if self.cnt >= self.episode_time:
            self.done = True
            #reward -= 10bb


        buffer = []





        buffer.append([self.state[:], self.next_state[:], reward, self.done] )
        self.state = self.next_state[:]
        

        return buffer

    def get_reward(self):
        dis = abs(self.target[0] -  self.state[0]) + abs(self.target[1] -  self.state[1]) #+ abs(self.target[2] -  self.state[2])
        next_dis = abs(self.target[0] -  self.next_state[0]) + abs(self.target[1] -  self.next_state[1]) #+ abs(self.target[2] -  self.next_state[2])
        return dis - next_dis
    def abs_dis(self):
        dis = abs(self.target[0] -  self.state[0]) + abs(self.target[1] -  self.state[1]) #+ abs(self.target[2] -  self.state[2])
        return dis
    def enrich_reward(self, state, next_state):
        dis = abs(self.target[0] -  state[0]) + abs(self.target[1] -  state[1]) #+ abs(self.target[2] -  state[2])
        next_dis = abs(self.target[0] -  next_state[0]) + abs(self.target[1] -  next_state[1]) #+ abs(self.target[2] -  next_state[2])
        return dis - next_dis

#go_to_start(self.arm, start=True)
#env = Reaching()
#print(env.get_obs())


#go_to_relative(act, collision_check= True, complete_action=True)
#print("*************************got it ****************************")
#print(env.get_obs())
