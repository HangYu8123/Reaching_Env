#!/usr/bin/env python

import cv2, os, time, math
import rospy
import numpy as np
import argparse
import imutils
from hlpr_manipulation_utils.arm_moveit2 import ArmMoveIt
from hlpr_manipulation_utils.manipulator import Gripper
import hlpr_manipulation_utils.transformations as Transform
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

class kf_recorder():
    def __init__(self,node_name = "kflib" ):
        rospy.init_node(node_name, disable_signals=True)
        self.ARM_RELEASE_SERVICE = '/j2s7s300_driver/in/start_force_control'
        self.ARM_LOCK_SERVICE = '/j2s7s300_driver/in/stop_force_control'
        self.arm_release_srv = rospy.ServiceProxy(self.ARM_RELEASE_SERVICE, Start)
        self.arm_lock_srv = rospy.ServiceProxy(self.ARM_LOCK_SERVICE, Stop)
        self.arm = ArmMoveIt("j2s7s300_link_base")
        self.tf_listener = tf.TransformListener()
        self.is_moving = False

        self.grip = Gripper()
        self.filename = "test.csv"
        self.keyframes = []

        self.topic_name = '/j2s7s300_driver/joints_action/joint_angles'
        self.client = actionlib.SimpleActionClient(self.topic_name, ArmJointAnglesAction)
        self.object_pos = None

    #rospy.spin()


    def release_arm(self):
            self.arm_release_srv()

    def lock_arm(self):
            self.arm_lock_srv()        



    def open_gripper(self):
        self.grip.open()

    def close_gripper(self):
        self.grip.close()
    
    def open_or_close(self):
        grip_pos = rospy.wait_for_message('/gripper/stat', GripperStat)
        print(grip_pos.requested_position)
        if grip_pos.requested_position == 0:
            self.close_gripper()
        else:
            self.open_gripper()


    def record_kf(self):

        joint_angles = rospy.wait_for_message('/j2s7s300_driver/out/joint_angles', JointAngles)
        #
        grip_pos = rospy.wait_for_message('/gripper/stat', GripperStat)
        # print("**************ffffffffffffffffff**************")
        # print(finger_pos.requested_position)
        # print("**************ffffffffffffffffff**************")
        frame = [
                joint_angles.joint1, 
                joint_angles.joint2, 
                joint_angles.joint3, 
                joint_angles.joint4, 
                joint_angles.joint5, 
                joint_angles.joint6, 
                joint_angles.joint7, 
                grip_pos.requested_position
                ]
        print(frame)
        return frame

    def write_kf_to_file(self, keyframes, filename):
        with open(filename + ".csv", 'w') as csvfile:
            kf_writer = csv.writer(csvfile, delimiter=' ',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in keyframes:
                kf_writer.writerow(row)
        #print ("Keyframes recorded")
    def write_traj_to_file(self, trajectories, filename):
        with open(filename + ".csv", 'w') as csvfile:
            kf_writer = csv.writer(csvfile, delimiter=' ',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for keyframes in trajectories:
                for row in keyframes:
                    kf_writer.writerow(row)

    def read_from_file(self, filename):
        with open(filename, 'rb') as file:
            reader = csv.reader(file, 
                            quoting = csv.QUOTE_ALL,
                            delimiter = ' ')
        
        # storing all the rows in an output list
            output = []
            for row in reader:
                row = map(float, row)
                output.append(row[:])
        #print(output)
        return output

    def record_traj(self,trajectory = []):
            
        obj_pose = self.get_obj_pos()
        r = rospy.Rate(1)
        
        while self.is_moving == False:
            #print(self.is_moving) 
            pass
        while (self.is_moving):
            joint_angles = rospy.wait_for_message('/j2s7s300_driver/out/joint_angles', JointAngles)
            grip_pos = rospy.wait_for_message('/gripper/stat', GripperStat)
            curr_ee_pose = rospy.wait_for_message('/j2s7s300_driver/out/tool_pose', PoseStamped)
            #obj_distance = np.linalg.norm([curr_ee_pose.pose.position.x, curr_ee_pose.pose.position.y, curr_ee_pose.pose.position.z]-obj_pose)

            frame = [
                joint_angles.joint1, 
                joint_angles.joint2, 
                joint_angles.joint3, 
                joint_angles.joint4, 
                joint_angles.joint5, 
                joint_angles.joint6, 
                joint_angles.joint7, 
                grip_pos.requested_position,
                curr_ee_pose.pose.position.x, curr_ee_pose.pose.position.y, curr_ee_pose.pose.position.z,
                curr_ee_pose.pose.position.x- obj_pose[0], curr_ee_pose.pose.position.y- obj_pose[1], curr_ee_pose.pose.position.z - obj_pose[2] ,  
                0
                ]
            # print(frame)
            # trajectory.append(frame)
            # print(self.is_moving,"/n-----------------")
                
            trajectory.append(frame)
            #r.sleep()
                
        joint_angles = rospy.wait_for_message('/j2s7s300_driver/out/joint_angles', JointAngles)
        grip_pos = rospy.wait_for_message('/gripper/stat', GripperStat)    
        curr_ee_pose = rospy.wait_for_message('/j2s7s300_driver/out/tool_pose', PoseStamped)
        #obj_distance = np.linalg.norm([curr_ee_pose.pose.position.x, curr_ee_pose.pose.position.y, curr_ee_pose.pose.position.z]-obj_pose)

        frame = [
                joint_angles.joint1, 
                joint_angles.joint2, 
                joint_angles.joint3, 
                joint_angles.joint4, 
                joint_angles.joint5, 
                joint_angles.joint6, 
                joint_angles.joint7, 
                grip_pos.requested_position,
                curr_ee_pose.pose.position.x, curr_ee_pose.pose.position.y, curr_ee_pose.pose.position.z,
                curr_ee_pose.pose.position.x- obj_pose[0], curr_ee_pose.pose.position.y- obj_pose[1], curr_ee_pose.pose.position.z - obj_pose[2] ,  
                0
                ]
        trajectory.append(frame)
        print("***************************************")
        print(trajectory)
        print("***************************************")
        return trajectory

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
            self.open_gripper()
        else:
            self.close_gripper()
        
        self.is_moving = False 

    def get_obj_pos(self):
        arm_frame = "/j2s7s300_link_base"
        camera_frame = "/camera_color_optical_frame"
        if self.object_pos is None:
            while self.object_pos is None:
                self.object_pos = rospy.wait_for_message('/aabl/poi', PoseStamped, timeout=.1)
            object_cameraFrame = PoseStamped()
            # p = point
            q = Quaternion()
            q.x = 0 #rot[0]
            q.y = 0 #rot[1]
            q.z = 0 #rot[2]
            q.w = 1. #rot[3]
            object_cameraFrame.pose.position = self.object_pos.pose.position
            object_cameraFrame.pose.orientation = q
            object_cameraFrame.header.frame_id = camera_frame
            goal_armFrame = self.tf_listener.transformPose(arm_frame, object_cameraFrame)
            pos = goal_armFrame.pose.position
                # NOT SURE IF TF IS RIGHT
            current_object_pos = [pos.x, pos.y, pos.z]
            self.object_pos = current_object_pos
            return current_object_pos 
        else:
            return self.object_pos

    def replay_demo(self, demo, recording = False):
        trajectory = []
        self.move_to_frame(demo[0])
        obj_pose = self.get_obj_pos()
        #probably there should be a step that moves the arm to initial pose
        if recording:
            for i in demo[1:]:
                traj =[]
                joint_angles = rospy.wait_for_message('/j2s7s300_driver/out/joint_angles', JointAngles)
                grip_pos = rospy.wait_for_message('/gripper/stat', GripperStat)    
                curr_ee_pose = rospy.wait_for_message('/j2s7s300_driver/out/tool_pose', PoseStamped)
                #obj_distance = np.linalg.norm(np.array([curr_ee_pose.pose.position.x, curr_ee_pose.pose.position.y, curr_ee_pose.pose.position.z])-np.array(obj_pose))
  
                frame = [
                joint_angles.joint1, 
                joint_angles.joint2, 
                joint_angles.joint3, 
                joint_angles.joint4, 
                joint_angles.joint5, 
                joint_angles.joint6, 
                joint_angles.joint7, 
                grip_pos.requested_position,
                curr_ee_pose.pose.position.x, curr_ee_pose.pose.position.y, curr_ee_pose.pose.position.z,
                curr_ee_pose.pose.position.x - obj_pose[0], curr_ee_pose.pose.position.y- obj_pose[1], curr_ee_pose.pose.position.z - obj_pose[2] ,  
                1
                ]
                traj.append(frame)
                thread.start_new_thread(self.move_to_frame,(i,))
                thread.start_new_thread(trajectory.append , (self.record_traj(traj),))
                while (self.is_moving):
                    pass
          
        else:
            for i in demo:
                self.move_to_frame(i)
                while (self.is_moving):
                    pass
                
        print(trajectory)
        return trajectory

    def current_state(self):
        r = rospy.Rate(1)
        obj_pose = self.get_obj_pos()
        joint_angles = rospy.wait_for_message('/j2s7s300_driver/out/joint_angles', JointAngles)
        grip_pos = rospy.wait_for_message('/gripper/stat', GripperStat)    
        curr_ee_pose = rospy.wait_for_message('/j2s7s300_driver/out/tool_pose', PoseStamped)
        #obj_distance = np.linalg.norm([curr_ee_pose.pose.position.x, curr_ee_pose.pose.position.y, curr_ee_pose.pose.position.z]-obj_pose)

        frame = [
                joint_angles.joint1, 
                joint_angles.joint2, 
                joint_angles.joint3, 
                joint_angles.joint4, 
                joint_angles.joint5, 
                joint_angles.joint6, 
                joint_angles.joint7, 
                grip_pos.requested_position,
                curr_ee_pose.pose.position.x, curr_ee_pose.pose.position.y, curr_ee_pose.pose.position.z,
                curr_ee_pose.pose.position.x- obj_pose[0], curr_ee_pose.pose.position.y- obj_pose[1], curr_ee_pose.pose.position.z - obj_pose[2] ,  
                0
                ]
        return frame


def cross_over_2_k(demo1,demo2, k1, k2):
    return demo1[:k1] + demo2[k2:]

def cross_over_2_all(demo1,demo2):
    offsprings = []
    for i in range(0,len(demo1)):
        for j in range(0,len(demo2)):
            offsprings.append(demo1[:i]+demo2[j:])
    return offsprings

def cross_over_random(demos, len_limt = 0):
    if len_limt == 0:
        demo = []
        while(True):
            d = np.random.randint(len(demos))
            index = np.random.randint(len(demos[d]))
            if index ==  len(demos[d]) - 1:
                demo.append(demos[d][index])
                return demo
            else:
                demo.append(demos[d][index])
    else:
        demo = []
        cnt = 0
        while(True):
            cnt += 1
            d = np.random.randint(len(demos))
            print(d)
            if cnt < len_limt:
                index = np.random.randint(len(demos[d]))
                if index ==  len(demos[d]) - 1:
                    demo.append(demos[d][index])
                    return demo
                else:
                    demo.append(demos[d][index])
            else:
                demo.append(demos[d][-1])
                return demo

def cross_over_all(demos,len_limt = 0):
    offsrpingts = [] 
    if len_limt == 0:
        pass








