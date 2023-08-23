#! /usr/bin/python3.8

import rospy
import numpy as np

from fast_fly.msg import TrackTraj
from geometry_msgs.msg import Point, Vector3
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

import os, sys
BASEPATH = os.path.abspath(__file__).split('script/', 1)[0]+'script/fast_fly/'
sys.path += [BASEPATH]

from gates.gates import Gates

rospy.init_node("gates_sim")
gates_pub = rospy.Publisher("~gates", TrackTraj, tcp_nodelay=True, queue_size=1)
gates_marker_pub = rospy.Publisher("/plan/gates_marker", Marker, queue_size=1)

gates = Gates(BASEPATH+"gates/gates_n7.yaml")

d_gate = gates._pos[5].copy()

def timer_cb(event):
    gates_traj = TrackTraj()
    gates_marker = Marker()
    for i in range(gates._N):
        pos = Point()
        pos.x = gates._pos[i][0]
        pos.y = gates._pos[i][1]
        pos.z = gates._pos[i][2]
        gates_traj.position.append(pos)

        pos = Point()
        pos.y = gates._pos[i][0]
        pos.x = gates._pos[i][1]
        pos.z = -gates._pos[i][2]
        gates_marker.header.frame_id = "world"
        gates_marker.action=Marker.ADD
        gates_marker.type = Marker.SPHERE_LIST
        gates_marker.pose.position.x = 0
        gates_marker.pose.position.y = 0
        gates_marker.pose.position.z = 0
        gates_marker.pose.orientation.w = 1
        gates_marker.pose.orientation.x = 0
        gates_marker.pose.orientation.y = 0
        gates_marker.pose.orientation.z = 0
        gates_marker.scale = Vector3(0.25,0.25,0.25)
        gates_marker.points.append(pos)
        gates_marker.colors.append(ColorRGBA(1,0,0,1))
    gates_pub.publish(gates_traj)
    gates_marker_pub.publish(gates_marker)

def odom_cb(msg: Odometry):
    pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
    g_pos = np.array(gates._pos[5])
    if np.linalg.norm(pos-g_pos) < 0.3:
        mov = np.random.normal(-0.5, 0.5, 3)
        gates._pos[5][0] = d_gate[0]+mov[0]
        gates._pos[5][1] = d_gate[1]+mov[1]
        gates._pos[5][2] = d_gate[2]+mov[2]
    
rospy.Timer(rospy.Duration(0.1), timer_cb)
rospy.Subscriber("~odom", Odometry, odom_cb, queue_size=1, tcp_nodelay=True)

rospy.spin()
