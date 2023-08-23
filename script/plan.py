#! /usr/bin/python3.8

import rospy
import numpy as np

from fast_fly.msg import TrackTraj
from geometry_msgs.msg import Point, Vector3, Quaternion, PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker

import os, sys
BASEPATH = os.path.abspath(__file__).split('script', 1)[0]+'script/fast_fly/'
sys.path += [BASEPATH]

from time_optimal_planner import WayPointOpt, cal_Ns
from gates.gates import Gates
from quadrotor import QuadrotorModel

import time

rospy.init_node("plan")
rospy.loginfo("plan started!!!!!!!!!!!!")
traj_pub = rospy.Publisher("~track_traj", TrackTraj, tcp_nodelay=True, queue_size=1)
planned_path_pub = rospy.Publisher("planed_path", Path, queue_size=1)

gate = Gates(BASEPATH+"gates/gates_n7.yaml")


quad = QuadrotorModel(BASEPATH+'quad/quad_real.yaml')
Ns = cal_Ns(gate, 0.3, loop=True)
dts = np.array([0.2]*gate._N)
wp_opt = WayPointOpt(quad, gate._N, Ns, loop=True)
wp_opt.define_opt()
wp_opt.define_opt_t()
res = wp_opt.solve_opt([], np.array(gate._pos).flatten(), dts)

def pub_traj(opt_t_res, opt:WayPointOpt):
    x = opt_t_res['x'].full().flatten()
    traj = TrackTraj()

    if opt._loop:
        s = x[(opt._Herizon-1)*opt._X_dim:]
    else:
        s = opt._xinit
    pos = Point()
    pos.x = s[0]
    pos.y = s[1]
    pos.z = s[2]
    vel = Vector3()
    vel.x = s[3]
    vel.y = s[4]
    vel.z = s[5]
    quat = Quaternion()
    quat.w = s[6]
    quat.x = s[7]
    quat.y = s[8]
    quat.z = s[9]
    angular = Vector3()
    angular.x = s[10]
    angular.y = s[11]
    angular.z = s[12]

    traj.position.append(pos)
    traj.velocity.append(vel)
    traj.orientation.append(quat)
    traj.angular.append(angular)
        
    for i in range(opt._wp_num):
        for j in range(opt._Ns[i]):
            idx = opt._N_wp_base[i]+j
            s = x[idx*opt._X_dim: (idx+1)*opt._X_dim]
            pos = Point()
            pos.x = s[0]
            pos.y = s[1]
            pos.z = s[2]
            vel = Vector3()
            vel.x = s[3]
            vel.y = s[4]
            vel.z = s[5]
            quat = Quaternion()
            quat.w = s[6]
            quat.x = s[7]
            quat.y = s[8]
            quat.z = s[9]
            angular = Vector3()
            angular.x = s[10]
            angular.y = s[11]
            angular.z = s[12]
            dt = x[-opt._wp_num+i]

            traj.position.append(pos)
            traj.velocity.append(vel)
            traj.orientation.append(quat)
            traj.angular.append(angular)
            traj.dt.append(dt)

    traj_pub.publish(traj)

def pub_path_visualization(opt_t_res, opt:WayPointOpt):
    x = opt_t_res['x'].full().flatten()

    msg = Path()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "world"
    for i in range(opt._Herizon):
        pos = PoseStamped()
        pos.header.frame_id = "world"
        pos.pose.position.y = x[i*opt._X_dim+0]
        pos.pose.position.x = x[i*opt._X_dim+1]
        pos.pose.position.z = -x[i*opt._X_dim+2]

        pos.pose.orientation.w = 1
        pos.pose.orientation.y = 0
        pos.pose.orientation.x = 0
        pos.pose.orientation.z = 0
        msg.poses.append(pos)
    planned_path_pub.publish(msg)

def gates_update_cb(msg:TrackTraj):
    gates = Gates()
    for g_pos in msg.position:
        gates.add_gate([g_pos.x, g_pos.y, g_pos.z], 0)
    res_t = wp_opt.solve_opt_t([], np.array(gates._pos).flatten())
    pub_traj(res_t, wp_opt)
    pub_path_visualization(res_t, wp_opt)

rospy.Subscriber("~gates", TrackTraj, gates_update_cb, queue_size=1)

rospy.spin()
