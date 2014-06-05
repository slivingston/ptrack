#!/usr/bin/env python
"""
rigit_tlistener.py [FRAME1 [FRAME2]] ...

where FRAMEi is the name of a file of the form

    fly
    X1 X2 ...
    Y1 Y2 ...

whence the frame is given the name "fly" and the points defining it
are (X1, Y1), (X2, Y2), ...

Based on source code by Shuo Han.
https://github.com/hanshuo/ptrack.git

SCL; 4 Jun 2014
"""

import roslib; roslib.load_manifest('ptrack')
import rospy
from trackem_ros.msg import MTCalPoints
from ptrack.msg import pose_objects, pose_object
from ptrack.rigit import rigit_ransac, rigit_nn

import numpy as np
import sys


class Listener:
    def __init__(self, ref_poses=None, max_freq=20):
        self.fresh_points = False
        self.max_freq = max_freq
        if ref_poses is not None and len(ref_poses) > 0:
            self.init_refs(ref_poses)
        else:
            self.origins = None
        rospy.init_node('rigit_from_trackem', anonymous=True)
        self.pub = rospy.Publisher('pose_estimation', pose_objects)
        rospy.Subscriber('/trackem/calpoints',
                         MTCalPoints,
                         self.callback)
        print 
        print 'rigit_listener intialized.\n'
        print 'Waiting input from trackem_ros...'

    def init_refs(self, ref_poses):
        print 'Start receiving input from trackem_ros. Publishing messages to \'pose estimation\'...'
        self.names = [ref_pose[0] for ref_pose in ref_poses]
        self.origins = [ref_pose[1].copy() for ref_pose in ref_poses]
        self.last_poses = [ref_pose[1].copy() for ref_pose in ref_poses]
        self.Rs = [np.eye(2) for o in self.origins]
        self.Ts = [np.zeros(2) for o in self.origins]
        self.hints = [None for o in self.origins]

    def callback(self, d):
        self.fresh_points = True
        self.world_pts = np.array([[obj.x, obj.y]
                                   for obj in d.points]).T

        if self.origins is None:
            print "="*60
            print "Initializing with"
            print self.world_pts
            print "="*60
            self.init_refs([("new", self.world_pts)])
            
    def run(self):
        loop_rate = rospy.Rate(self.max_freq)
        
        while not rospy.is_shutdown():
            if self.fresh_points:
                world_pts = self.world_pts.copy()
                bodies = []
                for body_index in range(len(self.origins)):
                    if world_pts.shape[1] == 0:
                        break
                    (dR, dT, err), idx_corresp = rigit_nn(self.last_poses[body_index], world_pts)
                    if err > 1e-3:
                        print "\nNearest neighbors (NN) failed, trying RANSAC:"
                        R, T, err, widx, is_successful, num_iter \
                            = rigit_ransac(self.origins[body_index], world_pts,
                                           100, 1e-3,
                                           self.hints[body_index])
                        print "\t", err
                        if is_successful:
                            self.Rs[body_index] = R
                            self.Ts[body_index] = T
                            self.hints[body_index] = widx
                            self.last_poses[body_index] = world_pts[:,widx].copy()
                            R = self.Rs[body_index]
                            R9 = np.array([[R[0,0], R[0,1], 0.],
                                           [R[1,0], R[1,1], 0.],
                                           [0., 0., 1.]])
                            T = self.Ts[body_index]
                            T3 = np.array([T[0], T[1], 0.])
                            bodies.append(pose_object(self.names[body_index],
                                                      R9.flatten().tolist(),
                                                      T3.tolist()))

                        else:
                            self.hints[body_index] = None
                    else:
                        self.Rs[body_index] = np.dot(dR, self.Rs[body_index])
                        self.Ts[body_index] = np.dot(dR, self.Ts[body_index]) + dT
                        self.last_poses[body_index] = world_pts[:,idx_corresp].copy()
                        R = self.Rs[body_index]
                        R9 = np.array([[R[0,0], R[0,1], 0.],
                                       [R[1,0], R[1,1], 0.],
                                       [0., 0., 1.]])
                        T = self.Ts[body_index]
                        T3 = np.array([T[0], T[1], 0.])
                        bodies.append(pose_object(self.names[body_index],
                                                  R9.flatten().tolist(),
                                                  T3.tolist()))

                if len(bodies) > 0:
                    self.pub.publish(bodies)
                    print ".",
                self.fresh_points = False

            loop_rate.sleep()


if __name__ == '__main__':
    max_freq = 20              # Running at 20 Hz maximum

    argv = rospy.myargv()
    if "-h" in argv:
        print "Usage: "+str(argv[0])+" [FRAME1 [FRAME2]] ..."
        exit(1)

    elif len(argv) >= 2:
        ref_poses = []
        for fname in argv[1:]:
            with open(fname, "r") as f:
                name = f.readline().strip()
                ref_poses.append((name, np.loadtxt(f)))
    else:
        ref_poses = None

    listener = Listener(ref_poses, max_freq)
    listener.run()
