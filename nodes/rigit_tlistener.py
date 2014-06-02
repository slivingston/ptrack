#!/usr/bin/env python
"""
SCL; 18 Aug 2013
"""

import roslib; roslib.load_manifest('ros_rigit')
import rospy
from trackem_ros.msg import MTCalPoints
from ros_rigit.msg import pose_objects, pose_object
from ros_rigit.rigit import rigit_ransac, rigit_nn

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
        print 'Start receiving input from flydra. Publishing messages to \'pose estimation\'...'
        self.names = [ref_pose[0] for ref_pose in ref_poses]
        self.origins = [ref_pose[1].copy() for ref_pose in ref_poses]
        self.last_poses = [ref_pose[1].copy() for ref_pose in ref_poses]
        self.Rs = [np.eye(3) for o in self.origins]
        self.Ts = [np.zeros(3) for o in self.origins]
        self.hints = [None for o in self.origins]

    def callback(self, d):
        self.fresh_points = True
        self.world_pts = np.array([[obj.x, obj.y, 0.]
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
                    # (dR, dT, err), idx_corresp = rigit_nn(self.last_poses[body_index], world_pts)
                    err = 1.
                    if err > .01:
                        R, T, err, widx, is_successful, num_iter = rigit_ransac(self.origins[body_index], world_pts, 100, .05, self.hints[body_index])
                        print "\t", err
                        if is_successful:
                            self.Rs[body_index] = R
                            self.Ts[body_index] = T
                            self.hints[body_index] = widx
                            # self.last_poses[body_index] = world_pts[:,widx].copy()
#                            world_pts = world_pts[:,[i for i in range(world_pts.shape[1]) if i not in widx]]
                            bodies.append(pose_object(self.names[body_index], self.Rs[body_index].flatten().tolist(), self.Ts[body_index].tolist()))
                        else:
                            self.hints[body_index] = None
                    else:
                        self.Rs[body_index] = np.dot(dR, self.Rs[body_index])
                        self.Ts[body_index] = np.dot(dR, self.Ts[body_index]) + dT
                        self.last_poses[body_index] = world_pts[:,idx_corresp].copy()
#                        world_pts = world_pts[:,[i for i in range(world_pts.shape[1]) if i not in widx]]
                        bodies.append(pose_object(self.names[body_index], self.Rs[body_index].flatten().tolist(), self.Ts[body_index].tolist()))

                if len(bodies) > 0:
                    self.pub.publish(bodies)
                    print ".",
                self.fresh_points = False

            loop_rate.sleep()


if __name__ == '__main__':
    max_freq = 20              # Running at 20 Hz maximum
    
    if "-h" in sys.argv:
        print "Usage: "+str(sys.argv[0])+" [FILE1 [...]]"
        exit(1)
    elif len(sys.argv) >= 2:
        # with open(sys.argv[1], "r") as f:
        #     ffile = np.load(f)
        # origin_pose = ffile.frame_points
        ref_poses = []
        for fname in sys.argv[1:]:
            if ":=" in fname:
                break  # Start ignoring arguments at first remap statement
            with open(fname, "r") as f:
                ref_poses.append((fname, np.loadtxt(f)))
    else:
        ref_poses = None

    listener = Listener(ref_poses, max_freq)
    listener.run()