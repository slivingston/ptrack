#!/usr/bin/env python
"""
Echo transforms to terminal and publish to tf.

Based on source code by Shuo Han.
https://github.com/hanshuo/ros_rigit.git

SCL; 4 Jun 2014
"""

import roslib; roslib.load_manifest('ros_rigit')
import rospy
import tf
from ros_rigit.msg import pose_objects

import numpy
from numpy import *


class Listener:
    def __init__(self):
        rospy.init_node('pose_listener', anonymous=True)
        rospy.Subscriber('pose_estimation', pose_objects, self.callback)
        self.br = tf.TransformBroadcaster()
        rospy.spin()
        
    def callback(self, pose_packet):
        for p in pose_packet.bodies:
            R = numpy.zeros((4,4))
            R[3,3] = 1.
            R[:3,:3] = array(p.R).reshape(3,3)
            T = array(p.T)
            aq = tf.transformations.quaternion_from_matrix(R)
            print '-'*80
            print p.name
            theta = numpy.arctan2(R[1,0], R[0,0])
            print 'pose = ', (T[0], T[1], theta)

            self.br.sendTransform(T, aq,
                                  rospy.Time.now(), p.name+str("/base_link"), "/odom")

if __name__ == '__main__':
    listener = Listener()
