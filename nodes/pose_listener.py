#!/usr/bin/env python
"""
Echo transforms to terminal and publish to tf.

Based on source code by Shuo Han.
https://github.com/hanshuo/ros_rigit.git

SCL; 5 Jun 2014
"""

import roslib; roslib.load_manifest('ptrack')
import rospy
import tf
from geometry_msgs.msg import Pose2D
from ptrack.msg import pose_objects

import numpy
from numpy import *


class Listener:
    def __init__(self):
        rospy.init_node('pose_listener', anonymous=True)
        rospy.Subscriber('pose_estimation', pose_objects, self.callback)
        self.br = tf.TransformBroadcaster()

        # Online dictionary of publishers, one for each rigid body for
        # which a pose message has been received.  Note that list
        # growth is monotonic.  Each key is a name str, and each value
        # is an instance of rospy.Publisher.
        self.pd = dict()
        rospy.spin()
        
    def callback(self, pose_packet):
        for p in pose_packet.bodies:
            R = numpy.zeros((4,4))
            R[3,3] = 1.
            R[:3,:3] = array(p.R).reshape(3,3)
            T = array(p.T)
            aq = tf.transformations.quaternion_from_matrix(R)
            theta = numpy.arctan2(R[1,0], R[0,0])

            if not self.pd.has_key(p.name):
                self.pd[p.name] = rospy.Publisher(p.name+"/overhead_tracking",
                                                  Pose2D, latch=True)

            self.pd[p.name].publish(Pose2D(x=T[0], y=T[1], theta=theta))
            self.br.sendTransform(T, aq,
                                  rospy.Time.now(), p.name+str("/base_link"), "/odom")

if __name__ == '__main__':
    listener = Listener()
