cmake_minimum_required(VERSION 2.8.3)
project(ros_rigit)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS rospy geometry_msgs message_generation)

catkin_python_setup()

add_message_files(
  DIRECTORY msg
  FILES
  pose_object.msg
  pose_objects.msg
 )
generate_messages()

catkin_package()
install(PROGRAMS
  nodes/pose_listener.py
  nodes/rigit_tlistener.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
