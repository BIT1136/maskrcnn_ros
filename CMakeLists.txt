cmake_minimum_required(VERSION 3.0.2)
project(maskrcnn_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  message_generation
)

catkin_python_setup()

add_service_files(
  FILES
  InstanceSeg.srv
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
