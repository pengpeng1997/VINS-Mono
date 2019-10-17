//
// Created by pengpeng on 2019/10/17.
//
#include <ros/ros.h>
#include "traffic_sign.h"

int main(int argc, char* argv[]) {
    // Initialize the node
    ros::init(argc, argv, "detection");

    ros::NodeHandle nh("~");
    //ros::NodeHandle nh;

    // Initialize the class
    TSDetector enet_ros(nh);

    ROS_INFO("[TSDetector] The node has been initialized");

    ros::spin();

//  ros::Rate rate(30.0);
//  while(ros::ok()) {
//    ros::spinOnce();
//
//    rate.sleep();
//  }
}
