//
// Created by pengpeng on 2019/10/17.
//
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include <iostream>
#include <memory>
#include <tuple>


#include "traffic_sign.h"

class Detector {
private:
    ros::NodeHandle nh_;

    image_transport::ImageTransport it_;

    image_transport::Subscriber sub_image_;
    image_transport::Publisher  pub_image_;

public:
    Detector(ros::NodeHandle & nh);

    void image_callback(const sensor_msgs::ImageConstPtr& msg);

    cv_bridge::CvImagePtr msg_to_cv_bridge(sensor_msgs::ImageConstPtr msg);
    cv_bridge::CvImagePtr msg_to_cv_bridge(sensor_msgs::Image msg);

};

Detector::Detector(ros::NodeHandle & nh)
        : it_(nh), nh_(nh)
{

    sub_image_ = it_.subscribe("image", 1, &Detector::image_callback, this);
    pub_image_ = it_.advertise("detection results", 1);

    // Import the model
    std::string filename;
    nh_.param<std::string>("model_file", filename, "");
//    if(!pt_wrapper_.import_module(filename)) {
//        ROS_ERROR("Failed to import the  model file [%s]", filename.c_str());
//        ros::shutdown();
//    }

}

void Detector::image_callback(const sensor_msgs::ImageConstPtr& msg)
{
    ROS_INFO("[Detector image_callback] Start");

    // Convert the image message to a cv_bridge object
    cv_bridge::CvImagePtr cv_ptr = msg_to_cv_bridge(msg);

    // Run inference
    sensor_msgs::ImagePtr label_msg;
//TODO:detection
    pub_image_.publish(label_msg);
}


/*
 * inference : Forward the given input image through the network and return the inference result
 */
//sensor_msgs::ImagePtr Detector::inference(cv::Mat & input_img)
//{
//    sensor_msgs::ImagePtr label_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", label).toImageMsg();
//
//}


/*
 * msg_to_cv_bridge : Generate a cv_image pointer instance from a given image message pointer
 */
cv_bridge::CvImagePtr Detector::msg_to_cv_bridge(sensor_msgs::ImageConstPtr msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return nullptr;
    }

    return cv_ptr;
}

/*
 * msg_to_cv_bridge : Generate a cv_image pointer instance from a given message
 */
cv_bridge::CvImagePtr Detector::msg_to_cv_bridge(sensor_msgs::Image msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return nullptr;
    }

    return cv_ptr;
}

int main(int argc, char* argv[]) {
    // Initialize the node
    ros::init(argc, argv, "detection");

    ros::NodeHandle nh("~");
    //ros::NodeHandle nh;

    // Initialize the class
    Detector enet_ros(nh);

    ROS_INFO("[TSDetector] The node has been initialized");

    ros::spin();

//  ros::Rate rate(30.0);
//  while(ros::ok()) {
//    ros::spinOnce();
//
//    rate.sleep();
//  }
}
