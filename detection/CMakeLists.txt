cmake_minimum_required(VERSION 3.0)
project(detection)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        image_transport
        cv_bridge
        )
set(Torch_DIR "/home/pengpeng/libtorch/share/cmake/Torch/")
find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

find_package(Ceres REQUIRED)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

catkin_package()

add_executable(detection
        src/detection_node.cpp
        src/pytorch_cpp_wrapper.cpp
        src/traffic_sign.cpp
        src/vehicle.cpp
        )


target_link_libraries(detection ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${TORCH_LIBRARIES})


