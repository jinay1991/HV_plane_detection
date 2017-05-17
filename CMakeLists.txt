cmake_minimum_required(VERSION 2.8)

project(Horizontal_Vertical_Planes)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIR})

add_executable(testHVPlaneDetect ${SOURCES})
target_link_libraries(testHVPlaneDetect ${OpenCV_LIBS})