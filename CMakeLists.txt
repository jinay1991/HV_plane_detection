cmake_minimum_required(VERSION 2.8)

project(Horizontal_Vertical_Planes)

file(GLOB LIBSOURCES ${CMAKE_CURRENT_SOURCE_DIR}/segmentation.cpp)

find_package(OpenCV REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(HVPlaneDetect SHARED ${LIBSOURCES})
target_link_libraries(HVPlaneDetect  ${OpenCV_LIBS})

# Executable
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testapp.cpp)
add_executable(testHVPlaneDetect ${SOURCES})
target_link_libraries(testHVPlaneDetect  ${OpenCV_LIBS} HVPlaneDetect)