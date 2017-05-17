#ifndef __SEGMENTATION_HPP__
#define __SEGMENTATION_HPP__

#include <iostream>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Segmentation
{
private:
    std::string m_Filename;
    cv::Mat m_InImage;
    cv::Mat m_RGBImage;
    cv::Mat m_GrayImage;
    cv::Mat m_ThreshImage;

    void preprocess(std::string filename);
    std::vector<std::vector<cv::Point> > compute_contours(cv::Mat image, int minContourArea=100);
    cv::Mat compute_houghLines(bool doProbablistic = true);
public:
    Segmentation(std::string filename);
    void floor();
};

#endif //__SEGMENTATION_HPP__