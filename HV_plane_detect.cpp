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

    void preprocess(std::string filename)
    {
        m_InImage = cv::imread(filename, cv::IMREAD_COLOR);

        cv::cvtColor(m_InImage, m_RGBImage, cv::COLOR_BGR2RGB);

        cv::Mat blur = cv::Mat::zeros(m_InImage.size(), CV_8UC3);
        cv::bilateralFilter(m_InImage, blur, 9, 30, 30);

        cv::cvtColor(blur, m_GrayImage, cv::COLOR_BGR2GRAY);

        cv::threshold(m_GrayImage, m_ThreshImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    }

    std::vector<std::vector<cv::Point> > compute_contours(cv::Mat image, int minContourArea=100)
    {
        std::vector<std::vector<cv::Point> > tcontours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(image, tcontours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<std::vector<cv::Point> > contours;
        for (size_t i = 0; i < tcontours.size(); i++)
        {
            if (cv::contourArea(tcontours[i]) < minContourArea)
            {
                continue;
            }
            else
            {
                contours.push_back(tcontours[i]);
            }
        }

        return contours;
    }


public:
    Segmentation(std::string filename)
    {
        m_Filename = filename;

        m_RGBImage = cv::Mat::zeros(m_InImage.size(), CV_8UC3);
        m_GrayImage = cv::Mat::zeros(m_InImage.size(), CV_8UC1);
        m_ThreshImage = cv::Mat::zeros(m_InImage.size(), CV_8UC1);

        preprocess(m_Filename);
    }
    void floor()
    {
        cv::Mat G_Thresh = m_ThreshImage.clone();

        std::vector<std::vector<cv::Point> > filteredContours = compute_contours(G_Thresh);

        cv::Mat drawing = cv::Mat::zeros(G_Thresh.size(), CV_8UC1);
        cv::drawContours(drawing, filteredContours, -1, 255, -1);


        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                cv::Size(1, 13), cv::Point(0, 0));
        cv::Mat G_morph = cv::Mat::zeros(G_Thresh.size(), CV_8UC1);
        cv::dilate(drawing, G_morph, kernel);

        cv::Mat G_edges = cv::Mat::zeros(G_Thresh.size(), CV_8UC1);
        cv::Canny(G_morph, G_edges, 40, 100);


        std::vector<cv::Point> floor_edge_pts;

        int height = G_Thresh.rows - 1;
        int width = G_Thresh.cols - 1;
        for (int w = 0; w < width; w += 8)
        {
            for (int h = height - 20; h > 0; h--)
            {
                if (G_edges.at<unsigned char>(h, w) == 255)
                {
                    floor_edge_pts.push_back(cv::Point(w, h));
                    break;
                }
            }
            // floor_edge_pts.push_back(cv::Point(w, height - 20));
        }
        cv::Mat mask = cv::Mat::zeros(G_Thresh.size(), CV_8UC1);
        for (size_t i = 0; i < floor_edge_pts.size() - 1; i++)
        {
            cv::line(mask, floor_edge_pts[i], floor_edge_pts[i + 1], 255, 1);
        }

        cv::Mat floor_mask = cv::Mat::zeros(G_Thresh.size(), CV_8UC1);
        cv::bitwise_not(mask, floor_mask);

        std::vector<std::vector<cv::Point> > floor_pts = compute_contours(mask, 0);

        for (int w = 0; w < width; w++)
        {
            for (int h = 0; h < height; h++)
            {
                if (floor_mask.at<unsigned char>(h, w) != 255)
                    break;
                else
                    floor_mask.at<unsigned char>(h, w) = 0;
            }
        }

        cv::Mat floor_image = m_InImage.clone();
        cv::drawContours(floor_image, floor_pts, -1, cv::Scalar(0, 255, 0), 2);

        cv::Mat channels[3];
        cv::split(floor_image, channels);

        for (int w = 0; w < width; w++)
        {
            for (int h = 0; h < height; h++)
            {
                if (floor_mask.at<unsigned char>(h, w) > 0)
                    channels[0].at<unsigned char>(h, w) = 255;
            }
        }
        // channels[0].setTo(255, floor_mask);
        cv::merge(channels, 3, floor_image);

        cv::namedWindow("floor_image", cv::WINDOW_AUTOSIZE);
        cv::moveWindow("floor_image", 640,480);
        cv::imshow("floor_image", floor_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
};

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: testHVPlaneDetect <input>" << std::endl;
    }

    Segmentation seg(argv[1]);

    seg.floor();
}