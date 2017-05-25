
#include "segmentation.hpp"


// -----------------------
// Private Methods
// -----------------------
void Segmentation::preprocess(std::string filename)
{
    m_InImage = cv::imread(filename, cv::IMREAD_COLOR);

    cv::cvtColor(m_InImage, m_RGBImage, cv::COLOR_BGR2RGB);

    cv::Mat blur = cv::Mat::zeros(m_InImage.size(), CV_8UC3);
    cv::bilateralFilter(m_InImage, blur, 9, 30, 30);

    cv::cvtColor(blur, m_GrayImage, cv::COLOR_BGR2GRAY);

    cv::threshold(m_GrayImage, m_ThreshImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

}

std::vector<std::vector<cv::Point> > Segmentation::compute_contours(cv::Mat image, int minContourArea)
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

cv::Mat Segmentation::compute_houghLines(bool doProbablistic)
{
    cv::Mat houghLines_image = m_InImage.clone();

    std::vector<std::vector<cv::Point> > filteredContours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(m_ThreshImage, filteredContours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat draw = cv::Mat::zeros(m_ThreshImage.size(), CV_8UC1);
    cv::drawContours(draw, filteredContours, -1, 255, -1);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13,13), cv::Point(0,0));
    cv::Mat G_morph = cv::Mat::zeros(m_ThreshImage.size(), CV_8UC1);
    cv::dilate(draw, G_morph, kernel);

    cv::Mat G_edges = cv::Mat::zeros(m_ThreshImage.size(), CV_8UC1);
    cv::Canny(G_morph, G_edges, 20, 40);

    cv::Mat mask = G_edges.clone();
    cv::Mat image_lineH = cv::Mat::zeros(houghLines_image.size(), CV_8UC3);
    cv::Mat image_lineV = cv::Mat::zeros(houghLines_image.size(), CV_8UC3);
    
    std::vector<cv::Vec4i> line_H, line_V;
    
    if (!doProbablistic)
    {
        std::vector<cv::Vec2f> lines;
        HoughLines(G_edges, lines, 1, (const float) (3.14/2), 2);

        for (size_t i = 0; i < lines.size(); i++)
        {
            float rho = lines[i][0];
            float theta = lines[i][1];
            cv::Point pt1, pt2;
            double a = std::cos(theta);
            double b = std::sin(theta);
            double x0 = a * rho;
            double y0 = b * rho;
            pt1.x = cvRound(x0 + 1000 * (-b));
            pt1.y = cvRound(y0 + 1000 * (a));
            pt2.x = cvRound(x0 - 1000 * (-b));
            pt2.y = cvRound(y0 - 1000 * (a));
            double theta_in_degree = theta * 180.0 / 3.14f;
            cv::Vec4i pts;
            pts[0] = pt1.x;
            pts[1] = pt1.y;
            pts[2] = pt2.x;
            pts[3] = pt2.y;
            if (theta_in_degree > 180 - 30 && theta_in_degree < 30)
            {
                cv::line(image_lineV, pt1, pt2, cv::Scalar(0, 255, 0), 2);
                cv::line(houghLines_image, pt1, pt2, cv::Scalar(0, 255, 0), 2);
                line_V.push_back(pts);
            }
            else if (theta_in_degree > 90 - 45 && theta_in_degree < 90 + 45)
            {
                cv::line(image_lineH, pt1, pt2, cv::Scalar(255, 0, 0), 2);
                cv::line(houghLines_image, pt1, pt2, cv::Scalar(255, 0, 0), 2);
                cv::line(mask, pt1, pt2, cv::Scalar(255, 255, 255), 2);
                line_H.push_back(pts);
            }
        }
    }
    else
    {
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(G_edges, lines, 1, (const float)(3.14) / 180.0f, 10, 5, 3);

        for (size_t i = 0; i < lines.size(); i++)
        {
            cv::Vec4i pts = lines[i];
            int x1 = pts[0];
            int y1 = pts[1];
            int x2 = pts[2];
            int y2 = pts[3];
            float angle = (std::atan2(y2 - y1, x2 - x1) * 180.0) / (const float)(3.14);

            if (angle < 45 && angle > -45)
            {
                line_H.push_back(pts);
            }
            else if ((angle > 85 && angle < 95) || (angle > -95 && angle < -85))
            {
                line_V.push_back(pts);
            }
        }

        for (size_t i = 0; i < line_V.size(); i++)
        {
            cv::Vec4i pts = line_V[i];
            int x1 = pts[0];
            int y1 = pts[1];
            int x2 = pts[2];
            int y2 = pts[3];
            cv::line(image_lineV, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
            cv::line(houghLines_image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
        }

        for (size_t i = 0; i < line_H.size(); i++)
        {
            cv::Vec4i pts = line_H[i];
            int x1 = pts[0];
            int y1 = pts[1];
            int x2 = pts[2];
            int y2 = pts[3];
            cv::line(image_lineH, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
            cv::line(houghLines_image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
            cv::line(mask, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 255), 2);
        }
    }
    cv::imshow("image_lineH", image_lineH);
    cv::imshow("image_lineV", image_lineV);
    cv::imshow("houghLine_image", houghLines_image);
    cv::imshow("mask", mask);
    cv::waitKey(0);
    // cv::destroyAllWindows();

    return mask;
}
// -----------------------
// Public Methods
// -----------------------
Segmentation::Segmentation(std::string filename)
{
    m_Filename = filename;

    m_RGBImage = cv::Mat::zeros(m_InImage.size(), CV_8UC3);
    m_GrayImage = cv::Mat::zeros(m_InImage.size(), CV_8UC1);
    m_ThreshImage = cv::Mat::zeros(m_InImage.size(), CV_8UC1);

    preprocess(m_Filename);
}
void Segmentation::floor()
{
    cv::Mat G_Thresh = compute_houghLines();

    std::vector<std::vector<cv::Point> > filteredContours = compute_contours(G_Thresh, 0);

    cv::Mat drawing = cv::Mat::zeros(G_Thresh.size(), CV_8UC1);
    cv::drawContours(drawing, filteredContours, -1, 255, -1);


    cv::Mat G_morph = cv::Mat::zeros(G_Thresh.size(), CV_8UC1);
    cv::Mat G_edges = cv::Mat::zeros(G_Thresh.size(), CV_8UC1);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                            cv::Size(3, 3), cv::Point(1, 1));
    cv::dilate(drawing, G_morph, kernel);
    cv::Canny(G_morph, G_edges, 40, 100);


    std::vector<cv::Point> floor_edge_pts;

    int height = G_edges.rows - 1;
    int width = G_edges.cols - 1;
    for (int w = 0; w < width; w += 8)
    {
        for (int h = height - 5; h > 0; h--)
        {
            if (G_edges.at<unsigned char>(h, w) == 255)
            {
                floor_edge_pts.push_back(cv::Point(w, h));
                break;
            }
        }
        // floor_edge_pts.push_back(cv::Point(w, height - 5));
    }
    cv::Mat mask = cv::Mat::zeros(G_Thresh.size(), CV_8UC1);
    for (size_t i = 0; i < floor_edge_pts.size() - 1; i++)
    {
        cv::line(mask, floor_edge_pts[i], floor_edge_pts[i + 1], 255, 1);
    }

    cv::Mat floor_mask = cv::Mat::zeros(G_Thresh.size(), CV_8UC1);
    cv::bitwise_not(mask, floor_mask);

    std::vector<std::vector<cv::Point> > floor_pts = compute_contours(mask, 0);

    cv::imshow("G_Thresh", G_Thresh);
    cv::imshow("G_morph", G_morph);
    cv::imshow("G_edges", G_edges);
    cv::imshow("debug", floor_mask);
    cv::waitKey(0);
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

    cv::imshow("floor_image", floor_image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}