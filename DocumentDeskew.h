#pragma once


#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>


#define _DEBUG	1


class DocumentDeskew
{
public:
	DocumentDeskew(cv::Mat src);
	~DocumentDeskew();

	cv::Mat processDeskew();

private:
	cv::Mat gray;
	cv::Mat original;

	bool isContourQualifiedQuad(std::vector<cv::Point> contour, std::vector<cv::Point>& approx);

	cv::Mat warpImage(std::vector<cv::Point2f> boundaryCorners);

protected:
	void enhanceImage(const cv::Mat& src, cv::Mat& dst);

	//finds the quadrilateral proportional to the image size
	void findQuads(const cv::Mat& src, std::vector<std::vector<cv::Point>>& quad, int threshLimit = 3);

	//finds the angle between a point and a line
	double findAngle(cv::Point pt1, cv::Point pt2, cv::Point pt0);

	//Sorting function expression
	static bool lArea(std::vector<cv::Point> a1, std::vector<cv::Point> a2);

	void searchDocumentBoundary(const cv::Mat& src, std::vector<cv::Point>& block);
	
	void reorderPoints(std::vector<cv::Point2f>& corners, cv::Point2f center);


};

