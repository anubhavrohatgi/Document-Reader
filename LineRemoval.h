#pragma once


#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>



//We are trying to remove the vertical lines here
class LineRemoval
{
public:
	LineRemoval();
	~LineRemoval();

	//Pass the deskewed color image to the function
	void process(const cv::Mat& src, cv::Mat& dst);

	void skeletonizaton(cv::Mat& im);


	std::vector<int> getInterLineGap() const{
		return inter_gaps;
	}

private:

	/*Variables*/

	cv::Mat original;
	cv::Mat gray;
	std::vector<int> inter_gaps;


	//Used by skeletonization
	void thinningIteration(cv::Mat& im, int iter);

	//Pre process image further to enhance the lines only
	void preProcess(const cv::Mat& src, cv::Mat& dst);


	//Extract the Hough Lines and filter/merge closest - input binary image with vertical lines
	std::vector<cv::Vec2f> extractLines(const cv::Mat& src);

	//hough to point transform
	void getendpoints(cv::Vec2f line, cv::Point& p1, cv::Point& p2);

	//Draw the lines on the image
	void drawLine(cv::Vec2f line, cv::Mat& img, cv::Scalar color);
};

