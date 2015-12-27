#pragma once

#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>


struct blockImage
{
	cv::Mat img;
	int x; // tl.x corner in the original image
	int y; // tl.x corner in the original image
};



class TextRead
{
public:
	TextRead(std::string filename, cv::Mat originalColor);
	~TextRead();

	bool process(const cv::Mat& srcGray);

	std::vector<blockImage> getBlocks()const {
		return block_img;
	}


private:

	std::string fname;	
	cv::Mat gray;
	cv::Mat original;
	std::vector<cv::Rect> blocks;
	std::vector<blockImage> block_img;

	//Sort function - sort with increasing y 
	static bool sortAscending(blockImage a, blockImage b);

};

