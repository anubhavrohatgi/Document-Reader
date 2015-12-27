#pragma once

#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

class OCR
{
public:
	OCR(std::string tessdataPath);
	~OCR();

	bool initOCR();

	std::string performOCR(const cv::Mat& src);


private:

	tesseract::TessBaseAPI tess;
	std::string ocr_path;

	void unsharpMask(cv::Mat& im);
	
};

