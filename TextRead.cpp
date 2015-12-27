#include "TextRead.h"


TextRead::TextRead(std::string filename, cv::Mat originalColor) : fname(filename), original(originalColor)
{
}


TextRead::~TextRead()
{
}


bool TextRead::process(const cv::Mat& srcGray )
{	
	blocks.clear();

	//Keep a copy of original grayscale
	gray = srcGray.clone();

	//Upscale the image
	cv::Mat UGray;
	cv::pyrUp(gray, UGray);

	cv::Mat gradient;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 1));
	cv::morphologyEx(UGray, gradient, cv::MORPH_GRADIENT, kernel);

	//Binarize image
	cv::threshold(gradient, gradient, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
	
	//Connecting horizontal fragments - words to sentences
	kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 1));
	cv::morphologyEx(gradient, gradient, cv::MORPH_CLOSE, kernel);


	//blocks = gradient.clone();


	//Now we obtain the bounding boxes/blocks and then we will pass these process blocks 
	//to OCR

	//Detect the blocks
	cv::Mat _canny = gradient.clone();
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(_canny, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	//Store the rectangles by halving their dimensions as the image was upscaled
	for (size_t i = 0; i < contours.size(); i++)
	{
		double area = cv::contourArea(cv::Mat(contours[i]));
		cv::Rect rect = cv::boundingRect(contours[i]);

		//filter 
		if (area > 200 && (rect.width > 32))
		{			
			cv::Rect rect_ = cv::Rect(rect.x / 2, rect.y / 2, rect.width / 2, rect.height / 2);
			//cv::rectangle(UGray, rect, cv::Scalar::all(120), 2);
			blocks.push_back(rect_);

			blockImage b;
			b.img = gray(rect_);
			b.x = rect_.tl().x;
			b.y = rect_.tl().y;
			block_img.push_back(b);
		}
	}

	//Sort w.r.t. Y in ascending
	std::sort(block_img.begin(), block_img.end(), sortAscending);

	//unsharpMask(UGray);
	//cv::imshow("ugray", UGray);

	return true;
}

bool TextRead::sortAscending(blockImage a, blockImage b)
{
	return a.y < b.y;
}


