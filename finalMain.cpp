
#include <iostream>
#include <opencv2\opencv.hpp>


#include "DocumentDeskew.h"
#include "LineRemoval.h"
#include "TextRead.h"
#include "OCR.h"




int main()
{
	cv::Mat img = cv::imread("f:/st.jpg");

	if (img.empty())
		return -1;

	//The pre processing steps are done by each class according to their respective demands.

	//Deskew the document and detect the boundaries - In this we deskew w.r.t. the boundary
	//In order to achieve the best deskewing we should do it w.r.t. text
	//cause it does matter in steps like line removal etc. when the lines are not perpendicular to respective axes
	DocumentDeskew instance(img);
	cv::Mat warped = instance.processDeskew();
	cv::imshow("Step 1 : Deskewed", warped);


	//Remove the vertical lines
	LineRemoval lInstance;
	cv::Mat lines;
	lInstance.process(warped, lines);
	cv::imshow("Step 2 :Lines Removed", lines);

	//Detect Text
	TextRead textInstance("table.txt",warped);

	textInstance.process(lines);

	std::vector<blockImage> blocks = textInstance.getBlocks();

	//Pass the blocks to OCR
	std::string tesPath = "C:\\libs\\tesseract\\tessdata";
	OCR ocrInstance(tesPath);

	ocrInstance.initOCR();

	//for (size_t i = 0; i < blocks.size(); i++)
	ocrInstance.performOCR(blocks[0].img);




	cv::waitKey(0);
	//system("pause");
	return 0;
}