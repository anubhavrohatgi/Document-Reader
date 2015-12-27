#include "OCR.h"


OCR::OCR(std::string tessdataPath)
{
	ocr_path = tessdataPath;
}


OCR::~OCR()
{
	tess.Clear();
}

bool OCR::initOCR()
{
	if (tess.Init(ocr_path.c_str(), "eng", tesseract::OEM_DEFAULT) == 0)  
	{
		tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
		tess.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz0123456789");
		std::cout << "\nSuccessful Initialize the OCR Engine\n";
		return true;
	}
	else
	{
		std::cout << "\nCannot Initialize the OCR Enginer\n";
		return false;
	}
}


std::string OCR::performOCR(const cv::Mat& src)
{
	cv::Mat block;
	cv::pyrUp(src, block);
	unsharpMask(block);

	
	//Binarize image
	//cv::threshold(block, block, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
	tess.TesseractRect((uchar*)block.data, 1, block.step, 0, 0, block.cols, block.rows);
	
	//TODO:
	//Recognize the text

	return "";
}


void OCR::unsharpMask(cv::Mat& im)
{
	cv::Mat tmp;
	cv::GaussianBlur(im, tmp, cv::Size(5, 5), 5);
	cv::addWeighted(im, 1.5, tmp, -0.5, 0, im);
}