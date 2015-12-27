#include "LineRemoval.h"


LineRemoval::LineRemoval()
{
}


LineRemoval::~LineRemoval()
{
	//Although this is not required because OpenCV already takes care of releasing the memory 
	original.release();
	gray.release();
}


void LineRemoval::process(const cv::Mat& src, cv::Mat& dst)
{
	//keep a copy of original
	original = src.clone();

	//perform color to gray conversion
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

	//Pre process the image now
	cv::Mat preprocessed;
	preProcess(gray, preprocessed);

	//Applying the skeletonization process to extract the verticals more sharply and accurately
	skeletonizaton(preprocessed);

	//Close the connections vertically
	cv::morphologyEx(preprocessed, preprocessed, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 5)));

	//Extract lines
	std::vector<cv::Vec2f> lines = extractLines(preprocessed);

	cv::Mat temp = original.clone();

	//Draw lines on a white image
	cv::Mat mask = cv::Mat(original.size(), CV_8UC1,cv::Scalar::all(0));
	for (size_t i = 0; i < lines.size(); i++)
	{
		if (lines[i][1] < 1)
		{
			drawLine(lines[i], mask, cv::Scalar::all(255));
			drawLine(lines[i], temp, cv::Scalar(0,0,255));
		}
	}

	cv::imshow("Lines", temp);

	//Add the pixels of gray and the mask  
	cv::add(gray, mask, dst);

	//Grayscale opening
	cv::Mat grayMask;
	cv::morphologyEx(dst, grayMask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 3)));

	//Copy the masked part so that we refill the lost parts except for the lines
	grayMask.copyTo(dst, mask);
}


void LineRemoval::preProcess(const cv::Mat& src, cv::Mat& dst)
{
	cv::Mat temp;

	//src has to be gray
	//Blurring - removing abnormalities 
	cv::GaussianBlur(src, temp, cv::Size(), 1.0, 5.0);

	//Now taking the sobel in vertical directions - y axis
	cv::Mat grad_y,absGradY;
	cv::Sobel(temp, grad_y, CV_16S, 1, 0, 3, 1.0, 0.0, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(grad_y, absGradY);

	//Post processing the sobel
	cv::GaussianBlur(absGradY, absGradY, cv::Size(3, 19), 1.0);
	cv::threshold(absGradY, absGradY, 200, 255, cv::THRESH_BINARY);
	cv::dilate(absGradY, dst, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 15)));
}


std::vector<cv::Vec2f> LineRemoval::extractLines(const cv::Mat& src)
{
	std::vector<cv::Vec2f> lines;

	//First lets filter out useless lines - small in length
	cv::Mat temp = src.clone(); // reason is that the contour function will write over this image and make it unusable further.
	std::vector<std::vector<cv::Point>> contours;

	//find the contours -- basically connected components
	cv::findContours(temp, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	//filtering the contours based on size 
	for (size_t i = 0; i < contours.size(); i++)
	{
		double length = cv::arcLength(cv::Mat(contours[i]), false);

		if (length > 150.) // minimum length that we are looking for
			cv::drawContours(src, contours, i, cv::Scalar::all(255), 1);
		else
			cv::drawContours(src, contours, i, cv::Scalar::all(0), 1);
	}

	//Find the hough lines
	cv::HoughLines(src, lines, 1, CV_PI / 180, 50);

	//std::cout << "\n B lines =" << lines.size() << "\n\n";

	// we need to merge these lines as there will be lot of lines intersecting each other we need to make them one for each intersection point
	// Criteria - intersection point + distance between the lines.
	// or we can check if the lines are parallel and their angle with the x axis is almost 90degrees
	
	std::vector<cv::Vec2f>::iterator itc;
	for (itc = lines.begin(); itc != lines.end(); itc++)
	{
		if ((*itc)[0] == 0 && (*itc)[1] == -100) continue;

		float p1 = (*itc)[0];
		float theta1 = (*itc)[1];

		cv::Point pt1current, pt2current;
		getendpoints((*itc), pt1current, pt2current);

		std::vector<cv::Vec2f>::iterator    pos;

		for (pos = lines.begin(); pos != lines.end(); pos++)
		{
			if (*itc == *pos) continue;

			if (fabs((*pos)[0] - (*itc)[0]) < 20 && fabs((*pos)[1] - (*itc)[1]) < CV_PI * 10 / 180)
			{
				float p = (*pos)[0];
				float theta = (*pos)[1];
				cv::Point pt1, pt2;
				getendpoints((*pos), pt1, pt2);

				if (((double)(pt1.x - pt1current.x)*(pt1.x - pt1current.x) + (pt1.y - pt1current.y)*(pt1.y - pt1current.y) < 64 * 64) &&
					((double)(pt2.x - pt2current.x)*(pt2.x - pt2current.x) + (pt2.y - pt2current.y)*(pt2.y - pt2current.y) < 64 * 64))
				{
					// Merge the two
					(*itc)[0] = ((*itc)[0] + (*pos)[0]) / 2;

					(*itc)[1] = ((*itc)[1] + (*pos)[1]) / 2;

					(*pos)[0] = 0;
					(*pos)[1] = -100;
				}
			}
		}
	}


	//Removal
	itc = lines.begin();
	while (itc != lines.end())
	{
		if ((*itc)[1] == -100)
			itc = lines.erase(itc);
		else
			++itc;
	}

	return lines;
}


void LineRemoval::getendpoints(cv::Vec2f line, cv::Point& p1, cv::Point& p2)
{
	//Equation =>
	// rho = x*cos(theta) + y*sin(theta)

	float rho = line[0];
	float theta1 = line[1];

	if (theta1>CV_PI * 45 / 180 && theta1<CV_PI * 135 / 180)
	{
		p1.x = 0;
		p1.y = rho / sin(theta1);

		p2.x = gray.size().width;
		p2.y = -p2.x / tan(theta1) + rho / sin(theta1);

		//std::cout << "\n H p1 = " << p1 << " p2 = " << p2 << std::endl;
	}
	else
	{
		p1.y = 0;
		p1.x = rho / cos(theta1);

		p2.y = gray.size().height;
		p2.x = rho / cos(theta1) - (p2.y * tan(theta1));

		//std::cout << "\n V p1 = " << p1 << " p2 = " << p2 << std::endl;
	}
}


void LineRemoval::thinningIteration(cv::Mat& im, int iter)
{
	cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

	for (int i = 1; i < im.rows - 1; i++)
	{
		for (int j = 1; j < im.cols - 1; j++)
		{
			uchar p2 = im.at<uchar>(i - 1, j);
			uchar p3 = im.at<uchar>(i - 1, j + 1);
			uchar p4 = im.at<uchar>(i, j + 1);
			uchar p5 = im.at<uchar>(i + 1, j + 1);
			uchar p6 = im.at<uchar>(i + 1, j);
			uchar p7 = im.at<uchar>(i + 1, j - 1);
			uchar p8 = im.at<uchar>(i, j - 1);
			uchar p9 = im.at<uchar>(i - 1, j - 1);

			int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
				(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
				(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
				(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
				marker.at<uchar>(i, j) = 1;
		}
	}

	im &= ~marker;
}



void LineRemoval::skeletonizaton(cv::Mat& im)
{
	im /= 255;

	cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
	cv::Mat diff;

	do {
		thinningIteration(im, 0);
		thinningIteration(im, 1);
		cv::absdiff(im, prev, diff);
		im.copyTo(prev);
	} while (cv::countNonZero(diff) > 0);

	im *= 255;
}



void LineRemoval::drawLine(cv::Vec2f line, cv::Mat &img, cv::Scalar color)
{
		
	float rho = line[0], theta = line[1];

	//std::cout << "\nRho = " << rho << "  theta = " << theta << "\n";

	cv::Point pt1, pt2;
	double a = cos(theta), b = sin(theta);
	double x0 = a*rho, y0 = b*rho;
	pt1.x = cvRound(x0 + img.size().width * (-b));
	pt1.y = cvRound(y0 + img.size().height  * (a));
	pt2.x = cvRound(x0 - img.size().width  * (-b));
	pt2.y = cvRound(y0 - img.size().height  * (a));
	
	cv::line(img, pt1, pt2, color, 2, CV_AA);
}