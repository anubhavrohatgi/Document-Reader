#include "DocumentDeskew.h"


DocumentDeskew::DocumentDeskew(cv::Mat src) : original(src)
{
	cv::cvtColor(original, gray, cv::COLOR_BGR2BGRA);
}


DocumentDeskew::~DocumentDeskew()
{
}

double DocumentDeskew::findAngle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
	double x1 = pt1.x - pt0.x;
	double y1 = pt1.y - pt0.y;
	double x2 = pt2.x - pt0.x;
	double y2 = pt2.y - pt0.y;
	return (x1*x2 + y1*y2) / sqrt((x1*x1 + y1*y1)*(x2*x2 + y2*y2) + 1e-10); // 1e-10 is to introduce some error to avaoid divide by zero

}

bool DocumentDeskew::lArea(std::vector<cv::Point> a1, std::vector<cv::Point> a2)
{
	double area1 = fabs(cv::contourArea(cv::Mat(a1)));
	double area2 = fabs(cv::contourArea(cv::Mat(a2)));
	return area1 >= area2;
}

void DocumentDeskew::reorderPoints(std::vector<cv::Point2f>& corners, cv::Point2f center)
{
	/* order of points 
	1st --------- 2nd
	|				|
	|				|
	4th --------- 3rd
	*/
	std::vector<cv::Point2f> top, bot;

	for (int i = 0; i < corners.size(); i++)
	{
		if (corners[i].y < center.y)
			top.push_back(corners[i]);
		else
			bot.push_back(corners[i]);
	}

	cv::Point2f tl = top[0].x > top[1].x ? top[1] : top[0];
	cv::Point2f tr = top[0].x > top[1].x ? top[0] : top[1];
	cv::Point2f bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
	cv::Point2f br = bot[0].x > bot[1].x ? bot[0] : bot[1];

	corners.clear();
	corners.push_back(tl);
	corners.push_back(tr);
	corners.push_back(br);
	corners.push_back(bl);
}

bool DocumentDeskew::isContourQualifiedQuad(std::vector<cv::Point> contour,std::vector<cv::Point>& approx)
{
	//std::vector<cv::Point> approx;
	// approximate contour with accuracy proportional
	// to the contour perimeter
	cv::approxPolyDP(cv::Mat(contour), approx, cv::arcLength(cv::Mat(contour), true)*0.02, true);

	double area = fabs(cv::contourArea(cv::Mat(approx)));
	double uL = static_cast<float>(original.size().area()*0.95);
	double lL = static_cast<float>(original.size().area()*0.70);

	if (approx.size() == 4 && (area > lL && area <uL ) && cv::isContourConvex(cv::Mat(approx)))
	{
		double maxCosine = 0;

		for (int j = 2; j < 5; j++)
		{
			double cosine = fabs(findAngle(approx[j % 4], approx[j - 2], approx[j - 1]));
			maxCosine = MAX(maxCosine, cosine);
		}

		if (maxCosine < 0.3)
			return true;
		else
			return false;
	}
	else
		return false;
}

void DocumentDeskew::findQuads(const cv::Mat& src, std::vector<std::vector<cv::Point>>& quad, int threshLimit)
{
	cv::Mat temp;

	//Blur the image and make the edges thin and strong
	cv::medianBlur(src, temp, 9);

	cv::Mat _gray(temp.size(), CV_8U);
	cv::Mat _canny;

	std::vector<std::vector<cv::Point>> contours;

	//find quad in each channel
	for (int c = 0; c < 3; c++)
	{
		int channel[] = { c, 0 };
		cv::mixChannels(&temp, 1, &_gray, 1, channel, 1);

		//Apply many thresholding limits on Canny
		for (int l = 0; l < threshLimit; l++){

			if (l == 0){
				cv::Canny(_gray, _canny, 10, 20);
				cv::dilate(_canny, _canny, cv::Mat());
			}

			else{
				//this thresholds the pixels that are greater than the respective threshold levels
				_canny = _gray >= (l + 1) * 255 / threshLimit;
			}

			cv::findContours(_canny, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
			
			for (size_t i = 0; i < contours.size(); i++)
			{
				std::vector<cv::Point> approx;
				if (isContourQualifiedQuad(contours[i], approx))
					quad.push_back(approx);
				else
					continue;
			}

		}//threshlevel loop

	}//channel iterator loop

	std::sort(quad.begin(), quad.end(), lArea);
}


void DocumentDeskew::searchDocumentBoundary(const cv::Mat& src, std::vector<cv::Point>& block)
{

	cv::Mat temp;

	//Blur the image and make the edges thin and strong
	cv::GaussianBlur(src, temp, cv::Size(5, 5), 1.0, 0.0);
	cv::medianBlur(temp, temp, 9);


	cv::Mat _gray(temp.size(), CV_8U);
	cv::Mat _canny;

	cv::Canny(temp, _canny, 125, 200);

	std::vector<std::vector<cv::Point>> contours,quad;

	cv::findContours(_canny, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);


	for (size_t i = 0; i < contours.size(); i++)
	{
		std::vector<cv::Point> approx;
		if (isContourQualifiedQuad(contours[i], approx))
			quad.push_back(approx);
		else
			continue;
	}

	std::sort(quad.begin(), quad.end(), lArea);

	block.assign(quad[0].begin(),quad[0].end());
}



cv::Mat DocumentDeskew::processDeskew()
{	
	std::vector<cv::Point2f> boundary;
	std::vector<std::vector<cv::Point>> conts;
	//searchDocumentBoundary(this->gray, boundary);
	//conts.push_back(boundary);
	//We can always use the searchdocumentBoundary fn by replacing the necessary io below

	findQuads(this->original, conts, 2);
	boundary.assign(conts[0].begin(), conts[0].end());

#ifdef _DEBUG

	cv::Mat tempColor = original.clone();
	cv::drawContours(tempColor, conts, 0, cv::Scalar(0, 0, 255), 2);
	cv::imshow("Document Boundary", tempColor);
	cv::waitKey(0);
#endif
	
	//Deskew now since we have the quad already

	//Get the centroid of the quad
	cv::Point2f center(0,0);
	for (int i = 0; i < boundary.size(); i++)
		center += boundary[i];
	center *= (1. / boundary.size());

	//Reorder the points in the standard opencv format
	reorderPoints(boundary, center);		

	return warpImage(boundary);
}


cv::Mat DocumentDeskew::warpImage(std::vector<cv::Point2f> boundaryCorners)
{
	//Calculate the width & height of the detected document only
	int distx = static_cast<int>(sqrt(pow((boundaryCorners[1].x - boundaryCorners[0].x), 2) + pow((boundaryCorners[1].y - boundaryCorners[0].y), 2)));
	int disty = static_cast<int>(sqrt(pow((boundaryCorners[3].x - boundaryCorners[0].x), 2) + pow((boundaryCorners[3].y - boundaryCorners[0].y), 2)));

	cv::Mat warped = cv::Mat::zeros(disty, distx, CV_8UC3);

	std::vector<cv::Point2f> quad_pts;
	quad_pts.push_back(cv::Point2f(0, 0));
	quad_pts.push_back(cv::Point2f(warped.cols, 0));
	quad_pts.push_back(cv::Point2f(warped.cols, warped.rows));
	quad_pts.push_back(cv::Point2f(0, warped.rows));

	//perform the perspective transformation
	cv::Mat transmtx = cv::getPerspectiveTransform(boundaryCorners, quad_pts);
	cv::warpPerspective(original, warped, transmtx, warped.size());

	cv::Rect roi = cv::Rect(5,5,distx-5,disty-5); // using this to remove some borderline artifacts

	return warped(roi);
}