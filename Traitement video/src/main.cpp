#include<stdio.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

int simple_video_capture()
{
	VideoCapture cap(0);
	if(!cap.isOpened())
	{
		cout << "Could not get webcam feed" << endl;
		return -1;
	}
	int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH); 
	int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	bool recording = false;

	VideoWriter video("../../images/video_capture.",CV_FOURCC('M','J','P','G'), 15, Size(frame_width,frame_height));

	while(1)
	{
		Mat frame;
		cap >> frame;

		if(frame.empty())
			break;
		
		imshow("Camera", frame);

		char c=(char)waitKey(25);
		if(c==27)
			break;
		if(c=='c')
			imwrite("../../images/capture.jpg", frame);
		if(c=='r')
			recording = !recording;
		if(c > 48 && c < 58)
		{
			cout << "Capturing frames" << endl;
			string filename;
			for(int i=1 ; i <= c - 48 ; i++)
			{
				filename = "../../images/frame";
				filename += to_string(i) + ".jpg";
				imwrite(filename, frame);
			}
		}
		if(recording)
			video.write(frame);
	}
	cap.release();
	video.release();

	cv::destroyAllWindows();

  return 0;
}



double otsu(Mat src)
{
	int bins_num = 256;

	// Get the histogram
	long double histogram[256];

	// initialize all intensity values to 0
	for(int i = 0; i < 256; i++)
			histogram[i] = 0;

	// calculate the no of pixels for each intensity values
	for(int y = 0; y < src.rows; y++)
			for(int x = 0; x < src.cols; x++)
					histogram[(int)src.at<uchar>(y,x)]++;

	// Calculate the bin_edges
	long double bin_edges[256];
	bin_edges[0] = 0.0;
	long double increment = 0.99609375;
	for(int i = 1; i < 256; i++)
			bin_edges[i] = bin_edges[i-1] + increment;

	// Calculate bin_mids
	long double bin_mids[256];
	for(int i = 0; i < 256; i++)
		bin_mids[i] = (bin_edges[i] + bin_edges[i+1])/2;

	// Iterate over all thresholds (indices) and get the probabilities weight1, weight2
	long double weight1[256];
	weight1[0] = histogram[0];
	for(int i = 1; i < 256; i++)
		weight1[i] = histogram[i] + weight1[i-1];

	int total_sum=0;
	for(int i = 0; i < 256; i++)
			total_sum = total_sum + histogram[i];
	long double weight2[256];
	weight2[0] = total_sum;
	for(int i = 1; i < 256; i++)
		weight2[i] = weight2[i-1] - histogram[i - 1];

	// Calculate the class means: mean1 and mean2
	long double histogram_bin_mids[256];
	for(int i = 0; i < 256; i++)
		histogram_bin_mids[i] = histogram[i] * bin_mids[i];

	long double cumsum_mean1[256];
	cumsum_mean1[0] = histogram_bin_mids[0];
	for(int i = 1; i < 256; i++)
		cumsum_mean1[i] = cumsum_mean1[i-1] + histogram_bin_mids[i];

	long double cumsum_mean2[256];
	cumsum_mean2[0] = histogram_bin_mids[255];
	for(int i = 1, j=254; i < 256 && j>=0; i++, j--)
		cumsum_mean2[i] = cumsum_mean2[i-1] + histogram_bin_mids[j];

	long double mean1[256];
	for(int i = 0; i < 256; i++)
		mean1[i] = cumsum_mean1[i] / weight1[i];

	long double mean2[256];
	for(int i = 0, j = 255; i < 256 && j >= 0; i++, j--)
		mean2[j] = cumsum_mean2[i] / weight2[j];

	// Calculate Inter_class_variance
	long double Inter_class_variance[255];
	long double dnum = 10000000000;
	for(int i = 0; i < 255; i++)
		Inter_class_variance[i] = ((weight1[i] * weight2[i] * (mean1[i] - mean2[i+1])) / dnum) * (mean1[i] - mean2[i+1]);
		

	// Maximize interclass variance
	long double maxi = 0;
	int getmax = 0;
	for(int i = 0;i < 255; i++){
		if(maxi < Inter_class_variance[i]){
			maxi = Inter_class_variance[i];
			getmax = i;
		}
	}
	return bin_mids[getmax];
}

int binary_segmentation_video()
{
	VideoCapture cap(0);
	if(!cap.isOpened())
	{
		cout << "Could not get wabcam feed" << endl;
		return -1;
	}

	while(true)
	{
		Mat frame;
		cap >> frame;
		cvtColor(frame, frame, COLOR_RGB2GRAY);

		if(frame.empty())
			break;
		
		double t = otsu(frame);
		Mat out;
		threshold(frame, out, t, 255, THRESH_BINARY);
		imshow("Threshold feed", out);

		char c=(char)waitKey(25);
		if(c==27)
			break;
	}


}

int main(int argc, char *argv[])
{
	//simple_video_capture();
	binary_segmentation_video();

	return 0;
}

