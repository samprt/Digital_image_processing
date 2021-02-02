#include<stdio.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

int calculate_hist(string filename)
{
	// Lire et afficher l'image
	Mat src=imread(filename,0);
	if(src.empty())
	{
		cout << "Could not open file" << endl;
		return -1;
	}
	imshow("src", src);
	
	// Calcul des valeurs de luminance
	double min, max;
	minMaxLoc(src, &min, &max);
	cout << "Luminance min = " << min << endl;
	cout << "Luminance max = " << max << endl;
	Scalar mean, stdDev;
	meanStdDev(src, mean, stdDev);
	cout << "Luminance moyenne = " << mean[0] << endl;
	cout << "Ecart type de luminances = " << stdDev[0] << endl;

	// Calcul de l'histogramme de l'image
	Mat hist;
	int histSize = 256;
	float range[] = { 0, 256 }; //the upper boundary is exclusive
  const float* histRange = { range };
	bool uniform = true, accumulate = false;
	calcHist( &src, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

	int hist_w = 512, hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );
  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for( int i = 1; i < histSize; i++ )
	{
			line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ),
						Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
						Scalar( 255, 255, 255), 2, 8, 0  );
	}
  imshow("Histogramme", histImage);

	waitKey(0);
	destroyAllWindows();
}

double otsu(string filename)
{
	// Lire et afficher l'image
	Mat src=imread(filename,0);
	if(src.empty())
	{
		cout << "Could not open file" << endl;
		return -1;
	}

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


int main(int argc, char *argv[])
{
	// Lire et afficher l'image
	Mat src = imread("../../images/voilier_oies_blanches.jpg", 0);
	if(src.empty())
	{
		cout << "Could not open file" << endl;
		return -1;
	}
	imshow("src", src);
	waitKey(0);

	// Afficher son histogramme
	calculate_hist("../../images/voilier_oies_blanches.jpg");
	/*
	On peut constater qu'il y a 2 pics sur l'histogramme, une segmentation binaire est donc adaptée
	*/

	// Ségmentation binaire en utilisant différent type
	Mat dst;
	double min, max;
	minMaxLoc(src, &min, &max);
	threshold(src, dst, (max + min)/2, 255, THRESH_BINARY);
	imshow("dst", dst);
	waitKey(0);
	destroyWindow("dst");
	threshold(src, dst, (max + min)/2, 255, THRESH_BINARY_INV);
	imshow("dst", dst);
	waitKey(0);
	destroyWindow("dst");
	threshold(src, dst, (max + min)/2, 255, THRESH_TRUNC);
	imshow("dst", dst);
	waitKey(0);
	destroyWindow("dst");
	threshold(src, dst, (max + min)/2, 255, THRESH_TOZERO);
	imshow("dst", dst);
	waitKey(0);
	destroyWindow("dst");
	threshold(src, dst, (max + min)/2, 255, THRESH_TOZERO_INV);
	imshow("dst", dst);
	waitKey(0);
	destroyWindow("dst");

	// Creation d'un trackbar
	namedWindow("Trackbar");
	int threshold_value = 50;
	createTrackbar("Threshold", "Trackbar", &threshold_value, max);

	while(true)
	{
		threshold(src, dst, threshold_value, 255, THRESH_BINARY);
		imshow("dst", dst);

		char c = waitKey(1);
		if(c==27)
			break;
	}
	destroyAllWindows();
	//src.release();
	//dst.release();

	// Segmentation de l'image img_ds.jpg
	src = imread("../../images/img_ds.jpg", 0);
	if(src.empty())
	{
		cout << "Could not open file" << endl;
		return -1;
	}
	imshow("img_ds", src); 
	namedWindow("Trackbar");
	threshold_value = 50;
	createTrackbar("Threshold", "Trackbar", &threshold_value, max);

	while(true)
	{
		threshold(src, dst, threshold_value, 255, THRESH_BINARY);
		imshow("dst", dst);

		char c = waitKey(1);
		if(c==27)
			break;
	}
	destroyAllWindows();

	//Binarisation automatique avec la méthode d'Otsu
	src = imread("../../images/voilier_oies_blanches.jpg", 0);
	if(src.empty())
	{
		cout << "Could not open file" << endl;
		return -1;
	}
	double t = otsu("../../images/voilier_oies_blanches.jpg");
	threshold(src, dst, t, 255, THRESH_BINARY);
	imshow("dst", dst);

	waitKey(0);
	destroyAllWindows();
  return 0;
}

