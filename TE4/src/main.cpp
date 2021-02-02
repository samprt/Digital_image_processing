#include<stdio.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;


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

int decouverte_operation_sur_images_binarisee()
{
	// Lire et afficher l'image
	Mat src = imread("../../images/voilier_oies_blanches.jpg", 0);
	if(src.empty())
	{
		cout << "Could not open file" << endl;
		return -1;
	}
	imshow("src", src);
	

	//Binarisation automatique avec la méthode d'Otsu
	Mat dst;
	double t = otsu("../../images/voilier_oies_blanches.jpg");
	threshold(src, dst, t, 255, THRESH_BINARY);
	namedWindow("dst", WINDOW_AUTOSIZE);
	moveWindow("dst", src.cols, 0);
	imshow("dst", dst);

	// Structuring element utilisée par les méthode d'érosion, dilataion, ouverture et fermeture
	Mat structuringElement = getStructuringElement(MORPH_RECT,  Size(4, 4));

	//Erosion de l'image binarisée
	Mat erode_dst;
	erode(dst, erode_dst, structuringElement);
	namedWindow("eroded dst", WINDOW_AUTOSIZE);
	moveWindow("eroded dst", 0, src.rows);
	imshow("eroded dst", erode_dst);

	//Dilatation de l'image binarisée
	Mat dilate_dst;
	dilate(dst, dilate_dst, structuringElement);
	namedWindow("dilated dst", WINDOW_AUTOSIZE);
	moveWindow("dilated dst", src.cols, src.rows);
	imshow("dilated dst", dilate_dst);
	
	waitKey(0);
	destroyAllWindows();
  
	//Ouverture de l'image binarisée
	Mat open_dst;
	morphologyEx(dst, open_dst, MORPH_OPEN, structuringElement);
	namedWindow("opened dst", WINDOW_AUTOSIZE);
	imshow("opened dst", open_dst);

	//Fermeture de l'image binarisée
	Mat close_dst;
	morphologyEx(dst, close_dst, MORPH_CLOSE, structuringElement);
	namedWindow("closed dst", WINDOW_AUTOSIZE);
	moveWindow("closed dst", open_dst.cols, 0);
	imshow("closed dst", open_dst);

	waitKey(0);
	destroyAllWindows();

	//Ouverture fermeture de l'image binarisée
	Mat open_close_dst;
	morphologyEx(dst, open_close_dst, MORPH_OPEN, structuringElement);
	morphologyEx(open_close_dst, open_close_dst, MORPH_CLOSE, structuringElement);
	namedWindow("opened-closed dst", WINDOW_AUTOSIZE);
	imshow("opened-closed dst", open_close_dst);

	waitKey(0);
	destroyAllWindows();
}

int etiquetage()
{
	// Lire et afficher l'image
	Mat src = imread("../../images/voilier_oies_blanches.jpg", 0);
	if(src.empty())
	{
		cout << "Could not open file" << endl;
		return -1;
	}
	imshow("src", src);

	//Binarisation automatique avec la méthode d'Otsu
	Mat dst;
	double t = otsu("../../images/voilier_oies_blanches.jpg");
	threshold(src, dst, t, 255, THRESH_BINARY_INV);
	namedWindow("dst", WINDOW_AUTOSIZE);
	moveWindow("dst", 2*src.cols, 0);
	imshow("dst", dst);

	//Etiquetage
	Mat labels;
	int birds = connectedComponents(dst, labels);
	cout << "Number of labels = " << birds << endl;
	Mat seeMyLabels;
	normalize(labels, seeMyLabels, 50, 255, NORM_MINMAX, CV_8U);
	/*for(int r=0 ; r < seeMyLabels.rows ; r++)
	{
		for(int c=0 ; c < seeMyLabels.cols ; c++)
		{
			if(seeMyLabels.at<uchar>(r, c) = 50)
				seeMyLabels.at<uchar>(r, c) = 0;
		}
	}*/
	imshow("labels", seeMyLabels);

	waitKey(0);
	destroyAllWindows();
}

int test_contours()
{
	// Lire et afficher l'image
	Mat src = imread("../../images/voilier_oies_blanches.jpg", 0);
	if(src.empty())
	{
		cout << "Could not open file" << endl;
		return -1;
	}
	imshow("src", src);

	//Binarisation automatique avec la méthode d'Otsu
	Mat dst;
	double t = otsu("../../images/voilier_oies_blanches.jpg");
	threshold(src, dst, t, 255, THRESH_BINARY_INV);
	namedWindow("dst", WINDOW_AUTOSIZE);
	moveWindow("dst", 2*src.cols, 0);
	imshow("dst", dst);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(dst, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

	Mat temp = imread("../../images/voilier_oies_blanches.jpg", 0);
	Mat img(temp.size(), CV_8UC3);
	cvtColor(temp, img, COLOR_GRAY2RGB);

	for (int i=0 ; i<contours.size() ; i++)
	{
		drawContours(img, contours, i, Scalar(0, 255, 255), 1);
	}
	imshow("contours", img);

	waitKey(0);
	destroyAllWindows();
}

int main(int argc, char *argv[])
{
	//decouverte_operation_sur_images_binarisee();
	//etiquetage();
	test_contours();

	return 0;
}

