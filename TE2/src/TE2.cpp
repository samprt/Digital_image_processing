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

	/*
	Egalisation de l'histogramme : Cette fonction permet d'étendre la plage d'intensité de l'image (min devient plus petit et max devient plus grand)
	*/
	Mat equalized_src;
	equalizeHist(src, equalized_src);
	imshow("Equalized src", equalized_src);
	
	// Calcul de l'histogramme de l'image égalisée
	Mat equalized_hist;
	int equalized_histSize = 256;
	float equalized_range[] = { 0, 256 }; //the upper boundary is exclusive
  const float* equalized_histRange = { range };
	calcHist( &equalized_src, 1, 0, Mat(), equalized_hist, 1, &equalized_histSize, &equalized_histRange, uniform, accumulate );

  Mat equalized_histImage( hist_h, hist_w, CV_8UC3, Scalar(0,0,0) );
	normalize(equalized_hist, equalized_hist, 0, equalized_histImage.rows, NORM_MINMAX, -1, Mat());

	for( int i = 1; i < histSize; i++ )
	{
			line( equalized_histImage, Point( bin_w*(i-1), hist_h - cvRound(equalized_hist.at<float>(i-1)) ),
						Point( bin_w*(i), hist_h - cvRound(equalized_hist.at<float>(i)) ),
						Scalar( 255, 255, 255), 2, 8, 0  );
	}
  imshow("Histogramme égalisé", equalized_histImage);

	/*
	// Enregistrement de l'image égalisée
	imwrite("../../images/src_égalisée.jpg", equalized_src);
	*/

	Mat equalized_src_2;
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->apply(src, equalized_src_2);
	imshow("Equalized src 2", equalized_src_2);


	waitKey(0);
	destroyAllWindows();
}

int filter(string filename)
{
	// Lire et afficher l'image
	Mat src=imread(filename,0);
	if(src.empty())
	{
		cout << "Could not open file" << endl;
		return -1;
	}
	imshow("src", src);

	// Filtre moyenneur : Size(i, i) défini les coefficients du filtre (plus grand = plus flou)
	Mat filtered_src;
	string windowName = "Filtered src at ";
	for (int i=1 ; i <= 15 ; i+=2)
	{
		blur(src, filtered_src, Size(i, i));
		windowName = "Filtered src at ";
		windowName += to_string(i);
		imshow(windowName, filtered_src);
		waitKey(0);
		destroyWindow(windowName);
	}

	// Filtre Gaussien : i défini l'écart type à utilisé par le filtre (plus grand = plus flou)
	windowName = "Gaussian filtered src at ";
	for (int i=1 ; i <= 15 ; i+=2)
	{
		GaussianBlur(src, filtered_src, Size(0, 0), i, i);
		windowName = "Gaussian filtered src at ";
		windowName += to_string(i);
		imshow(windowName, filtered_src);
		waitKey(0);
		destroyWindow(windowName);
	}

	// Filtre médian : i défini le nombre de pixels voisins à prendre pour caluler la valeur médianne
	windowName = "Median filtered src at ";
	for (int i=1 ; i <= 15 ; i+=2)
	{
		medianBlur(src, filtered_src, i);
		windowName = "Median filtered src at ";
		windowName += to_string(i);
		imshow(windowName, filtered_src);
		waitKey(0);
		destroyWindow(windowName);
	}


	destroyAllWindows();
}

int main(int argc, char *argv[])
{
	calculate_hist("../../images/pout.jpg");

	calculate_hist("../../images/paysage_sombre.png");

	filter("../../images/Image_epave.jpg");

	destroyAllWindows();
  return 0;
}

