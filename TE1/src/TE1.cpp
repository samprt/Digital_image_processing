#include<stdio.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;


int main(int argc, char *argv[])
{

	//aide en ligne création d'une matrice
	//https://docs.opencv.org/3.2.0/d6/d6d/tutorial_mat_the_basic_image_container.html
	//https://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html

	int c=0;

	// Extension à des images réelles présentant des périodicités
	Mat src1=imread("../../images//D1.tif",0);
	//Mat src1=imread("C:/temp/TNI_HEL/testFFT-build-desktop/ImageMireSinusoidale.png",0);
	// 0 : CV_LOAD_IMAGE_GRAYSCALE

	Mat src2=imread("../../images/D4.tif",0);
	//Mat src1=imread("C:/temp/TNI_HEL/testFFT-build-desktop/crayons.jpg",0);
	//Mat src2=imread("bibliotheque.jpg",0);

	if(src1.empty())
	{
		cout << "Could not open file D1.tif" << endl;
		return -1;
	}
			
	if(src2.empty())
	{
		cout << "Could not open file D4.tif" << endl;
		return -1;
	}

	namedWindow("SRC 1");
	imshow("SRC 1", src1);
	namedWindow("SRC 2");
	imshow("SRC 2", src2);
	c = waitKey(0);

	// Etape 1 : expand input image to optimal size : inutile si les tailles des images sont d�j� des multiples de 2 ce qui n'est pas le cas donc utile ici

	//Expand the image to an optimal size. The performance of a DFT is dependent of the image size. 
	//It tends to be the fastest for image sizes that are multiple of the numbers two, three and five. Therefore, 
	//to achieve maximal performance it is generally a good idea to pad border values to the image to get a size with such traits. 
	//The getOptimalDFTSize() returns this optimal size and we can use the copyMakeBorder() function to expand the borders of an image

	Mat padded1;

	//expand src1
	int m = getOptimalDFTSize(src1.rows);
	/* C++: int getOptimalDFTSize(int vecsize): Arrays whose size is a power-of-two (2, 4, 8, 16, 32, ...) are the fastest to process. 
	Though, the arrays whose size is a product of 2�s, 3�s, and 5�s (for example, 300 = 5*5*3*2*2) are also processed quite efficiently.
	The function getOptimalDFTSize returns the minimum number N that is greater than or equal to vecsize so that the DFT of a vector of 
	size N can be processed efficiently. In the current implementation N = 2 p * 3 q * 5 r for some integer p, q, r.
	*/
	int n = getOptimalDFTSize(src1.cols); // on the border add zero values
	copyMakeBorder(src1, padded1, 0, m - src1.rows, 0, n - src1.cols, BORDER_CONSTANT, Scalar::all(0));

	/* Etape 2 : conversion en float 
	Make place for both the complex and the real values. The result of a Fourier Transform is complex. This implies that for each image 
	value the result is two image values (one per component). Moreover, the frequency domains range is much larger than its spatial counterpart. 
	Therefore, we store these usually at least in a float format. Therefore we'll convert our input image to this type and expand it with 
	another channel to hold the complex values
	*/

	Mat planes1[] = {Mat_<float>(padded1), Mat::zeros(padded1.size(), CV_32F)};
	Mat complex_src1;
	merge(planes1, 2, complex_src1);

	// Etape 3 : calcul de la DFT
	// Make the Discrete Fourier Transform. It's possible an in-place calculation (same input as output):

	dft(complex_src1, complex_src1);

	// Etape 4 : calcul du spectre d'amplitude
	// Transform the real and complex values to magnitude. A complex number has a real (Re) and a complex (imaginary - Im) part.
	// The results of a DFT are complex numbers. The magnitude of a DFT is sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)

	split(complex_src1, planes1); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes1[0], planes1[1], planes1[0]); // planes[0] = magnitude
	Mat mag_src1 = planes1[0];

	/*
	Etape 5 : passage au log
	Switch to a logarithmic scale. It turns out that the dynamic range of the Fourier coefficients is too large to be 
	displayed on the screen. We have some small and some high changing values that we can�t observe like this. Therefore 
	the high values will all turn out as white points, while the small ones as black. To use the gray scale values to 
	for visualization we can transform our linear scale to a logarithmic one
	*/

	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	mag_src1 += Scalar::all(1);
	log(mag_src1, mag_src1);

	/*
	Etape 6 : réaménagement des quadrants pour une visualisation avec les basses fréquences au centre
	Crop and rearrange. Remember, that at the first step, we expanded the image? Well, it's time to 
	throw away the newly introduced values. For visualization purposes we may also rearrange the quadrants 
	of the result, so that the origin (zero, zero) corresponds with the image center.
	*/

	// crop the spectrum, if it has an odd number of rows or columns
	mag_src1 = mag_src1(Rect(0, 0, mag_src1.cols & -2, mag_src1.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = mag_src1.cols/2;
	int cy = mag_src1.rows/2;

	Mat q0(mag_src1, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(mag_src1, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(mag_src1, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(mag_src1, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	/*
	Etape 7 : normalisation pour visualisation et visualisation
	Normalize. This is done again for visualization purposes. We now have the magnitudes, however this are 
	still out of our image display range of zero to one. We normalize our values to this range using 
	the normalize() function.
	*/

	// Transform the matrix with float values into a
	// viewable image form (float between values 0 and 1).
	normalize(mag_src1, mag_src1, 0, 1, CV_MINMAX); 

	imshow("Spectre damplitude 1", mag_src1);

	c = waitKey(0);

	//deuxième image
	Mat padded2;

	//expand src2
	m = getOptimalDFTSize( src2.rows );
	n = getOptimalDFTSize( src2.cols ); // on the border add zero values
	copyMakeBorder(src2, padded2, 0, m - src2.rows, 0, n - src2.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes2[] = {Mat_<float>(padded2), Mat::zeros(padded2.size(), CV_32F)};
	Mat complex_src2;
	merge(planes2, 2, complex_src2);

	dft(complex_src2, complex_src2);

	split(complex_src2, planes2); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes2[0], planes2[1], planes2[0]); // planes[0] = magnitude
	Mat mag_src2 = planes2[0];

	mag_src2 += Scalar::all(1);
	log(mag_src2, mag_src2);
	//pareil pour la 2ème image
	// crop the spectrum, if it has an odd number of rows or columns
	mag_src2= mag_src2(Rect(0, 0, mag_src2.cols & -2, mag_src2.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	cx = mag_src2.cols/2;
	cy = mag_src2.rows/2;

	Mat q0_2(mag_src2, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1_2(mag_src2, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2_2(mag_src2, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3_2(mag_src2, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp2;                           // swap quadrants (Top-Left with Bottom-Right)
	q0_2.copyTo(tmp2);
	q3_2.copyTo(q0_2);
	tmp2.copyTo(q3_2);

	q1_2.copyTo(tmp2);                    // swap quadrant (Top-Right with Bottom-Left)
	q2_2.copyTo(q1_2);
	tmp2.copyTo(q2_2);

	normalize(mag_src2, mag_src2, 0, 1, CV_MINMAX);

	imshow("Spectre damplitude 2", mag_src2);
	c = waitKey(0);

	cv::destroyAllWindows();
	src1.release();

	src2.release();

	// ----------------- Personal Code ------------------
	Mat mire = imread("../../images/ImageMireSinusoidale.png");

	if(mire.empty())
	{
		cout << "Could not open file ImageMireSinusoidale.png" << endl;
		return -1;
	}

	namedWindow("Mire");
	imshow("Mire", mire);
	c = waitKey(0);

	Mat padded;

	m = getOptimalDFTSize(mire.rows);
	n = getOptimalDFTSize(mire.cols);
	copyMakeBorder(mire, padded, 0, m - mire.rows, 0, n - mire.cols, BORDER_CONSTANT, Scalar::all(0));

	/*
	Mat planes_mire[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complex_mire;
	merge(planes_mire, 2, complex_mire);
	

	dft(complex_mire, complex_mire);

	split(complex_mire, planes_mire);
	magnitude(planes_mire[0], planes_mire[1], planes_mire[0]);
	Mat mag_mire = planes_mire[0];

	mag_mire += Scalar::all(1);
	log(mag_mire, mag_mire);

	mag_mire = mag_mire(Rect(0, 0, mag_mire.cols & -2, mag_mire.rows & -2));

	cx = mag_mire.cols/2;
	cy = mag_mire.rows/2;

	q0 = Mat(mag_mire, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	q1 = Mat(mag_mire, Rect(cx, 0, cx, cy));  // Top-Right
	q2 = Mat(mag_mire, Rect(0, cy, cx, cy));  // Bottom-Left
	q3 = Mat(mag_mire, Rect(cx, cy, cx, cy)); // Bottom-Right

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(mag_mire, mag_mire, 0, 1, CV_MINMAX); 

	imshow("Spectre damplitude mire", mag_mire);

	c = waitKey(0);
	destroyAllWindows();*/


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

