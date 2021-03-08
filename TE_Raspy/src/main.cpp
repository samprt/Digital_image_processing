#include<stdio.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

void binarize_RGB(Mat src, Scalar R_range, Scalar G_range, Scalar B_range, Mat& binarized){
    // Séparation des channels
    Mat channels[3];
    split(src, channels);

    Mat Mask_B;
    inRange(channels[0], B_range[0],B_range[1], Mask_B);
    Mask_B /= 255;
    Mat Mask_G;
    inRange(channels[1], G_range[0], G_range[1], Mask_G);
    Mask_G /= 255;
    Mat Mask_R;
    inRange(channels[2], R_range[0],R_range[1], Mask_R);
    Mask_R /= 255;

    Mat b_temp;
    b_temp = channels[0].mul(Mask_B);
    Mat g_temp;
    g_temp = channels[1].mul(Mask_G);
    Mat r_temp;
    r_temp = channels[2].mul(Mask_R);


    vector<Mat> temp;
    temp.push_back(b_temp);
    temp.push_back(g_temp);
    temp.push_back(r_temp);
    merge(temp, binarized);
}

int main(int argc, char *argv[]) {

    string filename = "../../images/crayons.jpg";

    // Lire et afficher l'image
    Mat src=imread(filename,IMREAD_COLOR);
    if(src.empty())
    {
        cout << "Could not open file" << endl;
        return -1;
    }

    // Creation d'un trackbar
    namedWindow("Trackbars");
    moveWindow("Trackbars", 2*src.cols, 0);
    int Rmin = 50;
    createTrackbar("Rmin", "Trackbars", &Rmin, 255);
    int Rmax = 50;
    createTrackbar("Rmax", "Trackbars", &Rmax, 255);
    int Gmin = 50;
    createTrackbar("Gmin", "Trackbars", &Gmin, 255);
    int Gmax = 50;
    createTrackbar("Gmax", "Trackbars", &Gmax, 255);
    int Bmin = 50;
    createTrackbar("Bmin", "Trackbars", &Bmin, 255);
    int Bmax = 50;
    createTrackbar("Bmax", "Trackbars", &Bmax, 255);
    imshow("Trackbars", src);

    /*Scalar mean;
    Scalar alpha;
    meanStdDev(src, mean, alpha);*/

    while(true)
    {
        Scalar R_range = Scalar(Rmin, Rmax);
        Scalar G_range = Scalar(Gmin, Gmax);
        Scalar B_range = Scalar(Bmin, Bmax);
        Mat binarized;
        binarized = Mat::zeros(Size(src.rows, src.cols), CV_8UC1);
        binarize_RGB(src, R_range, G_range, B_range, binarized);
        imshow("binarized", binarized);

        char c = waitKey(1);
        if(c==27)
            break;
    }
    
    destroyAllWindows();
    return 0;
}

