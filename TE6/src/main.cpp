#include<stdio.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

int spliting(string filename) {
    // Lire et afficher l'image
    Mat src = imread(filename, IMREAD_COLOR);
    if (src.empty()) {
        cout << "Could not open file" << endl;
        return -1;
    }
    imshow("src", src);

    // Séparation des channels
    Mat bgr_channels[3];
    split(src, bgr_channels);

    // Affichage
    namedWindow("blue", WINDOW_AUTOSIZE);
    moveWindow("blue", src.cols, 0);
    imshow("blue", bgr_channels[0]);
    namedWindow("green", WINDOW_AUTOSIZE);
    moveWindow("green", 2 * src.cols, 0);
    imshow("green", bgr_channels[1]);
    namedWindow("red", WINDOW_AUTOSIZE);
    moveWindow("red", 3 * src.cols, 0);
    imshow("red", bgr_channels[2]);

    waitKey(0);
}

int convert_2_HSV(string filename) {
    // Lire l'image
    Mat src = imread(filename, IMREAD_COLOR);
    if (src.empty()) {
        cout << "Could not open file" << endl;
        return -1;
    }

    // Convertion en HSV
    cvtColor(src, src, CV_BGR2HSV);

    namedWindow("src_hsv", WINDOW_AUTOSIZE);
    moveWindow("src_hsv", 0, src.rows);
    imshow("src_hsv", src);

    // Séparation des channels
    Mat hsv_channels[3];
    split(src, hsv_channels);

    // Affichage
    namedWindow("h", WINDOW_AUTOSIZE);
    moveWindow("h", src.cols, src.rows);
    imshow("h", hsv_channels[0]);
    namedWindow("s", WINDOW_AUTOSIZE);
    moveWindow("s", 2 * src.cols, src.rows);
    imshow("s", hsv_channels[1]);
    namedWindow("v", WINDOW_AUTOSIZE);
    moveWindow("v", 3 * src.cols, src.rows);
    imshow("v", hsv_channels[2]);

    waitKey(0);
}

int change_H_channel(string filename) {
    // Lire l'image
    Mat src = imread(filename, IMREAD_COLOR);
    if (src.empty()) {
        cout << "Could not open file" << endl;
        return -1;
    }

    // Convertion en HSV
    cvtColor(src, src, CV_BGR2HSV);

    // Augmentation de 60 du channel h
    Mat hsv;
    hsv = src;
    for (int i = 0; i < hsv.rows; i++) {
        for (int j = 0; j < hsv.cols; j++) {
            hsv.at<Vec3b>(i, j)[0] += 60;
            if (hsv.at<Vec3b>(i, j)[0] > 180)
                hsv.at<Vec3b>(i, j)[0] -= 180;
        }
    }
    namedWindow("Changed h", WINDOW_AUTOSIZE);
    moveWindow("Changed h", 0, 2 * src.rows);
    imshow("Changed h", hsv);

    // Convertion en BGR
    Mat out;
    cvtColor(hsv, out, CV_HSV2BGR);
    namedWindow("Converted h", WINDOW_AUTOSIZE);
    moveWindow("Converted h", src.cols, 2 * src.rows);
    imshow("Converted h", out);

    waitKey(0);
}

int change_S_channel(string filename) {
    // Lire l'image
    Mat src = imread(filename, IMREAD_COLOR);
    if (src.empty()) {
        cout << "Could not open file" << endl;
        return -1;
    }

    // Convertion en HSV
    cvtColor(src, src, CV_BGR2HSV);

    // Multiplication par 2 du channel S
    Mat hsv;
    hsv = src;
    for (int i = 0; i < hsv.rows; i++) {
        for (int j = 0; j < hsv.cols; j++) {
            hsv.at<Vec3b>(i, j)[1] *= 2;
            if (hsv.at<Vec3b>(i, j)[1] > 255)
                hsv.at<Vec3b>(i, j)[1] = 255;
        }
    }
    namedWindow("Changed s", WINDOW_AUTOSIZE);
    moveWindow("Changed s", 2 * src.cols, 2 * src.rows);
    imshow("Changed s", hsv);

    // Convertion en BGR
    Mat out;
    cvtColor(hsv, out, CV_HSV2BGR);
    namedWindow("Converted s", WINDOW_AUTOSIZE);
    moveWindow("Converted s", 3 * src.cols, 2 * src.rows);
    imshow("Converted s", out);

    waitKey(0);
}

int change_V_channel(string filename) {
    // Lire l'image
    Mat src = imread(filename, IMREAD_COLOR);
    if (src.empty()) {
        cout << "Could not open file" << endl;
        return -1;
    }

    // Convertion en HSV
    cvtColor(src, src, CV_BGR2HSV);

    // Multiplication par 1.5 du channel V
    Mat hsv;
    hsv = src;
    for (int i = 0; i < hsv.rows; i++) {
        for (int j = 0; j < hsv.cols; j++) {
            hsv.at<Vec3b>(i, j)[2] *= 1.5;
            if (hsv.at<Vec3b>(i, j)[2] > 255)
                hsv.at<Vec3b>(i, j)[2] = 255;
        }
    }
    namedWindow("Changed v", WINDOW_AUTOSIZE);
    moveWindow("Changed v", 0, 3 * src.rows);
    imshow("Changed v", hsv);

    // Convertion en BGR
    Mat out;
    cvtColor(hsv, out, CV_HSV2BGR);
    namedWindow("Converted v", WINDOW_AUTOSIZE);
    moveWindow("Converted v", src.cols, 3 * src.rows);
    imshow("Converted v", out);

    waitKey(0);
}

Mat binarize_HSV(Mat &HSV, int Hmin, int Hmax) {
    // TODO : Faire en sorte de pouvoir donner Hmin et Hmax sur 0-360 sans singularité
    // Split matrix
    Mat channels[3];
    split(HSV, channels);

    // Find max saturation value (S) and compute Smin
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    minMaxLoc(channels[1], &minVal, &maxVal, &minLoc, &maxLoc);
    int Smax = int(maxVal);
    int Smin = int(0.1 * Smax);

    // -------------- Prune unwanted values
    // Create S mask (mask = 1 if Smin < S < Smax ; 0 otherwise)
    Mat Mask_S;
    inRange(channels[1], Mat::ones(HSV.rows, HSV.cols, channels[1].type()) * Smin,
            Mat::ones(HSV.rows, HSV.cols, channels[1].type()) * Smax, Mask_S);
    Mask_S /= 255;
    // Multiply H channel by mask (Keeps the colors that are saturated enough)
    Mat Ij = channels[0].mul(Mask_S);
    // Create V mask (mask = 1 if 60 < V < 230 ; 0 otherwise)
    Mat Mask_V;
    inRange(channels[2], Mat::ones(HSV.rows, HSV.cols, channels[2].type()) * 60,
            Mat::ones(HSV.rows, HSV.cols, channels[2].type()) * 230, Mask_V);
    Mask_V /= 255;
    // Multiply H channel by mask (Keeps the colors that are not black, gray or white)
    Ij = Ij.mul(Mask_V);
    // Keep the values that are between Hmin and Hmax ( Ibin = 1 if Hmin < Ij < Hmax ; 0 otherwise)
    Mat Ibin;
    inRange(Ij, Mat::ones(HSV.rows, HSV.cols, Ij.type()) * Hmin,
            Mat::ones(HSV.rows, HSV.cols, Ij.type()) * Hmax, Ibin);
    imshow("binarised", Ibin);
    Ibin /= 255;

    return Ibin;
}

int seuillage_HSV(string filename) {
    // Lire l'image
    Mat src = imread(filename, IMREAD_COLOR);
    if (src.empty()) {
        cout << "Could not open file" << endl;
        return -1;
    }
    imshow("src", src);

    // Convertion en HSV
    Mat HSV;
    cvtColor(src, HSV, CV_BGR2HSV);

    int Hmin = 90;
    int Hmax = 150;

    Mat Ibin =  binarize_HSV(HSV, Hmin, Hmax);

    Mat segmented;
    bitwise_and(src, src, segmented, Ibin);
    imshow("segmented img", segmented);

    waitKey(0);
}




int main(int argc, char *argv[]) {

    string filename = "../../images/house.jpg";

    spliting(filename);

    convert_2_HSV(filename);

    change_H_channel(filename);

    change_S_channel(filename);

    change_V_channel(filename);

    // destroyAllWindows();

    seuillage_HSV(filename);

    destroyAllWindows();
    return 0;
}

