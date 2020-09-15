#ifndef ENTROPY_H
#define ENTROPY_H

#include <opencv2/opencv.hpp>
using namespace cv;

void globalEntropy(Mat input, Mat output);

void localEntropy(Mat input, Mat output, int radius);

void localWeightedEntropy(Mat input, Mat output, int radius, int *weights);

void localDistanceEntropy(Mat input, Mat output, int radius);

void localGaussianEntropy(Mat input, Mat output, int radius);

void normalEntropy(Mat input, Mat output);

#endif