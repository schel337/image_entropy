#include <stdio.h>
#include <math.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include "Entropy.h"
using namespace cv;

void globalEntropy(Mat input, Mat output) {
	int nRows = input.rows;
	int nCols = input.cols;
	if (input.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }
    int histo[256] = {0};
    int x, y;
    uchar *row;
    for (x=0; x<nRows; x++) {
    	row = input.ptr(x);
    	for (y=0; y<nCols; y++) {
    		histo[row[y]]++;
		}
    }
    double prob;
	Mat histoMat(1, 256, CV_8U);
    uchar* histoMatPtr= histoMat.ptr();
    double size = input.rows * input.cols;
    for (int i=0; i<256; i++) {
    	prob = histo[i] / size;
    	if (prob != 0) {
    		histoMatPtr[i] = (uchar) (-255*log2(prob)*prob);
    	} else {
    		histoMatPtr[i] = 0;
    	}
    }
    LUT(input, histoMat, output);
}

void localEntropy(Mat input, Mat output, int radius) {
	const int diam = 2*radius+1;
	int nRows = input.rows;
	int nCols = input.cols;
    Mat histosMat(input.rows, input.cols, CV_8U, Scalar(0));
    uchar *histosRow;
    int X, Y, dX, dY, left, right, top, bottom;
	uchar *rows[diam];
	for (X=0; X<nRows; X++) {
		bottom = X>radius ? -radius : -X;
		top = X+radius<nRows ? radius : nRows-X-1;
		histosRow = histosMat.ptr(X);
		for (dX=bottom; dX<=top; dX++) {
			rows[radius+dX] = input.ptr(X+dX);
		}
		for (Y=0; Y<nCols; Y++) {
			left = Y>radius ? -radius : -Y;
			right = Y+radius<nCols ? radius : nCols-Y-1;
			for (dX=bottom; dX<=top; dX++) {
				for (dY=left; dY<=right; dY++) {
					if (rows[radius+dX][Y+dY] == rows[radius][Y]) {
						histosRow[Y]++;
					}
				}
			}
		}
	} 
    double prob, entropy;
    int area, mid_area = diam*diam, side_area=(radius+1)*(diam), corner_area = (radius+1)*(radius+1);
	int on_edge = 0;
    uchar *outRow;
    for (X=0; X<input.rows; X++) {
    	outRow = output.ptr(X);
    	histosRow = histosMat.ptr(X);
    	for (Y=0; Y<input.rows; Y++) {
    		if (histosRow[Y] == 0) {
    			outRow[Y] = 0;
    		} else {
				on_edge = 0;
				if (X == 0 || X==(nRows-1)) {on_edge++;}
				if ( Y==0 || Y==(nCols-1)) {on_edge++;}
				if (on_edge==2) {area = corner_area;} 
				else if (on_edge==1) {area = side_area;} 
				else {area = mid_area;}
    			prob = (double) histosRow[Y] / area;
    			entropy = -log2(prob) * prob;
    			outRow[Y] = (uchar) (255.0 * entropy);
    		}
    	}
    }
}

void localWeightedEntropy(Mat input, Mat output, int radius, double *weights) {
	double denom, mid_denom=0, side_denom=0, corner_denom=0;
	const int diam = 2*radius+1;
	int nRows = input.rows, nCols = input.cols;
	int X, Y, dX, dY;
	for (X=0; X<diam; X++) {
		for (Y=0; Y<diam; Y++) {
			if (X <= radius) {
				side_denom += weights[X*diam+Y];
				if (Y <= radius) {
					corner_denom += weights[X*diam+Y];
				}
			}
			mid_denom += weights[X*diam+Y];
		}
    }
    Mat histosMat(nRows, nCols, CV_8U, Scalar(0));
	double accum, prob, entropy;
	int on_edge = 0;;
    uchar *histosRow;
    int left, right, top, bottom;
	uchar *rows[2*radius+1];
	for (X=0; X<nRows; X++) {
		bottom = X>radius ? -radius : -X;
		top = X+radius<nRows ? radius : nRows-X-1;
		histosRow = histosMat.ptr(X);
		for (dX=bottom; dX<=top; dX++) {
			rows[radius+dX] = input.ptr(X+dX);
		}
		for (Y=0; Y<nCols; Y++) {
			accum = 0;
			left = Y>radius ? -radius : -Y;
			right = Y+radius<nCols ? radius : nCols-Y-1;
			for (dX=bottom; dX<=top; dX++) {
				for (dY=left; dY<=right; dY++) {
					if (rows[radius+dX][Y+dY] == rows[radius][Y]) {
						accum += weights[(radius+dX)*diam+radius+dY];
					}
				}
			}
			if (accum == 0) {
    			histosRow[Y] = 0;
    		} else {
				on_edge = 0;
				if (X == 0 || X==(nRows-1)) {on_edge++;}
				if ( Y==0 || Y==(nCols-1)) {on_edge++;}
				if (on_edge==2) { denom = corner_denom;} 
				else if (on_edge==1) { denom = side_denom;} 
				else {denom = mid_denom;}
    			prob = accum / denom;
    			entropy = -log2(prob) * prob;
    			histosRow[Y] = (uchar) (255.0 * entropy);
    		}
		}
	} 
    uchar *outRow;
    for (X=0; X<nRows; X++) {
    	outRow = output.ptr(X);
    	histosRow = histosMat.ptr(X);
    	for (Y=0; Y<nCols; Y++) {
    		outRow[Y] = histosRow[Y];
    	}
    }
}

void localDistanceEntropy(Mat input, Mat output, int radius) {
	double mult = (double) radius * sqrt(2.0);
	const int diam = 2*radius+1;
	double weights[diam*diam], dist;
	int X, Y, dX, dY;
	for (X=0; X<diam; X++) {
		dX = X - radius;
		for (Y=0; Y<diam; Y++) {
			dY = Y - radius;
			dist = dY*dY+dX*dX;
			weights[X*diam+Y] = dist==0 ? mult : mult / sqrt(dist);
		}
    }
	localWeightedEntropy(input, output, radius, weights);
}

void localGaussianEntropy(Mat input, Mat output, int radius) {
	const int diam = 2*radius+1;
	//Multiplier reduces rounding errors later, but can easily overflow
	double mult = exp(radius);
	double gaussian[diam*diam];
	int X, Y, dX, dY;
	for (X=0; X<diam; X++) {
		dX = X - radius;
		for (Y=0; Y<diam; Y++) {
			dY = Y - radius;
			gaussian[X*diam+Y] = mult*exp(-(dX*dX+dY*dY)/2.0);
		}
    }
	localWeightedEntropy(input, output, radius, gaussian);
}

void normalEntropy(Mat input, Mat output) {
	double mean = 0;
	int nRows = input.rows;
	int nCols = input.cols;
	int size = nRows * nCols;
	if (input.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }
	int X, Y;
	uchar *row;
	for (X=0; X<nRows; X++) {
    	row = input.ptr(X);
    	for (Y=0; Y<nCols; Y++) {
			mean += row[Y];
    	}
    }
	mean = mean / size;
	double variance = 0;
	double diff;
	for (X=0; X<nRows; X++) {
    	row = input.ptr(X);
    	for (Y=0; Y<nCols; Y++) {
			diff = (row[Y]-mean);
			variance += diff*diff;
    	}
    }
	double varSquareInv = variance / (size>1 ? (size - 1): 1);
	variance = sqrt(varSquareInv);
	varSquareInv = -0.5 / varSquareInv;
	Mat lookUpMat(1, 256, CV_8U);
	uchar *lookUpRow = lookUpMat.ptr();
	for (X=0; X<256; X++) {
		diff = X-mean;
    	lookUpRow[X] = (uchar) 255*exp(diff*diff*varSquareInv);
    }
	LUT(input, lookUpMat, output);
}


int main(int argc, char** argv )
{
	if ( argc < 3 )
    {
        printf("Please input an image file name and an entropy type with parameter\n");
		printf("Optionallly enter a file name to output to\n");
        return -1;
    }
	char *outFile = NULL;
	if (argc == 4) {
		outFile = argv[3];
	}
    Mat img;
    img = imread(argv[1], IMREAD_GRAYSCALE );
    if ( !img.data )
    {
        printf("No image data \n");
        return -1;
    }
    Mat output;
	int paramLen = strlen(argv[2])-1;
	int radius;
	if (paramLen > 0) {
		switch (argv[2][0]) {
			case 'L':
				radius = strtol(argv[2]+1, NULL, 10);
				localEntropy(img, img, radius);
				break;
			case 'G':
				radius = strtol(argv[2]+1, NULL, 10);
				localGaussianEntropy(img, img, radius);
				break;
			case 'D':
				radius = strtol(argv[2]+1, NULL, 10);
				localDistanceEntropy(img, img, radius);
				break;
			default:
				printf("Please enter a radius or a different entropy\n");
				return -1;
		}
	} else {
		switch (argv[2][0]) {
			case 'H':
				globalEntropy(img, img);
				break;
			case 'N':
				normalEntropy(img, img);
				break;
			default:
				printf("Please enter a different entropy\n");
				return -1;
		}
	}
    namedWindow("Entropy", WINDOW_AUTOSIZE );
    imshow("Entropy", img);
    waitKey(0);
	if (outFile) {
		imwrite(outFile, img);
	}
    return 0;
}