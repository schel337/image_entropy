#include <stdio.h>
#include <math.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include "Entropy.h"
using namespace cv;

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