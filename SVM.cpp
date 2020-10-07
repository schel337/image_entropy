#include <stdio.h>
#include <iostream>
#include <fstream>
#include "Entropy.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace cv::ml;

const int CLASSES = 10;
const int N_TRAIN = 900;
const int N_TEST = 100;
int IMG_SIZE = 28*28;

const int RADIUS = 2;

//Currently set to local distance entropies, but can be swapped out easily. 
//Note that the size of train_datas must be changed to remove the entropy values
void read_datas(Mat train_datas[CLASSES], int start, int end) {
    int N = end - start;
    std::ifstream data_file;
	char path[] = "./MNIST/data0";
    Mat buffer = Mat(N, IMG_SIZE, CV_8U);
    Mat ent = Mat(N, IMG_SIZE, CV_8U);
    for (int i=0; i<CLASSES; i++, path[12]++) {
        data_file.open(path, std::ios::in | std::ios::binary);
		if (!data_file.is_open()) {
			printf("Could not open file: %s\n", path);
		} else {
            data_file.ignore(start*IMG_SIZE);
            printf("Opened file: %s\n", path);
            for (int j=0; j<N; j++) {
                if (!data_file) {
                    printf("error reading file: %s\n", path);
                    return;
                }
			    data_file.read((char*) buffer.ptr(j), IMG_SIZE);
                localDistanceEntropy(
                    Mat(buffer, Range(j,j+1), Range::all()),
                    Mat(ent, Range(j,j+1), Range::all()),
                    RADIUS);
            }
		}
		data_file.close();
        train_datas[i] = Mat(N, 2*IMG_SIZE, CV_8U);
        hconcat(buffer, ent, train_datas[i]);
        printf("Rows: %d \n Columns: %d \n", train_datas[i].rows, train_datas[i].cols);
        train_datas[i].convertTo(train_datas[i], CV_32F);
    }
}

int main(int argc, char** argv)
{
    Ptr<SVM> svms[CLASSES][CLASSES];
    if (argc == 1 || (argc>2 && argv[1][0]=='T')) {
        Mat train_datas[CLASSES];
        read_datas(train_datas, 0, N_TRAIN);
        Mat labels(2*N_TRAIN, 1, CV_32SC1);
        for (int i=0; i<N_TRAIN; i++) {
            ((int*) labels.ptr(i))[0]=-1;
            ((int*) labels.ptr(i+N_TRAIN))[0]=1;
        }
        Mat votes = Mat::zeros(CLASSES*N_TRAIN, CLASSES, CV_32SC1);
        float resp;
        // Train the SVM(s)
        #pragma omp parallel for collapse(2)
        for (int i=0; i<CLASSES; i++) {
            for (int j=0; j<CLASSES; j++) {
                if (j<i) {
                Mat train_data;
                svms[i][j] = SVM::create();
                svms[i][j]->setType(SVM::C_SVC);
                svms[i][j]->setKernel(SVM::LINEAR);
                svms[i][j]->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
                vconcat(train_datas[i], train_datas[j], train_data);
                svms[i][j]->trainAuto(train_data, ROW_SAMPLE, labels, 10);
                for (int k=0; k<N_TRAIN; k++) {
                    resp = svms[i][j]->predict(Mat(train_datas[i], Range(k,k+1), Range::all()));
                    if (resp == -1) {((int* )votes.ptr(i*N_TRAIN+k))[i]++;}
                    else {((int* )votes.ptr(i*N_TRAIN+k))[j]++;}
                    resp = svms[i][j]->predict(Mat(train_datas[j], Range(k,k+1), Range::all()));
                    if (resp == -1) {((int* )votes.ptr(j*N_TRAIN+k))[i]++;}
                    else {((int* )votes.ptr(j*N_TRAIN+k))[j]++;}
                }
                }
                
            }
        }
        int count = 0;
        int count2 = 0;
        int win;
        for (int i=0; i<N_TRAIN*CLASSES; i++) {
            win = 0;
            int *vote = (int*) votes.ptr(i);
            for (int j=0; j<CLASSES; j++) {
                if (vote[j] > vote[win]) {
                    win = j;
                }
            }
            count2 += vote[i/N_TRAIN];
            if (win == i/N_TRAIN) {
                count++;
            }
        }
        printf("Training Accuracy Predicted:%f\n", (double) count / (N_TRAIN*CLASSES));
        printf("Training Accuracy of SVMs:%f\n", (double) count2 / (N_TRAIN*CLASSES*CLASSES));
    }
    else if (argc>=2 && argv[1][0]=='L') {
        char svm_path[] = "./SVM_H_saves/svm00.xml";
        for (int i=0; i<CLASSES; i++) {
            svm_path[15] = (char)('0'+i);
            for (int j=0; j<i; j++) {
                svm_path[16] = (char)('0'+j);
                svms[i][j] = Algorithm::load<SVM>(svm_path);
            }
        }
    }
    else {
        printf("Please enter T to train a new SVM or L to load a saved SVM\n");
        return -1;
    }
    Mat test_datas[CLASSES];
    read_datas(test_datas, N_TRAIN, N_TRAIN+N_TEST);
    Mat votes = Mat::zeros(CLASSES*N_TEST, CLASSES, CV_32SC1);
    float resp;
    for (int i=0; i<CLASSES; i++) {
        for (int j=0; j<i; j++) {
            for (int k=0; k<N_TEST; k++) {
                resp = svms[i][j]->predict(Mat(test_datas[i], Range(k,k+1), Range::all()));
                if (resp == -1) {((int* )votes.ptr(i*N_TEST+k))[i]++;}
                else {((int* )votes.ptr(i*N_TEST+k))[j]++;}
                resp = svms[i][j]->predict(Mat(test_datas[j], Range(k,k+1), Range::all()));
                if (resp == -1) {((int* )votes.ptr(j*N_TEST+k))[i]++;}
                else {((int* )votes.ptr(j*N_TEST+k))[j]++;}
            }
        }
    }
    int count = 0;
    int count2 = 0;
    int win;
    for (int i=0; i<N_TEST*CLASSES; i++) {
        win = 0;
        int *vote = (int*) votes.ptr(i);
        for (int j=0; j<CLASSES; j++) {
            if (vote[j] > vote[win]) {
                win = j;
            }
        }
        count2 += vote[i/N_TEST];
        if (win == i/N_TEST) {
            count++;
        }
    }
    printf("Test Accuracy of Votes:%f\n", (double) count2 / (N_TEST*CLASSES*CLASSES));
    printf("Test Accuracy of Majority prediction:%f\n", (double) count / (N_TEST*CLASSES));
    return 0;
}
