#define _USE_MATH_DEFINES
#include <cmath>
#include "FastKDEBackground.h"


float FastKDEBackground::gaussian(float x,float sigma2){
    return 1.0/(std::sqrt(2*M_PI)*sigma2)*std::exp(-x*x/(2*sigma2));
}

FastKDEBackground::FastKDEBackground(int _h, float _sigma2, int _time_window, float _threshold)
:h(_h),sigma2(_sigma2),twin(_time_window),count(0),threshold(_threshold),isInit(false){
	parwin.resize(h*2+1,0.0f);
	for(int i=0;i<h*2+1;i++) parwin[i]=gaussian(i-h,sigma2)/twin;
}

void FastKDEBackground::init(const cv::Mat& img){
	sizes[0]=img.rows;
	sizes[1]=img.cols;
	sizes[2]=256+h*2;
	probabilities=cv::Mat(sizeof(sizes)/sizeof(int),sizes,CV_32FC1);
	imagebuffer.resize(0);
	count=0;
	isInit=true;
}

cv::Mat FastKDEBackground::operator()(const cv::Mat& img){
	if(!isInit) init(img);
	cv::Mat probimg;
	cv::Mat mask;
	imagebuffer.push_back(img);

	probimg=get_probabilities(img);
	cv::imshow("Probabilities",probimg*100);
	cv::threshold(probimg,mask, threshold, 255, CV_THRESH_BINARY_INV);
	mask.convertTo(mask,CV_8UC1);

	if(count<=twin){count++;}
	else{imagebuffer.pop_front();}

	return mask;
}

cv::Mat FastKDEBackground::get_probabilities(const cv::Mat& image){
	cv::Mat probimg=cv::Mat::zeros(image.size[0],image.size[1], CV_32FC1);
	int l=parwin.size();
	int h=(l-1)/2;

	if(count > twin){
		for(int i=0;i<image.rows;i++){
			const unsigned char *imgl=image.ptr<unsigned char>(i);
			const unsigned char *prevl=imagebuffer[0].ptr<unsigned char>(i);
			float *probl=probabilities.ptr<float>(i);
			float *probimgl=probimg.ptr<float>(i);
			for(int j=0;j<image.cols;j++){
				float *probll=probl+j*l;
				for(int k=0;k<l;k++){
					probll[prevl[j]+k]-=parwin[k];
					probll[imgl[j]+k]+=parwin[k];
					probimgl[j]=probll[imgl[j]+h];
				}
			}
		}
	}else{
		for(int i=0;i<image.rows;i++){
			const unsigned char *imgl=image.ptr<unsigned char>(i);
			float *probl=probabilities.ptr<float>(i);
			float *probimgl=probimg.ptr<float>(i);
			for(int j=0;j<image.cols;j++){
				float *probll=probl+j*l;
				for(int k=0;k<l;k++){
					probll[imgl[j]+k]+=parwin[k];
					probimgl[j]=probll[imgl[j]+h];
				}
			}
		}
	}
	return probimg;
}

