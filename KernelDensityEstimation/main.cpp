#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include "FastKDEBackground.h"

int main(int argc, char **argv){
	if(argc < 2){
		std::cerr << "usage:" << argv[0] << "filelist" << std::endl;
		exit(-1);
	}

	const std::string filelist=argv[1];

	int h=20;
	float sigma2=50.0f;
	int twin=500;
	float threshold=0.001f;
	FastKDEBackground fpb(h,sigma2,twin,threshold);

	std::ifstream ifs(filelist.c_str());
	while(true){
		std::string filename;
		ifs >> filename;
		if(ifs.eof()) break;
		cv::Mat img=cv::imread(filename,0);
		cv::Mat mask=fpb(img);
		
		cv::imshow("frame",img);
		cv::imshow("mask",mask);
		cv::waitKey(1);
	}

	return 0;
}
