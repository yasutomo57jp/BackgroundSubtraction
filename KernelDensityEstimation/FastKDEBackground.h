#ifndef __FAST_PARZEN_BACKGROUND_H__
#define __FAST_PARZEN_BACKGROUND_H__
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>

class FastKDEBackground{
	public:
		FastKDEBackground(int h, float sigma2, int time_window,float threshold);
		cv::Mat operator()(const cv::Mat& img);

	private:
		void init(const cv::Mat& img);
		cv::Mat get_probabilities(const cv::Mat& img);
		float gaussian(float x,float sigma2);
		int h;
		float sigma2;
		int twin;
		int count;
		float threshold;
		bool isInit;
		std::vector<float> parwin;
		int sizes[3];
		cv::Mat probabilities;
		std::deque<cv::Mat> imagebuffer;
};
#endif
