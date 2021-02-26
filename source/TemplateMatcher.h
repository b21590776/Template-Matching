#pragma once

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

class TemplateMatcher {

public:

	// Constructor
	TemplateMatcher();

	// Deconstructor
	~TemplateMatcher();
	  
	// Template matching algorithm
	cv::Mat matchTemplate(cv::Mat& image, cv::Mat& templateImg);
	 
private:

	class Impl;  // Hidden data & implementation
	std::unique_ptr<Impl> m_impl;// impl class instance 

};
 