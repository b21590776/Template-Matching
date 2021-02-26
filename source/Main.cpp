#include <ctime>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "TemplateMatcher.h"

int main(int argc, char** argv) {

    clock_t start;
    double duration;
    start = clock();

    cv::Mat img, templateImage;

    // check arguments
    if (argc != 3)
    {
        ("Program needs 2 arguments: Image and template image");
        return -1;
    }
    
    img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);            

    templateImage = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    if (!img.data) {
        std::cout << "image load failed" << std::endl;
        return -1;
    }

    if (!templateImage.data) {
        std::cout << "template image load failed" << std::endl;
        return -1;
    } 
     
    TemplateMatcher templateMatcher;
    cv::Mat result = templateMatcher.matchTemplate(img, templateImage);

    duration = (clock() - start) / (double)CLOCKS_PER_SEC;
    std::cout << "passed time: " << duration << " s"<< std::endl;


    cv::imwrite("result.png", result);
    cv::namedWindow("result");
    cv::imshow("result", result);
    cv::waitKey(0);

    return 0;
}