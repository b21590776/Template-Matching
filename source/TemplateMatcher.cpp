#include "TemplateMatcher.h"
 
class TemplateMatcher::Impl
{
public:
    int thresh = 35; // detection threshold score.
    int octaves = 3; // detection octaves. Use 0 to do single scale.
    float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
     
};


TemplateMatcher::TemplateMatcher():m_impl(new TemplateMatcher::Impl)
{
};

TemplateMatcher::~TemplateMatcher()
{
};

cv::Mat TemplateMatcher::matchTemplate(cv::Mat& image, cv::Mat& templateImg)
{
    // Detect the keypoints and extract descriptors using BRISK
    auto detector = cv::BRISK::create(m_impl->thresh, m_impl->octaves, m_impl->patternScale);  // ORB::create();
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> templateKeypoints;
    cv::Mat descriptors;
    cv::Mat templateDescriptors;
    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    detector->detectAndCompute(templateImg, cv::noArray(), templateKeypoints, templateDescriptors);

    // Matching descriptor vectors using BF matcher
    std::vector<cv::DMatch> matches;
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);  // FlannBasedMatcher::create();
    matcher->match(templateDescriptors, descriptors, matches);
    std::vector<cv::Point2f> templatePoints;
    std::vector<cv::Point2f> imagePoints;
    for (size_t i = 0; i < matches.size(); i++)
    {
        templatePoints.push_back(templateKeypoints[matches[i].queryIdx].pt);
        imagePoints.push_back(keypoints[matches[i].trainIdx].pt);
    }

    // Compute homography
    cv::Mat Homography = cv::findHomography(templatePoints, imagePoints, cv::RANSAC);

    // Localize the template
    std::vector<cv::Point2f> templateCorners(4);
    templateCorners[0] = cv::Point2f(0.f, 0.f);
    templateCorners[1] = cv::Point2f((float)templateImg.cols, 0.f);
    templateCorners[2] = cv::Point2f((float)templateImg.cols, (float)templateImg.rows);
    templateCorners[3] = cv::Point2f(0, (float)templateImg.rows);
    std::vector<cv::Point2f> detectedCorners(4);
    perspectiveTransform(templateCorners, detectedCorners, Homography);

    // draw the matches
    cv::Mat matchedImage;
    cv::drawMatches(templateImg, templateKeypoints, image, keypoints, matches, matchedImage, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // draw lines between the transformed corners
    for (int i = 0; i < 4; i++)
    {
        line(matchedImage, detectedCorners[i] + cv::Point2f((float)templateImg.cols, 0),
            detectedCorners[(i + 1) % 4] + cv::Point2f((float)templateImg.cols, 0), cv::Scalar(0, 255, 0), 4);
    }

    // Print transformed corners coordinates
    for (int i = 0; i < 4; i++) {
        std::cout << "Template image corner: " << templateCorners[i] << " corresponding to image point:" << detectedCorners[i] << std::endl;
    }

    return matchedImage;

}