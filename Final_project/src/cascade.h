#include <opencv2/objdetect.hpp>
#include <iostream>

class CascadeClassifier{
protected:
  cv::CascadeClassifier classifier;
public:
  /**
  *  Constructor
  *
  * -Param filename string representing the path from which the cascade is loaded
  */
  CascadeClassifier(const std::string filename);

  /**
  *  predicts the locations of boats in the image
  *
  * -Param img input image
  * -Return a vector of bounding boxes representing boat locations
  */
  std::vector<cv::Rect> predict(cv::Mat img);
};
