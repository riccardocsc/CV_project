#include "cascade.h"


//constructor

CascadeClassifier::CascadeClassifier(const std::string filename){
  if (!classifier.load(filename)){
    std::cerr << "ERROR: Could not load classifier cascade!" << std::endl;
  }
}

std::vector<cv::Rect> CascadeClassifier::predict(cv::Mat img){
  if(img.depth() != CV_8U) std::cerr << "ERROR: wrong input image type. It must be CV_8U!" << '\n';
  std::vector<double> weights;
  std::vector<int> levels;
  std::vector<cv::Rect> detections;
  int W = img.cols;
  int H = img.rows;
  cv::Size size = cv::Size(H/10,W/10);
  classifier.detectMultiScale(img, detections, levels, weights, 1.1, 3, 0, size, cv::Size(), true);
  return detections;
}
