#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

class HogClassifier{

public:
  HogClassifier(cv::Size s1, cv::Size s2, cv::Size s3, cv::Size s4, std::string modelPath);
  cv::Mat HOG_Feature(cv::Mat img);
  std::tuple<cv::Mat, cv::Mat> constructLabels(cv::Mat sample_pos, cv::Mat sample_neg);
  std::vector<cv::Mat> loadImage(std::vector<cv::String> filename, int n);
  void train(cv::Mat samples, cv::Mat responses);
  std::tuple<int, float> predict(cv::Mat img);

protected:
  cv::HOGDescriptor hog;
  std::string svmModelpath;
  cv::Ptr<cv::ml::SVM> svm;
};
