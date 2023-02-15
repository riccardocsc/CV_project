#include "hog.h"
#include <tuple>
//constructor
/**
 * -Param s1 _winSize
 * -Param s2 _blockSize
 * -Param s3 _blockStride
 * -Param s4 _cellSize
 * -Param modelPath string representing the SVM classifier path where to find it.
 *
 */

HogClassifier::HogClassifier(cv::Size s1, cv::Size s2, cv::Size s3, cv::Size s4, std::string modelPath){
  hog = cv::HOGDescriptor(s1, s2, s3, s4, 9, //nbins
                                              1, //derivAper,
                                              -1, //winSigma,
                                              cv::HOGDescriptor::L2Hys, //histogramNormType,
                                              0.2, //L2HysThresh,
                                              1,//gammal correction,
                                              64,//nlevels=64
                                              1);//Use signed gradients)
  svmModelpath = modelPath;
  cv::FileStorage svmReadStorage = cv::FileStorage(svmModelpath, cv::FileStorage::READ);
  if (!svmModelpath.empty() && svmReadStorage.isOpened()){ //We check if a SVM model is already trained with the name passed in svmModepath
      svm = cv::Algorithm::load<cv::ml::SVM>(svmModelpath);
  }
  else{

      svm = cv::ml::SVM::create();
      svm -> setType(cv::ml::SVM::C_SVC);
      svm -> setKernel(cv::ml::SVM::RBF);
      svm -> setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 10000, 1e-6));
  }
  svmReadStorage.release();
}

/**
 * Computes the HOG descriptor of the input image.
 *
 * -Param image the image of which we want to compute the HOG descriptor
 *
 * -Returns Mat containing the descriptor
 */

cv::Mat HogClassifier::HOG_Feature(cv::Mat img){

  cv::Mat mat;
  std::vector<float> descriptors;   //destination vector for hog feat
  cv::resize(img, img, cv::Size(48, 48), cv::INTER_LINEAR);  //resize all images to the fixed size of 64x64
  hog.compute(img, descriptors);    //compute HOG descriptor
  mat = cv::Mat(1, descriptors.size(), CV_32F, descriptors.data());
  /*
  for(int i = 1; i < rois.size(); i++){ //concatenate descriptors to form the examples for every roi in the image
    tmp = cv::Mat(img.clone(), rois[i]);
    cv::resize(tmp, tmp, cv::Size(64, 64), cv::INTER_LINEAR);
    hog.compute(tmp, descriptors);
    concat = cv::Mat(1, descriptors.size(), CV_32F, descriptors.data());
    mat.push_back(concat);
  */
  return mat;
}

/**
 * Trains the SVM given the set of samples and their labels
 *
 * -Param samples matrix containing the set of examples to train the model
 * -Param responses matrix containing the labels of the examples
 *
 */

void HogClassifier::train(cv::Mat samples, cv::Mat responses){
  if(svm->isTrained()){
      std::cout << "Svm model is loaded from file" << std::endl;
      return;
  }

  std::cout << "Svm started to train" << std::endl;
  int kfold = 3;
  bool trained = svm->trainAuto(samples, cv::ml::ROW_SAMPLE, responses, kfold); //Trains SVM with the computed descriptors
  if(trained){
      std::cout << "Svm successfully trained" << std::endl;
      if(!svmModelpath.empty()){ //Stores the SVM model to file
          svm->save(svmModelpath);
          std::cout << "Svm model saved to file" << std::endl;
      }
  }
  else{
      std::cerr << "Svm didn't train successfully" << std::endl;
  }

}
/**
 * Computes the labels given the set of positive and negative examples
 *
 * -Param samples_pos matrix of positive examples (HOG descriptors), one for each row
 * -Param samples_neg matrix of negative examples (HOG descriptors), one for each row
 *
 * -Returns a tuple containing the the labels
 */

std::tuple<cv::Mat, cv::Mat> HogClassifier::constructLabels(cv::Mat sample_pos, cv::Mat sample_neg){
  cv::Mat labels_pos = cv::Mat(sample_pos.rows, 1, CV_32F, 1); //column vector of 1s
  cv::Mat labels_neg = cv::Mat(sample_neg.rows, 1, CV_32F, -1); //column vector of -1s
  return {labels_pos, labels_neg};
}

/**
 * Predicts the class of the given image
 *
 * -Param img matrix to be classified
 *
 * -Returns an integer representing the prediction label
 */

std::tuple<int, float> HogClassifier::predict(cv::Mat img){
  if (!svm->isTrained()) {
      std::cout << "You cannot predict without a model!" << std::endl;
      return {-2, -2};
  }
  //compute the HOG descriptor
  cv::Mat mat;
  std::vector<float> descriptors;
  cv::resize(img, img, cv::Size(48, 48), cv::INTER_LINEAR);
  hog.compute(img, descriptors);    //compute HOG descriptor
  mat = cv::Mat(1, descriptors.size(), CV_32F, descriptors.data());
  //use the model to predict the label
  int prediction = svm -> predict(mat);
  float raw = svm -> predict(mat, cv::noArray(), cv::ml::StatModel::RAW_OUTPUT);
  return {prediction, raw};
}

/**
 * Loads images from a file
 *
 * -Param filename vector of strings representing the file from which we load images
 * -Param n number of images to load
 *
 * -Returns a vector of images loaded.
 */

std::vector<cv::Mat> HogClassifier::loadImage(std::vector<cv::String> filename, int n){

  std::vector<cv::Mat> v;
  int i =0;
  for (const auto& fn : filename){
  	if(i <= n){
  		cv::Mat img = cv::imread(fn);
  		v.push_back(img);
  	}
  	i++;
  }
  std::cout << i - 1 << " images loaded for the HOG classifier"<< '\n';
  return v;
}
