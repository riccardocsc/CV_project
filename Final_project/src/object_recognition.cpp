#include "object_recognition.h"

ObjectRecognition::ObjectRecognition(std::string vocabularyFileName, std::string svmModelFileName){
  vocabularypath = vocabularyFileName;
  svmModelpath = svmModelFileName;
  detector = cv::SiftDescriptorExtractor::create();
  cv::FileStorage svmReadStorage = cv::FileStorage(svmModelpath, cv::FileStorage::READ);
  if (!svmModelpath.empty() && svmReadStorage.isOpened()) { //We check if a SVM model is already trained with the name passed in svmModelFileName
      svm = cv::Algorithm::load<cv::ml::SVM>(svmModelpath);
  } else {

      //A one class SVM is created
      svm = cv::ml::SVM::create();
      svm -> setType(cv::ml::SVM::ONE_CLASS);
      svm -> setNu(0.5);
      svm -> setKernel(cv::ml::SVM::RBF);
      svm -> setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 10000, 1e-6));
  }
  svmReadStorage.release();
  cv::FileStorage vocabularyReadStorage = cv::FileStorage(vocabularypath, cv::FileStorage::READ);
  if (!vocabularypath.empty() && vocabularyReadStorage.isOpened()) { //Checks wheter a vocabulary is already stored
      vocabularyReadStorage["vocabulary"] >> vocabulary;
      bowExtraction.setVocabulary(vocabulary);
  }
  vocabularyReadStorage.release();
}


void ObjectRecognition::createVocabulary(std::vector<cv::Mat> dataset, uint vocabularySize) {
    if (!vocabulary.empty()){
        std::cout << "Vocabulary already trained" << std::endl;
        return;
    }

    std::cout << "Started vocabulary training" << std::endl;
    std::vector<cv::KeyPoint> keyPoints;
    //std::vector<cv::KeyPoint> kps;
    cv::Mat untrainedDescriptors;

    //Extracts descriptors of images in the dataset
    cv::Ptr<cv::SiftDescriptorExtractor> extractor = cv::SiftDescriptorExtractor::create();
    for(cv::Mat image : dataset){
        extractor->detect(image, keyPoints);
        cv::Mat descriptors;
        extractor->compute(image, keyPoints, descriptors);

        untrainedDescriptors.push_back(descriptors);
    }

    //std::cout << untrainedDescriptors.size() << '\n';

    cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER, 1000, 0.001);
    std::cout << "Keypoints extracted" << '\n';
    //Trains the vocabulary using kmeans clustering
    cv::BOWKMeansTrainer trainer = cv::BOWKMeansTrainer(vocabularySize, criteria);
    trainer.add(untrainedDescriptors);
    vocabulary = trainer.cluster();
    bowExtraction.setVocabulary(vocabulary);
    std::cout << "Vocabulary train finished" << std::endl;

    //Stores the vocabulary on a file
    if (!vocabularypath.empty()) {
        cv::FileStorage writeStorage = cv::FileStorage(vocabularypath, cv::FileStorage::WRITE);

        writeStorage << "vocabulary" << vocabulary;
        writeStorage.release();
        std::cout << "Vocabulary saved to file" << std::endl;
    }
}


void ObjectRecognition::train(std::vector<cv::Mat> boats) {
    if (svm->isTrained()) {
        std::cout << "Svm model is loaded from file" << std::endl;
        return;
    }

    if (vocabulary.rows == 0) {
        std::cerr << "The vocabulary must be created first!" << std::endl;
        return;
    }

    std::cout << "Svm started to train" << std::endl;

    cv::Mat descriptors_to_train;
    std::vector<int> labels;
    //We extract the descriptors for each image in the dataset using BoW extraction.
    for (int i = 0; i < boats.size(); i++) {
        cv::Mat image = boats[i];

        std::cout << "Process boat image " << i << std::endl;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detector->detect(image, keypoints); //We compute keypoints that are used for training descriptors
        bowExtraction.compute(image, keypoints, descriptors); //Extract the descriptors of an image using BoW
        if(!descriptors.empty()) descriptors_to_train.push_back(descriptors);

        if (!descriptors.empty())
            labels.push_back(1); //Ones label which tells that this image is a positive image that contains the object we want recognize
    }


    descriptors_to_train.convertTo(descriptors_to_train, CV_32FC1);
    int kfold = 5;
    bool trained = svm->trainAuto(descriptors_to_train, cv::ml::ROW_SAMPLE, labels, kfold); //Trains SVM with the computed descriptors
    if (trained) {
        std::cout << "Svm successfully trained" << std::endl;
        if (!svmModelpath.empty()) { //Stores the SVM model to file
            svm->save(svmModelpath);
            std::cout << "Svm model saved to file" << std::endl;
        }
    } else {
        std::cerr << "Svm didn't train successfully" << std::endl;
    }

}


int ObjectRecognition::predict(cv::Mat image) {
    if (vocabulary.empty() || !svm->isTrained()) {
        std::cerr << "You can't predict an image without a trained vocabulary and a prediction model" << std::endl;
        return -1;
    }
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;

    detector->detect(image, keypoints);

    bowExtraction.compute(image, keypoints, descriptors);
    int res;
    if(!descriptors.empty()) res = svm->predict(descriptors);
    return res;
}

std::vector<cv::Mat> ObjectRecognition::loadImage(std::vector<cv::String> filename, int n){

  std::vector<cv::Mat> v;
  int i =0;
  for (const auto& fn : filename){
  	if(i <= n){
  		cv::Mat img = cv::imread(fn);
  		v.push_back(img);
  	}
  	i++;
  }
  std::cout << i - 1 << " images loaded for the vocabulary"<< '\n';
  return v;
}
