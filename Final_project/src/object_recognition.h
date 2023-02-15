#include <opencv2/opencv.hpp>
#include <tuple>

class ObjectRecognition {
private:
    std::string vocabularypath;
    std::string svmModelpath;
    cv::Mat vocabulary;
    cv::Ptr<cv::ml::SVM> svm;
    cv::Ptr<cv::SiftDescriptorExtractor> detector;

    //Bag of words extractor which is initialized with SiftDescriptorExtractor and FlanBasedMatcher
    cv::BOWImgDescriptorExtractor bowExtraction = cv::BOWImgDescriptorExtractor(cv::SiftDescriptorExtractor::create(),
                                                                                cv::FlannBasedMatcher::create());

public:
    /**
     * -Param vocabularyFileName is the file name of the vocabulary if we want to store it locally
     * -Param svmModelFileName is the file name of the SVM model if we want to store it locally
     */
    ObjectRecognition(std::string vocabularyFileName, std::string svmModelFileName);

    /**
     * Creates the BoW vocabulary using the passed dataset
     * -Param dataset is the dataset for training the vocabulary
     * -Param vocabularySize defines the size of the BoW vocabulary
     */
    void createVocabulary(std::vector<cv::Mat> dataset, uint vocabularySize);
    
    /**
     * Trains an SVM model using the passed dataset
     */
    void train(std::vector<cv::Mat> boats);

    /**
     * Predicts wheter the passed image contains the desired object.
     *
     * -Param image the image where we want detect the desired object
     * -Param keypoints is used when keypoints of an image are already available, so we can pass them here to avoid a recomputation. If this parameter is not passed, the keypoints will be computed inside this function
     *
     * -Returns 1 if the image contains the expected object, 0 otherwise. -1 if the prediction is impossible
     */
    int predict(cv::Mat image);


    std::vector<cv::Mat> loadImage(std::vector<cv::String> filename, int n);
};
