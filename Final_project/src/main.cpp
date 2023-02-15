#include <iostream>
#include <fstream>
#include <opencv2/dnn.hpp>
#include "object_recognition.h"
#include "cascade.h"
#include "search.h"
#include "hog.h"

std::vector<cv::Rect> constructGroundTruth(cv::String filename);
std::vector<cv::Mat> loadImages(std::vector<cv::String> filename, std::vector<cv::String> gdtruth_filename);
float computeIoU(cv::Rect bBox1, cv::Rect bBox2);
cv::Mat equalize(cv::Mat img);
float computeIoUMax(cv::Rect pred, std::vector<cv::Rect> gt_boxes);

int main(){
  bool createPos = false;
  bool trainSVM = false;
  bool trainHOG = false;
  bool test = true;

  //load the image to test and the relative ground truth
  std::string string = "../TEST_DATASET/kaggle/02.jpg";
  cv::String gtruth_txt = "../TEST_DATASET/kaggle_labels_txt/02.txt";
  cv::Mat test_img = cv::imread(string);
  cv::Mat filtered, out;
  //apply the bilateral filter and the CLAHE equalization
  cv::bilateralFilter(test_img, filtered, 9, 100, 100);
  out = equalize(filtered);

  if(createPos){//code used to extract the positive examples from the ground truth
    std::vector<cv::String> img_filenames, gdtruth_filenames;

    cv::glob("../TRAINING_DATASET/IMAGES/*.png", img_filenames, false);
    cv::glob("../TRAINING_DATASET/LABELS_TXT/*.txt", gdtruth_filenames, false);
    std::cout << "Loading images..." << '\n';
    std::vector<cv::Mat> imgs = loadImages(img_filenames, gdtruth_filenames);
    std::cout << imgs.size() << " images loaded"<< '\n';

    //construct the objects ground truths
    std::vector<std::vector<cv::Rect>> obj_rects (imgs.size());
    std::cout << "Constructing the ground truths..." << '\n';
    for(int i = 0; i < imgs.size(); i++){
      std::vector<cv::Rect> tmp = constructGroundTruth(gdtruth_filenames[i]);
      for(int k = 0; k < tmp.size(); k++){
        cv::Mat cut;
        cut = cv::Mat(imgs[i].clone(), tmp[k]);
        //save the image ground truths
        cv::imwrite("../dataset/boat/" + std::to_string(i) + "_" + std::to_string(k) +".jpg", cut);
      }
      obj_rects[i] = tmp;
    }
  }
  if(trainSVM){ //code used to train the BoW framework
    std::vector<cv::String> bow_filenames;
    cv::glob("../Train_BoW/*", bow_filenames, false);

    std::string modelPath = "../model.xml";
    std::string vocabularyPath = "../vocabulary.yml";

    ObjectRecognition *objrec = new ObjectRecognition(vocabularyPath, modelPath);
    std::cout << "Loading images..." << '\n';
    std::vector<cv::Mat> img_bow = objrec -> loadImage(bow_filenames, bow_filenames.size());
    objrec -> createVocabulary(img_bow, 1000);
    std::vector<cv::String> boat_filenames;

    cv::glob("../dataset/boat/*.jpg", boat_filenames, false);
    std::cout << "Loading examples" << '\n';
    std::vector<cv::Mat> boats = objrec -> loadImage(boat_filenames, boat_filenames.size());
    objrec -> train(boats);
  }

  if(trainHOG){ //code used to train the HOG-based classifier
    std::vector<cv::String> boat_filenames, no_boat_filenames;
    cv::glob("../dataset/boat/*", boat_filenames, false);
    cv::glob("../dataset/no_boat/*", no_boat_filenames, false);
    std::string modelPath = "../SVM_HOG_final.xml";

    // Compute the Hog descriptor
    cv::Size win = cv::Size(48, 48);
    cv::Size block_size = cv::Size(16, 16);
    cv::Size cell = cv::Size(8, 8);
    cv::Size block_stride = cv::Size(8, 8);

    HogClassifier *hog = new HogClassifier(win, block_size, block_stride, cell, modelPath);
    std::cout << "Loading examples..." << '\n';
    std::vector<cv::Mat> boats = hog -> loadImage(boat_filenames, boat_filenames.size());
    std::vector<cv::Mat> no_boats = hog -> loadImage(no_boat_filenames, no_boat_filenames.size());

    std::cout << "Extracting HOG descriptors..." << '\n';
    cv::Mat pos_examples, neg_examples;
    if(!boats[0].empty()) pos_examples = hog -> HOG_Feature(boats[0]);
    if(!no_boats[0].empty()) neg_examples = hog -> HOG_Feature(no_boats[0]);
    cv::Mat descriptor;
    for(int i = 1; i < boats.size(); i++){
      if(!boats[i].empty()) descriptor = hog -> HOG_Feature(boats[i]);
      pos_examples.push_back(descriptor);
    }
    for(int i = 1; i < no_boats.size(); i++){
      if(!no_boats[i].empty()) descriptor = hog -> HOG_Feature(no_boats[i]);
      neg_examples.push_back(descriptor);
    }

    std::cout << "Constructing labels..." << '\n';
    auto [pos_lab, neg_lab] = hog -> constructLabels(pos_examples, neg_examples);

    cv::Mat samples, responses;
    cv::vconcat(pos_examples, neg_examples, samples);
    cv::vconcat(pos_lab, neg_lab, responses);
    responses.convertTo(responses, CV_32SC1);

    std::cout << "Training the SVM..." << '\n';
    hog -> train(samples, responses);
  }

  if(test){ //test the model

    std::cout << "Testing the boat detector" << '\n';
    const std::string modelName = "../cascade.xml";
    std::cout << "Cascade..." << '\n';
    CascadeClassifier *classifier = new CascadeClassifier(modelName);
    std::vector<cv::Rect> boxes = classifier -> predict(filtered);

    Searcher *src = new Searcher();
    std::cout << "Selective Search..." << '\n';
    std::vector<cv::Rect> v = src -> segmenter(filtered);
    std::cout << "Canny edge extractor..." << '\n';
    std::vector<cv::Rect> edges_prop = src -> edgeExtractor(out, 100);
    boxes.insert(boxes.end(), v.begin(), v.end());
    std::string modelPath = "../model_1000/model.xml";
    std::string vocabularyPath = "../model_1000/vocabulary.yml";
    ObjectRecognition *objrec = new ObjectRecognition(vocabularyPath, modelPath);

    std::string HOGmodelPath = "../SVM_HOG_final.xml";

    // Compute the Hog descriptor
    cv::Size win = cv::Size(48, 48);
    cv::Size block_size = cv::Size(16, 16);
    cv::Size cell = cv::Size(8, 8);
    cv::Size block_stride = cv::Size(8, 8);

    HogClassifier *hog = new HogClassifier(win, block_size, block_stride, cell, HOGmodelPath);

    std::cout << "Predicting locations..." << '\n';
    std::vector<cv::Rect> final;
    std::vector<float> confidence;
    std::vector<float> raw_pred;

    for(int i = 0; i < boxes.size(); i++){//predict the boxes coming from the selective search
      int pred = objrec -> predict(cv::Mat(filtered.clone(), boxes[i]));
      if(pred == 1){
        auto [prediction, raw] = hog -> predict(cv::Mat(filtered.clone(), boxes[i]));
        if(prediction == 1) {
          final.push_back(boxes[i]);
          raw_pred.push_back(raw);
        }
      }
    }

    for(int i = 0; i < edges_prop.size(); i++){ //predict the boxes of the Canny edge detector
      auto [prediction, raw] = hog -> predict(cv::Mat(filtered.clone(), edges_prop[i]));
      if(prediction == 1){
        final.push_back(edges_prop[i]);
        raw_pred.push_back(raw);
      }
    }

    float max = 0;
    for(int i = 0; i < raw_pred.size(); i++){
      if(abs(raw_pred[i]) > max) max = abs(raw_pred[i]);
    }
    for(int i = 0; i < raw_pred.size(); i++){
      raw_pred[i] = abs(raw_pred[i])/max;
    }

    std::cout << "NMS..." << '\n';
    std::vector<int> final_after_NMS_index;
    std::vector<cv::Rect> final_after_NMS;
    cv::dnn::NMSBoxes(final, raw_pred, 0.6, 0.05, final_after_NMS_index);

    for(int k = 0; k < final_after_NMS_index.size(); k++){
      final_after_NMS.push_back(final[final_after_NMS_index[k]]);
    }

    std::vector<cv::Rect> bound_ground = constructGroundTruth(gtruth_txt);
    float iou;
    for(int i = 0; i < final_after_NMS.size(); i++){
      iou += computeIoUMax(final_after_NMS[i], bound_ground);
      cv::rectangle(test_img, final_after_NMS[i], cv::Scalar(0,255,0), 2);
    }

    std::cout << "Average IoU in the image: " << iou/bound_ground.size() << '\n';
    cv::imshow("test", test_img);
    cv::waitKey(0);
    //cv::imwrite("../Kaggle_8.png", test_img);
  }


  return 0;
}

std::vector<cv::Rect> constructGroundTruth(cv::String filename){
  /*
    function used to construct the ground truth reading the coordinates from txt file
  */
  std::vector<cv::Rect> roi;
  std::ifstream myfile(filename);
  std::string line;
  if (myfile.is_open()){
		while (getline(myfile, line)){
			size_t pos = 0;
			std::vector<float> bounding_box; //xmin, xmax, ymin, ymax
			line.erase(0, line.find(":") + 1);
			while ((pos = line.find(";")) != std::string::npos) {
				bounding_box.push_back(stod(line.substr(0, pos)));
				line.erase(0, pos + 1);
			}
      cv::Rect tmp = cv::Rect(bounding_box[0], bounding_box[2],bounding_box[1] -  bounding_box[0], bounding_box[3] - bounding_box[2]);
      roi.push_back(tmp);
		}
		myfile.close();
	}
	else std::cout << "Unable to open file" << std::endl;
  return roi;
}


std::vector<cv::Mat> loadImages(std::vector<cv::String> filename, std::vector<cv::String> gdtruth_filename){
	// Method that load the images from a string representing the file's path. It loads the image only if it can find also
	// the corresponding ground truth.
  std::vector<cv::Mat> v;
	cv::Mat img;
  size_t index1, index2;
  for(int i = 0; i < filename.size(); i++){
	   index1 = filename[i].find_last_of("/");
	   index2 = filename[i].find_last_of(".");
	   std::string rawname = filename[i].substr(index1, index2 - index1);
	   std::string name = "../TRAINING_DATASET/LABELS_TXT" + rawname + ".txt";
	   if (std::find(gdtruth_filename.begin(), gdtruth_filename.end(), name) != gdtruth_filename.end()){
       img = cv::imread(filename[i]);
       v.push_back(img);
     }
   }
	return v;
}

/**
  *  Equalizes the image adopting CLAHE
  *
  * -Param img input image
  * -Return matrix that is the equalized version of the input
*/


cv::Mat equalize(cv::Mat img){
  // Apply CLAHE on Lab Color Space
  cv::Mat out = img.clone();
  cv::Mat imgL;
	cv::cvtColor(out, imgL, cv::COLOR_BGR2Lab);
	std::vector<cv::Mat> labPlanes;
	cv::split(imgL, labPlanes);

  //take the l channel
	cv::Mat l_channel = labPlanes[0];
  cv::Mat a_channel = labPlanes[1];
  cv::Mat b_channel = labPlanes[2];


  //create the object clahe and apply the equalization
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0);
	clahe->apply(l_channel, l_channel);

	labPlanes[0] = l_channel;
  labPlanes[1] = a_channel;
  labPlanes[2] = b_channel;

	cv::merge(labPlanes, out);
	cv::cvtColor(out, out, cv::COLOR_Lab2BGR);
  return out;
}
/**
  *  Computes the IoU of two BB
  *
  * -Param bBox1 rectangle
  * -Param bBox2 rectangle
  * -Return float that is the IoU of the two bounding boxes
*/

float computeIoU(cv::Rect bBox1, cv::Rect bBox2){

 float interArea = (bBox1 & bBox2).area(); //compute the intersection

 float unionArea = (bBox1 | bBox2).area(); //compute the union

 //compute the intersection over union by taking the intersection area and dividing it by the union
 float iou = interArea / unionArea;

 //return the intersection over union value
 return iou;
}

/**
  *  Computes the IoU of the relative ground truth
  *
  * -Param pred rectangle representing the predicted bounding box
  * -Param gt_boxes ground truth boundind boxes for the image
  * -Return float that is the IoU
*/
float computeIoUMax(cv::Rect pred, std::vector<cv::Rect> gt_boxes){
  float max = 0;
  for(int i = 0; i < gt_boxes.size(); i++){
    cv::Rect box = gt_boxes[i];
    float iou = computeIoU(box, pred);
    if(iou > max) max = iou;
  }
  return max;
}
