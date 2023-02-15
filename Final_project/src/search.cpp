#include "search.h"

using namespace cv::ximgproc::segmentation;

//constructor

Searcher::Searcher(){

}

std::vector<cv::Rect> Searcher::segmenter(cv::Mat v){
  if(v.empty()) std::cout << "Immagine non caricata" << '\n';
  cv::Ptr<SelectiveSearchSegmentation> ss = createSelectiveSearchSegmentation();
  ss -> setBaseImage(v);
  ss -> switchToSelectiveSearchFast();
  std::vector<cv::Rect> roi;
  ss -> process(roi);
  std::vector<cv::Rect> tmp;
  //the following removes all the proposals that are unlikely possible object locations
  int i = 0;
  while(i < roi.size() && i < 1000){
    if(roi[i].height * roi[i].width >= 0.1 * v.cols * v.rows){
		    tmp.push_back(roi[i]);
    }
    i++;
  }
	return tmp;
}

std::vector<cv::Rect> Searcher::edgeExtractor(cv::Mat img, int threshold){
  cv::RNG rng;
  cv::Mat canny_output;
  std::vector<int> areas;
  //use Canny edge detector and find the contours
  cv::Canny(img, canny_output, threshold, 2*threshold);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(canny_output, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
  std::vector<std::vector<cv::Point> > contours_poly(contours.size());
  std::vector<cv::Rect> boundRect(contours.size());

  for(int i = 0; i < contours.size(); i++){ //compute the bounding box starting from the edges found before and store its area
    cv::approxPolyDP(contours[i], contours_poly[i], 3, true);
    boundRect[i] = cv::boundingRect(contours_poly[i]);
    int a = boundRect[i].width*boundRect[i].height;
    areas.push_back(a);
  }
  //compute the maximum area
  int max = 0;
  for(int i = 0; i < contours.size(); i++) if (areas[i] > max) max = areas[i];

  //Now retain the bounding box if its area is greater than the maximum one times a static threshold
  std::vector<cv::Rect> bounding_boxes;
  for(int i = 0; i < contours.size(); i++){
    if(areas[i] >= max*0.2){
      bounding_boxes.push_back(boundRect[i]);
    }
  }
  return bounding_boxes;
}


void Searcher::printProposal(cv::Mat m, std::vector<cv::Rect> roi){
  for(int i = 0; i < roi.size(); i++){
    cv::rectangle(m, roi[i], cv::Scalar(0, 0, 255));
  }
  cv::imshow("Region proposal", m);
}
