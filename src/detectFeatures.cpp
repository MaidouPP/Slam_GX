#include <iostream>
#include "slamBase.h"
using namespace std;

// opencv feature detection model
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>

int main() {
  cv::Mat rgb1 = cv::imread("./data/rgb1.png");
  cv::Mat rgb2 = cv::imread("./data/rgb2.png");
  cv::Mat depth1 = cv::imread("./data/depth1.png", -1);
  cv::Mat depth2 = cv::imread("./data/depth2.png", -1);

  // declare feature detector and descriptor
  cv::Ptr<cv::FeatureDetector> _detector;
  cv::Ptr<cv::DescriptorExtractor> _descriptor;

  // default detector is SIFT
  // before SIFT or SURF, need to initialize nonfree module
  cv::initModule_nonfree();
  _detector = cv::FeatureDetector::create("GridSIFT");
  _descriptor = cv::DescriptorExtractor::create("SIFT");

  vector< cv::KeyPoint > kp1, kp2;
  _detector->detect( rgb1, kp1 );
  _detector->detect( rgb2, kp2 );

  cout << "Keypoints of two images: " << kp1.size() << "and" << kp2.size() << endl;

  // visualization
  cv::Mat imgShow;
  cv::drawKeypoints( rgb1, kp1, imgShow, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::imshow("keypoints", imgShow);
  cv::imwrite("./data/keypoints.png", imgShow);
  cv::waitKey(0); // pause and wait for a key push

  cv::Mat desp1, desp2;
  _descriptor->compute( rgb1, kp1, desp1);
  _descriptor->compute( rgb2, kp2, desp2);

  // match descriptor
  vector< cv::DMatch> matches;
  cv::FlannBasedMatcher matcher;
  matcher.match( desp1, desp2, matches);
  cout << "Find total " << matches.size() << "matches." << endl;

  // display matched features
  cv::Mat imgMatches;
  cv::drawMatches( rgb1, kp1, rgb2, kp2, matches, imgMatches);
  cv::imshow("matches", imgMatches);
  cv::imwrite("./data/matches.png", imgMatches);
  cv::waitKey(0);

  // abandon the matches whose distances are the 4 times of the minimum
  vector< cv::DMatch> goodMatches;
  double minDis = 9999;
  for( int i = 0 ; i < matches.size(); i++) {
    if( matches[i].distance < minDis)
      minDis = matches[i].distance;
  }

  for( int i = 0; i < matches.size(); i++) {
    if( matches[i].distance < 4 * minDis)
      goodMatches.push_back( matches[i]);
  }

  // display good matches
  cout<<"good matches="<<goodMatches.size()<<endl;
  cv::drawMatches( rgb1, kp1, rgb2, kp2, goodMatches, imgMatches );
  cv::imshow( "good matches", imgMatches );
  cv::imwrite( "./data/good_matches.png", imgMatches );
  cv::waitKey(0);

  // motion model relation
  // cv::solvePnPRansac()
  // 3d point in the first frame
  vector< cv::Point3f> pts_obj;
  vector< cv::Point2f> pts_img;

  // intrinsic parameters of the camera
  CAMERA_INTRINSIC_PARAMETERS C;
  C.cx = 325.5;
  C.cy = 253.5;
  C.fx = 518.0;
  C.fy = 519.0;
  C.scale = 1000.0;

  cv::imshow("Depth1", depth1);

  for( size_t i = 0; i < goodMatches.size(); i++) {

    // query is the first one and train is the second one
    cv::Point2f p = kp1[goodMatches[i].queryIdx].pt;

    // get depth
    // y downward, x rightward. don't get confused.
    int d = depth1.ptr<int>( int(p.y))[ int(p.x) ];
    if(d==0)
      continue;
    pts_img.push_back( cv::Point2f( kp2[goodMatches[i].trainIdx].pt ));

    // transform (u,v,d) to (x,y,z);
    cv::Point3f pt(p.x, p.y, d);
    cv::Point3f pd = point2dTo3d(pt, C);
    pts_obj.push_back(pd);
  }
  
  double camera_matrix_data[3][3] = {
    {C.fx, 0, C.cx},
    {0, C.fy, C.cy},
    {0,0,1}
  };

  // camera matrix construction
  cv::Mat cameraMatrix(3, 3, CV_64F, camera_matrix_data);
  cv::Mat rvec, tvec, inliers;

  // solve pnp
  cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 100, inliers );

   cout<<"inliers: "<<inliers.rows<<endl;
    cout<<"R="<<rvec<<endl;
    cout<<"t="<<tvec<<endl;

    // 画出inliers匹配 
    vector< cv::DMatch > matchesShow;
    for (size_t i=0; i<inliers.rows; i++)
    {
        matchesShow.push_back( goodMatches[inliers.ptr<int>(i)[0]] );    
    }
    cv::drawMatches( rgb1, kp1, rgb2, kp2, matchesShow, imgMatches );
    cv::imshow( "inlier matches", imgMatches );
    cv::imwrite( "./data/inliers.png", imgMatches );
    cv::waitKey( 0 );

    return 0;
}
