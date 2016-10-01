#pragma once

#include <fstream>
#include <vector>
#include <map>
using namespace std;

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// camera parameters
struct CAMERA_INTRINSIC_PARAMETERS {
  double cx,cy,fx,fy,scale;
};

struct FRAME {
  cv::Mat rgb, depth; 
  cv::Mat desp; // descriptor 
  vector< cv::KeyPoint> kp; // keypoints
};

// result of pnp
struct RESULT_OF_PNP {
  cv::Mat rvec, tvec;
  int inliers;
};

// compute keypoints and extract descriptors
void computeKeyPointsAndDesp( FRAME& frame, string detector, string descriptor);

// compute the relative motion
RESULT_OF_PNP estimateMotion( FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS camera);


// image2PointCloud rgb -> point cloud
PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera);

// point2dTo3d single point -> spatial coordinate
// input: 3d Point3f(u,v,d) image
cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera);

class ParametersReader {
 public:
  ParametersReader(string filename="./parameters.txt") {
    ifstream fin( filename.c_str());
    if(!fin) {
      cerr << "parameter file does not exist"<< endl;
    }
    while(!fin.eof()) {
      string str;
      getline(fin, str);
      if(str[0] == '#')
	continue;
      int pos = str.find("=");
      if(pos==-1)
	continue;
      string key = str.substr(0, pos);
      string value = str.substr(pos+1, str.length());
      data[key] = value;

      // check if the file stream is normal
      if(!fin.good())
	break;
    }
  }

  string getData(string key) {
    map<string, string>::iterator iter = data.find(key);
    if(iter==data.end()) {
      cerr << "parameter name " << key << " not found!" << endl;
      return string("NOT_FOUND");
    }
    return iter->second;
  }

 public:
  map<string,string> data;
}
