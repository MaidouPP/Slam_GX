// -*- C++ -*-
#include <iostream>
#include <string>
using namespace std;

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// define data cloud
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// camera parameters
const double camera_factor=1000;
const double camera_cx = 325.5;
const double camera_cy = 253.5;
const double camera_fx = 518.0;
const double camera_fy = 519.0;

int main( int argc, char** argv) {

  // read rgb and depth image

  cv::Mat rgb, depth;
  // use cv::imread to read the image
  rgb = cv::imread("./rgb.png");
  // rgb image is 8UC3
  // depth image is 16UC1
  // -1 means no change to original image
  depth = cv::imread("./depth.png");

  // data cloud
  PointCloud::Ptr cloud( new PointCloud );
  // iterate on depth image
  for (int m=0; m<depth.rows;m++) {
    for (int n=0; n<depth.cols;n++) {
      // get (m,n) data in depth image
      unsigned short d = depth.ptr<unsigned short>(m)[n];
      if(d==0)
	continue;

      // create a data
      PointT p;
      p.z = double(d) / camera_factor;
      p.x = (n - camera_cx)* p.z / camera_fx;
      p.y = (m - camera_cy)* p.z / camera_fy;

      // get the color from rgb
      p.b = rgb.ptr<uchar>(m)[n*3];
      p.g = rgb.ptr<uchar>(m)[n*3+1];
      p.r = rgb.ptr<uchar>(m)[n*3+2];

      // add p to the cloud cluster
      cloud->points.push_back(p);
    }
  }

  cloud->height = 1;
  cloud->width = cloud->points.size();
  cout<< "point cloud size = " << cloud->points.size() << endl;
  cloud->is_dense = false;
  pcl::io::savePCDFile("./data/pointcloud.pcd", *cloud);

  cloud->points.clear();
  cout << "Point cloud saved." << endl;
  return 0;
}
