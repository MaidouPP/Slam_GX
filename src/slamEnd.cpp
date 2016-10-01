#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "slamBase.h"

#include <g2o/types/slam3d/types_slam3d.h>  // types of vertex
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

// give the index and return this frame
FRAME readFrame( int index, ParameterReader& pd);
// measure the motion
double normofTransform( cv::Mat rvec, cv::Mat tvec);

int main( int argc, char** argv) {
  ParameterReader pd;
  int startIndex = atoi(pd.getData("start_index").c_str());
  int endIndex = atoi(pd.getData("end_index").c_str());

  // initialize
  cout << "Initializing... " << endl;
  int currIndex = startIndex;
  FRAME lastFrame = readFrame( currIndex, pd); // last frame
  // we always compare this one and last frame
  string detector = pd.getData("detector");
  string descriptor = pd.getData("descriptor");
  CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
  computeKeyPointsAndDesp( lastFrame, detector, descriptor );
  PointCloud::Ptr cloud = image2PointCloud( lastFrame.rgb, lastFrame.depth, camera );

  pcl::visualization::CloudViewer viewer("viewer");

  bool visualize = pd.getData("visualize_pointcloud")==string("yes");
  int min_inliers = atoi( pd.getData("min_inliers").c_str() );
  double max_norm = atof( pd.getData("max_norm").c_str() );

  /*************
  // add optimization here
  ************/

  // choose a optimization method
  typedef g2o::BlockSolver_6_3 SlamBlockSolver;
  typedef g2o::LinearSolverCSparse< SlamBlockSolver::PoseMatrixType > SlamLinearSolver;

  // initilize solver
  SlamLinearSolver* linearSolver = new SlamLinearSolver();
  linearSolver->setBlockOrdering(false);
  SlamBlockSolver* blockSolver = new SlamBlockSolver( linearSolver );
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);

  g2o::SparseOptimizer globalOptimizer;  // use it in the end
  globalOptimizer.setAlgorithm( solver );
  // no debugging messages
  globalOptimizer.setVerbose( false );

  // add the first vertec to the graph
  g2o::VertexSE3* v = new g2o::VertexSE3();
  v->setId( currIndex );
  v->setEstimate( Eigen::Isometry3d::Identity() );
  v->setFixed( true ); // fix the first one. It doesn't need to be optimized.
  globalOptimizer.addVertex( v );

  int lastIndex = currIndex; 

  for ( currIndex=startIndex+1; currIndex<endIndex; currIndex++) {

    cout << "Reading files..."<< endl;
    FRAME currFrame = readFrame( currIndex, pd);
    computeKeyPointsAndDesp( currFrame, detector, descriptor);
    // compare currFrame and lastFrame
    RESULT_OF_PNP result = estimateMotion( lastFrame, currFrame, camera);
    if ( result.inliers < min_inliers)
      continue;  // not enough inliers, abandon this one
    // judge if range of motion is too large which is unreasonable
    double norm = normofTransform(result.rvec, result.tvec);
    cout << "not = " << norm << endl;
    if ( norm >= max_norm)
      continue;
    Eigen::Isometry3d T  = cvMat2Eigen( result.rvec, result.tvec);
    cout << "T = " << T.matrix() << endl;


    //cloud = joinPointCloud( cloud, currFrame, T, camera );

    // add this vertec to g2o as well as the edge to last frame
    // vertex part (only need index)
    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setId( currIndex );
    v->setEstimate( Eigen::Isometry3d::Identity());
    globalOptimizer.addVertex(v);

    // edge part
    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
    // two index connecting this edge
    edge->vertices() [0] = globalOptimizer.vertex( lastIndex );
    edge->vertices() [1] = globalOptimizer.vertex( currIndex );

    // information matrix
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();
    // covariance is 0.01, information matrix is 100
    information(0,0) = information(1,1) = information(2,2) = 100;
    information(3,3) = information(4,4) = information(5,5) = 100;

    edge->setInformation( information );

    // edge's estimation is the result of PnP
    edge->setMeasurement( T );
    // add edge to graph
    globalOptimizer.addEdge(edge);
    
    //    if ( visualize == true)
    //      viewer.showCloud( cloud);

    lastFrame = currFrame;
    lastIndex = currIndex;
  }
  // pcl::io::savePCDFile( "data/result.pcd", *cloud);

  // optimize all of the edges
  cout<<"optimizing pose graph, vertices: "<<globalOptimizer.vertices().size()<<endl;
  globalOptimizer.save("./data/result_before.g2o");
  globalOptimizer.initializeOptimization();
  globalOptimizer.optimize( 100 ); //可以指定优化步数
  globalOptimizer.save( "./data/result_after.g2o" );
  cout<<"Optimization done."<<endl;

  globalOptimizer.clear();
  
  return 0;
}

FRAME readFrame( int index, ParameterReader& pd) {
  FRAME f;
  string rgbDir   =   pd.getData("rgb_dir");
  string depthDir =   pd.getData("depth_dir");

  string rgbExt   =   pd.getData("rgb_extension");
  string depthExt =   pd.getData("depth_extension");

  stringstream ss;
  ss<<rgbDir<<index<<rgbExt;
  string filename;
  ss >> filename;
  f.rgb = cv::imread( filename );

  ss.clear();
  filename.clear();
  ss<<depthDir<<index<<depthExt;
  ss>>filename;

  f.depth = cv::imread( filename, -1 );
  return f;
}

double normofTransform( cv::Mat rvec, cv::Mat tvec ) {
  return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}
