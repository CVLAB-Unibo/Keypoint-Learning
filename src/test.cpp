/*
* Software License Agreement (BSD License)
*
*  Point Cloud Library (PCL) - www.pointclouds.org
*  Copyright (c) 2010-2011, Willow Garage, Inc.
*
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of Willow Garage, Inc. nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*/

#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/visualization/pcl_visualizer.h>
#define PCL_NO_PRECOMPILE
#include "KeypointLearning.h"
#undef PCL_NO_PRECOMPILE


/*
Sample program that loads a point cloud passed as argv[0] and show the keypoints detected by the KeypointLearningDetector.
This sample use the forest "../data/tree.yaml" and the default parameters for that instance
\author Alessio Tonioni
*/
int main(int argc, char** argv){
	//create detector
	pcl::keypoints::KeypointLearningDetector<pcl::PointXYZ, pcl::PointXYZI>::Ptr detector(new pcl::keypoints::KeypointLearningDetector<pcl::PointXYZ, pcl::PointXYZI>());
	detector->setNAnnulus(5);
	detector->setNBins(10);
	detector->setNonMaxima(true);
	detector->setNonMaxRadius(4);
	detector->setNonMaximaDrawsRemove(false);
	detector->setPredictionThreshold(0.80);
	detector->setRadiusSearch(40);  //40mm change according to the unit of measure used in your point cloud
	detector->loadForest("../data/forest/tree.yaml.gz");
	std::cout << "Detector created" << std::endl;

	//load and subsample point cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<int> sampled_cloud;
	pcl::io::loadPCDFile("../data/point_cloud_test/cheff000.pcd", *temp);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr sub_sampler_tree(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::UniformSampling<pcl::PointXYZ>::Ptr sub_sampler(new pcl::UniformSampling<pcl::PointXYZ>());
	sub_sampler->setRadiusSearch(2); //2mm
	sub_sampler->setSearchMethod(sub_sampler_tree);
	sub_sampler->setInputCloud(temp);
	sub_sampler->compute(sampled_cloud);
	for (int j = 0; j < sampled_cloud.points.size(); j++)
	{
		cloud->push_back(temp->points[sampled_cloud.points[j]]);
	}
	std::cout << "Point cloud loaded" << std::endl;

	//compute normals
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	ne.setInputCloud(cloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setKSearch(10);
	ne.setSearchMethod(kdtree);
	ne.compute(*normals);
	std::cout << "Normals Computed" << std::endl;

	//setup the detector
	detector->setInputCloud(cloud);
	detector->setNormals(normals);

	//detect keypoints
	pcl::PointCloud<pcl::PointXYZI>::Ptr keypoint(new pcl::PointCloud<pcl::PointXYZI>());
	detector->compute(*keypoint);
	std::cout << "Keypoint computed" << std::endl;

	//show results
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	viewer.reset(new pcl::visualization::PCLVisualizer);
	viewer->setBackgroundColor(0, 0, 0);
	viewer->initCameraParameters();

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr colorKeypoint(new pcl::PointCloud<pcl::PointXYZRGBA>());
	pcl::copyPointCloud(*keypoint, *colorKeypoint);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> rgb(colorKeypoint, 255, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, "model cloud");
	viewer->addPointCloud<pcl::PointXYZRGBA>(colorKeypoint, rgb, "Keypoint");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "Keypoint");
	std::cout << "viewer ON" << std::endl;
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
	viewer->removeAllPointClouds();
	std::cout << "DONE" << std::endl;

}