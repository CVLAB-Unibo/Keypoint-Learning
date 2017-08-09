/*
* Software License Agreement (BSD License)
*
*  Copyright (c) 2017, Alessio Tonioni, Riccardo Spezialetti
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
*   * Neither the name of the copyright holder(s) nor the names of its
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
*
*/

#include <boost\program_options.hpp>

#include <ml.h>

#include <pcl\io\pcd_io.h>
#include <pcl\features\normal_3d.h>
#include <pcl\features\integral_image_normal.h>
#include <pcl\filters\uniform_sampling.h>
#include <pcl\visualization\pcl_visualizer.h>

#define PCL_NO_PRECOMPILE
#include "KeypointLearning.h"
#undef PCL_NO_PRECOMPILE

namespace po = boost::program_options;

bool parseCommandLine(int argc, char** argv, po::variables_map & vm)
{
	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("flipNormals", "If present flip normals, some dataset needs normal re-orientation.")
		("subSampling", "If present, subsample cloud with leaf.")
		("leaf", po::value<float>(), "Leaf size for subsampling.")
		("pathCloud", po::value<std::string>()->default_value("../../../data/point_cloud_test/cheff001.pcd"), "Path to dataset.")
		("pathRF", po::value<std::string>()->default_value("../../../data/forest/tree.yaml.gz"), "Path to Random Forest.")
		("pathKP", po::value<std::string>(), "Path for keypoints point cloud.")
		("radiusFeatures", po::value<float>()->default_value(20.f), "Radius for features computation.")
		("radiusNMS", po::value<float>()->default_value(4.f), "Radius for non maxima suppresion.")
		("threshold,t", po::value<float>()->default_value(0.85), "Threshold for random forest prediction.")
		;
	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
	}
	catch (std::exception & e) {
		std::cerr << e.what() << std::endl;
		std::cout << desc << "\n";
		return false;
	}

	if (vm.count("help")) {
		std::cout << desc << "\n";
		return false;
	}

	if (vm.count("subSampling") > 0 && vm.count("leaf") < 0)
	{
		std::cout << "Subsampling needs leaf." << std::endl;
		return false;
	}

	return true;
}

typedef pcl::PointXYZ	PointInT;
typedef pcl::Normal		PointNormalT;
typedef pcl::PointXYZI	KeypointT;


/*
Program that loads a point cloud passed as and show the keypoints detected by the KeypointLearningDetector.
This sample use the forest "../data/tree.yaml" and the default parameters for that instance.
\author Alessio Tonioni
\author Riccardo Spezialetti
*/

#define ANNULI	5
#define BINS	10

int main(int argc, char** argv)
{

	boost::program_options::variables_map vm;
	if (!parseCommandLine(argc, argv, vm))
		return 0;

	const float radius_nms = vm["radiusNMS"].as<float>();
	const float radius_features = vm["radiusFeatures"].as<float>();
	const float threshold = vm["threshold"].as<float>();

	const std::string path_rf = vm["pathRF"].as<std::string>();
	const std::string path_cloud = vm["pathCloud"].as<std::string>();

	//create detector
	pcl::keypoints::KeypointLearningDetector<PointInT, KeypointT>::Ptr detector(new pcl::keypoints::KeypointLearningDetector<PointInT, KeypointT>());
	detector->setNAnnulus(ANNULI);
	detector->setNBins(BINS);
	detector->setNonMaxima(true);
	detector->setNonMaxRadius(radius_nms);
	detector->setNonMaximaDrawsRemove(false);
	detector->setPredictionThreshold(threshold);
	detector->setRadiusSearch(radius_features);  //40mm change according to the unit of measure used in your point cloud
	
	if (detector->loadForest(path_rf)) 
	{
		std::cout << "Detector created." << std::endl;
	}
	else 
	{
		return -1;
	}

	//load and subsample point cloud
	pcl::PointCloud<PointInT>::Ptr cloud(new pcl::PointCloud<PointInT>());
	pcl::io::loadPCDFile(path_cloud, *cloud);

 	pcl::UniformSampling<PointInT>::Ptr source_uniform_sampling(new pcl::UniformSampling<PointInT>());

	bool sub_sampling = vm.count("subSampling") > 0;
	float leaf = 0.0;

	if(sub_sampling)
	{
		leaf = vm["leaf"].as<float>();

		source_uniform_sampling->setRadiusSearch(leaf);
		source_uniform_sampling->setInputCloud(cloud);
		source_uniform_sampling->filter(*cloud);
	}

	std::cout << "Point cloud loaded" << std::endl;

	//Compute normals
	pcl::NormalEstimation<PointInT, PointNormalT> ne;
	pcl::PointCloud<PointNormalT>::Ptr normals(new pcl::PointCloud<PointNormalT>);
	ne.setInputCloud(cloud);

	pcl::search::KdTree<PointInT>::Ptr kdtree(new pcl::search::KdTree<PointInT>());
	ne.setKSearch(10);
	ne.setSearchMethod(kdtree);
	ne.compute(*normals);
	std::cout << "Normals Computed" << std::endl;

	//Set true with laser scanner dataset
	bool flip_normals = vm.count("flipNormals") > 0;
	if (flip_normals)
	{
		std::cout << "Flipping "<< std::endl;
		//Sensor origin is inside the model, flip normals to make normal sign coherent with scenes
		std::transform(normals->points.begin(), normals->points.end(), normals->points.begin(), [](pcl::Normal p) -> pcl::Normal {auto q = p; q.normal[0] *= -1; q.normal[1] *= -1; q.normal[2] *= -1; return q; });
	}

	//setup the detector
	detector->setInputCloud(cloud);
	detector->setNormals(normals);

	//detect keypoints
	pcl::PointCloud<KeypointT>::Ptr keypoint(new pcl::PointCloud<KeypointT>());
	detector->compute(*keypoint);
	std::cout << "Keypoint computed" << std::endl;

	//show results
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	viewer.reset(new pcl::visualization::PCLVisualizer);
	viewer->setBackgroundColor(0, 0, 0);
	

	pcl::visualization::PointCloudColorHandlerCustom<PointInT> blu(cloud, 0, 0, 255);

	viewer->addPointCloud<pcl::PointXYZ>(cloud, blu,"model cloud");

	pcl::visualization::PointCloudColorHandlerCustom<KeypointT> red(keypoint, 255, 0, 0);
	viewer->addPointCloud<KeypointT>(keypoint, red, "Keypoint");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "Keypoint");
	std::cout << "viewer ON" << std::endl;
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
	viewer->removeAllPointClouds();
	std::cout << "DONE" << std::endl;

	if (vm.count("pathKP")) 
	{
		const std::string path = vm["pathKP"].as<std::string>();
		pcl::io::savePCDFileASCII<KeypointT>(path, *keypoint);
	}
	

}
