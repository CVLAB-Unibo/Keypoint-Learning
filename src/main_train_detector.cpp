/*
* Software License Agreement (BSD License)
*
*  Copyright (c) 2017, Riccardo Spezialetti, riccardo.spezialetti@unibo.it
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

#include <chrono>
#include <iostream>
#include <vector>

#include <boost\format.hpp>
#include <boost\program_options.hpp>
#include <boost\range\algorithm.hpp>
#include <boost\timer.hpp>

#include <omp.h>

#include <ml.h>
#include <opencv2\core\core.hpp>

#include <pcl\io\pcd_io.h>
#include <pcl\io\ply_io.h>
#include <pcl\filters\uniform_sampling.h>
#include <pcl\features\normal_3d_omp.h>
#include <pcl\point_types.h>
#include <pcl\search\kdtree.h>
#include <pcl\visualization\pcl_plotter.h>
#include <pcl\visualization\pcl_visualizer.h>

#include "general_utilities.h"
#include "point_cloud_utilities.h"

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
		("annuli,a", po::value<int>()->required(), "Annuli for features computation.")
		("batch,b", "If present, runs in batch mode, no visualization. ")
		("bins", po::value<int>()->required(), "Bins for features computation.")
		("depth,d", po::value<int>()->required(), "Depth of Random Forest.")
		("ext,e", po::value<std::string>()->default_value(".pcd"), "Dataset extension.")
		("flipNormals", "If present flip normals, some dataset needs normal re-orientation.")	
		("leaf", po::value<float>(), "Leaf size for subsampling.")
		("pathDataset", po::value<std::string>()->required(), "Path to dataset.")
		("pathTrainingData", po::value<std::string>()->required(), "Path for training data-> Positives in: pathTrainingData\\Model_Name\\positives and Negatives in: pathTrainingData\\Model_Name\\negatives.")
		("pathRF", po::value<std::string>()->required(), "Path for Random Forest.")
		("radiusFeatures", po::value<float>()->required(), "Radius for features computation.")
		("radiusNormals", po::value<float>(), "Radius for normals computation with mode 1.")
		("msc", po::value<int>()->required(), "Min samples count of Random Forest.")
		("nameRF", po::value<std::string>()->required(), "Name of YAML file.")
		("ntrees", po::value<int>()->required(), "Number of trees of Random Forest.")
		("normalsMode", po::value<int>()->required(), "0 normals with knn || 1 normals with radius.")
		("nnNormals", po::value<int>()->required(), "Number of NN forn normals computation with normalsMode 0.")
		("showFeatures", "If present and not in batch mode, for each positive and negative point plot histogram of compute features.")
		("showNormals", "If present and not in batch mode, show normals.")
		("showTrainingSet", "If present and not in batch mode, show positives and negative from training set.")
		("subSampling", "If present, subsample cloud with leaf.")
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

	if(vm["normalsMode"].as<int>() == 0 && vm.count("nnNormals") < 0)
	{
		std::cout << "Knn normals needs NN."<< std::endl;
		return false;
	}

	if (vm["normalsMode"].as<int>() == 1 && vm.count("radiusNormals") < 0)
	{
		std::cout << "Radius normals needs radius." << std::endl;
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


static cv::Ptr<cv::ml::TrainData>
prepare_train_data(const cv::Mat& data, const cv::Mat& responses, int ntrain_samples)
{
	cv::Mat sample_idx = cv::Mat::zeros(1, data.rows, CV_8U);
	cv::Mat train_samples = sample_idx.colRange(0, ntrain_samples);
	train_samples.setTo(cv::Scalar::all(1));

	int nvars = data.cols;
	cv::Mat var_type(nvars + 1, 1, CV_8U);
	var_type.setTo(cv::Scalar::all(cv::ml::VAR_ORDERED));
	var_type.at<uchar>(nvars) = cv::ml::VAR_CATEGORICAL;

	return cv::ml::TrainData::create(data, cv::ml::ROW_SAMPLE, responses, cv::noArray(), sample_idx, cv::noArray(), var_type);
}

int main(int argc, char** argv)
{
	////////////////////////////////////////////////////////////////////
	//////---------------------- COMMAND LINE	 ----------------------
	////////////////////////////////////////////////////////////////////
	boost::program_options::variables_map vm;
	if (!parseCommandLine(argc, argv, vm))
		return 0;

	//Path dataset
	const std::string path_dataset = vm["pathDataset"].as < std::string >();
	const std::string ext_dataset = vm["ext"].as < std::string >();

	//Path training data
	const std::string path_td = vm["pathTrainingData"].as<std::string>();

	//Path for random forest
	const std::string path_rf = vm["pathRF"].as <std::string>();

	boost::filesystem::path dir_random_forest_dir(path_rf);
	if (!boost::filesystem::exists(dir_random_forest_dir))
	{
		std::cout << "Random forest's path doesn't exists. Now Exit!" << std::endl;
		return 0;
	}
	//Random Forest parameters
	const std::string name_rf = vm["nameRF"].as<std::string>() + ".yaml";
	const int msc = vm["msc"].as < int >();
	const int ntrees = vm["ntrees"].as < int >();
	const int depth = vm["depth"].as < int >();

	//Subsampling cloud
	pcl::UniformSampling<PointInT>::Ptr uniform_sampler;
	const bool subsampling = vm.count("subSampling") > 0;
	float sub_sampler_leaf = 0;
	if (subsampling)
	{
		uniform_sampler.reset(new pcl::UniformSampling<PointInT>());

		sub_sampler_leaf = vm["leaf"].as <float>();
		uniform_sampler->setRadiusSearch(sub_sampler_leaf);
	}

	//Normals
	pcl::NormalEstimationOMP<PointInT, PointNormalT> normal_estimator;
	normal_estimator.setNumberOfThreads(omp_get_max_threads());
	const bool flip_normals = vm.count("flipNormals") > 0;
	float radius_normals = 0.0;
	int nn_normals = 0;

	const bool normals_on_radius = vm["normalsMode"].as<int>();
	if (normals_on_radius)
	{
		radius_normals = vm["radiusNormals"].as<float>();
	}
	else
	{
		nn_normals = vm["nnNormals"].as<int>();
	}

	//Features parameters
	const int annuli = vm["annuli"].as<int>();
	const int bins = vm["bins"].as<int>();
	const float radius_features = vm["radiusFeatures"].as<float>();

	std::ofstream f_training_parameters(path_rf + "training_parameters.log");

	////////////////////////////////////////////////////////////////
	//------COMMAND LINE PARAMTERS TO SAVE IN FILE-----------------
	////////////////////////////////////////////////////////////////
	f_training_parameters << "Name rf: " << boost::lexical_cast<std::string>(name_rf) << std::endl;
	f_training_parameters << "Number of trees: " << boost::lexical_cast<std::string>(ntrees) << std::endl;
	f_training_parameters << "Depth of each tree: " << boost::lexical_cast<std::string>(depth) << std::endl;
	f_training_parameters << "Dataset: " << path_dataset << std::endl;
	f_training_parameters << "Training data: " << path_td << std::endl;
	f_training_parameters << "Msc: " << boost::lexical_cast<std::string>(msc) << std::endl;
	f_training_parameters << "Features annuli: " << boost::lexical_cast<std::string>(annuli) << std::endl;
	f_training_parameters << "Features bins: " << boost::lexical_cast<std::string>(bins) << std::endl;
	f_training_parameters << "Features radius: " << boost::lexical_cast<std::string>(radius_features) << std::endl;
	f_training_parameters << "Forest path: " << path_rf << std::endl;

	//Training data
	cv::Mat label_data(0, 1, CV_32SC1);
	cv::Mat training_data;

	//Visualization parameters
	const bool batch_mode = vm.count("batch") > 0;
	const bool show_normals = vm.count("showNormals") > 0;
	const bool show_features = vm.count("showFeatures") > 0;
	const bool show_ts = vm.count("showTrainingSet") > 0;
	
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	float r_sphere_visualization = 0;
	int scale_coordinates_system_visualization = 0;

	boost::shared_ptr<pcl::visualization::PCLPlotter> plotter_viewer;

	unsigned total_positives = 0;
	unsigned total_negatives = 0;

	double time_training = 0.0;

	if (!batch_mode)
	{
		viewer.reset(new pcl::visualization::PCLVisualizer("Model_Viewer"));
		viewer->setBackgroundColor(0, 0, 0);

		if (show_features)
			plotter_viewer.reset(new pcl::visualization::PCLPlotter("Features_Viewer"));
	}

	//////////////////////////////////////////////////////////////////
	////-------------- SETTINGS FOR FOREST---------------------------
	//////////////////////////////////////////////////////////////////
	cv::Ptr<cv::ml::RTrees> forest_ = cv::ml::RTrees::create();

	forest_->setMaxDepth(depth);
	forest_->setMinSampleCount(msc);
	forest_->setRegressionAccuracy(0);
	forest_->setMaxCategories(15);
	forest_->setUseSurrogates(false);
	forest_->setPriors(cv::Mat());
	forest_->setCalculateVarImportance(true);
	forest_->setActiveVarCount(0);

	forest_->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, ntrees, 1e-6));
	//////////////////////////////////////////////////////////////////
	////------------DETECTOR USED FOR FEATURES COMPUTATION-----------
	//////////////////////////////////////////////////////////////////
	pcl::keypoints::KeypointLearningDetector<PointInT, KeypointT> detector;

	for (boost::filesystem::directory_iterator dataset_iterator(path_dataset); dataset_iterator != boost::filesystem::directory_iterator(); dataset_iterator++)
	{
		std::vector<std::string> v_clouds_name;
		if (kpl::getAllFilesInFolder(dataset_iterator->path(), ext_dataset, v_clouds_name) < 0)
		{
			std::cerr << "No files for " << dataset_iterator->path() << "with extension " << ext_dataset << std::endl;
			continue;
		}

		std::string folder_name = dataset_iterator->path().filename().string();
		for (size_t i_f = 0; i_f < v_clouds_name.size(); ++i_f)
		{
			//Read view
			pcl::PointCloud<PointInT>::Ptr c_points(new pcl::PointCloud<PointInT>);
			std::string f_name = dataset_iterator->path().string() + "\\" + v_clouds_name[i_f];

			if (ext_dataset == ".pcd")
			{
				if (pcl::io::loadPCDFile<PointInT>(f_name, *c_points) == -1)
				{
					std::cerr << "Impossible to read point cloud from: " << v_clouds_name[i_f] << std::endl;
					exit(-1);
				}
			}

			if (ext_dataset == ".ply")
			{
				if (pcl::io::loadPLYFile(f_name, *c_points) == -1)
				{
					std::cerr << "Impossible to read point cloud from: " << v_clouds_name[i_f] << std::endl;
					exit(-1);
				}
			}

			//Read positives
			pcl::PointCloud<KeypointT>::Ptr c_positives(new pcl::PointCloud<KeypointT>());
			const std::string f_name_pos = path_td + "\\" + folder_name + "\\positives\\" + v_clouds_name[i_f];
			
			if (pcl::io::loadPCDFile(f_name_pos, *c_positives) == -1)
			{
				std::cerr << "Impossible to read positives cloud for: " << v_clouds_name[i_f] << std::endl;
				continue;
			}

			//Read negatives
			pcl::PointCloud<KeypointT>::Ptr c_negatives(new pcl::PointCloud<KeypointT>());
			const std::string f_name_neg = path_td + "\\" + folder_name + "\\negatives\\" + v_clouds_name[i_f];

			if (pcl::io::loadPCDFile(f_name_neg, *c_negatives) == -1)
			{
				std::cerr << "Impossible to read positives cloud for: " << v_clouds_name[i_f] << std::endl;
				exit(-1);
			}

			////////////////////////////////////////////////////////////////
			//------------- UNIFORM SAMPLING AND NAN REMOVAL --------------
			////////////////////////////////////////////////////////////////
			std::vector<int> indices;
			pcl::removeNaNFromPointCloud(*c_points, *c_points, indices);

			//Uniform sampling
			if (subsampling)
			{
				uniform_sampler->setInputCloud(c_points);
				uniform_sampler->filter(*c_points);
			}

			const double cloud_resolution = kpl::computeCloudResolution<PointInT>(c_points);
			////////////////////////////////////////////////////////////////
			//------------------- NORMALS ESTIMATION----------------------//
			////////////////////////////////////////////////////////////////
			pcl::PointCloud<PointNormalT>::Ptr c_normals(new pcl::PointCloud<PointNormalT>);
			normal_estimator.setInputCloud(c_points);

			if(normals_on_radius)
				normal_estimator.setRadiusSearch(radius_normals);
			else
				normal_estimator.setKSearch(nn_normals);

			normal_estimator.compute(*c_normals);

			if (flip_normals)
			{
				//Sensor origin is inside the model, flip normals to make normal sign coherent with scenes
				std::transform(c_normals->points.begin(), c_normals->points.end(), c_normals->points.begin(), [](pcl::Normal p) -> pcl::Normal {auto q = p; q.normal[0] *= -1; q.normal[1] *= -1; q.normal[2] *= -1; return q; });
			}
			
			if (!batch_mode)
			{
				pcl::visualization::PointCloudColorHandlerRandom<PointInT> random_color(c_points);
				viewer->addPointCloud<PointInT>(c_points, random_color, "cloud");
				viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, "cloud");

				if (show_normals) 
				{
					viewer->addPointCloudNormals<PointInT,PointNormalT>(c_points, c_normals, 100, 5 * cloud_resolution, "cloud_with_normals");
				}

				if (show_ts)
				{
					pcl::visualization::PointCloudColorHandlerCustom<KeypointT> green(c_positives,0, 255, 0);
					viewer->addPointCloud<KeypointT>(c_positives, green, "positives");
					viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8.0, "positives");

					pcl::visualization::PointCloudColorHandlerCustom<KeypointT> red(c_negatives, 255, 0, 0);
					viewer->addPointCloud<KeypointT>(c_negatives, red, "negatives");
					viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8.0, "negatives");
				}

				viewer->spinOnce();
				while (!viewer->wasStopped())
				{
					viewer->spinOnce(100);
				}
				viewer->removeAllPointClouds();
				viewer->removeAllShapes();
			}

			////////////////////////////////////////////////////////////////
			//--------------------- COMPUTE FEATURES ---------------------//
			////////////////////////////////////////////////////////////////
			//Intensity=0--->Keypoint, Intensity=1--->NOT Keypoint
			std::transform(c_positives->points.begin(), c_positives->points.end(), c_positives->points.begin(), [](pcl::PointXYZI kp) -> pcl::PointXYZI {auto q = kp; q.intensity = 0.0f; return q; });
			std::transform(c_negatives->points.begin(), c_negatives->points.end(), c_negatives->points.begin(), [](pcl::PointXYZI kp) -> pcl::PointXYZI {auto q = kp; q.intensity = 1.0f; return q; });

			pcl::PointCloud<KeypointT> cloud_ts;
			cloud_ts += *c_positives;
			cloud_ts += *c_negatives;

			detector.setInputCloud(c_points);
			detector.setNormals(c_normals);
			detector.setNAnnulus(annuli);
			detector.setNBins(bins);
			detector.setRadiusSearch(radius_features);

			pcl::KdTreeFLANN<PointInT> kdtree_input_cloud;
			kdtree_input_cloud.setInputCloud(c_points);

			pcl::PointIndicesPtr indices_ts(new pcl::PointIndices());
			indices_ts->indices.resize(cloud_ts.points.size());

			for (size_t i_p = 0; i_p < cloud_ts.points.size(); ++i_p)
			{
				std::vector<int> indices(1);
				std::vector<float> distances(1);

				PointInT point_to_search;
				pcl::copyPoint<KeypointT, PointInT>(cloud_ts.points[i_p], point_to_search);

				kdtree_input_cloud.nearestKSearch(point_to_search, 1, indices, distances);
				indices_ts->indices[i_p] = indices[0];
			}

			std::cout << "[START] Compute features for: "<< v_clouds_name[i_f] << std::endl;

			cv::Mat points_features = detector.computePointsForTrainingFeatures(indices_ts);

			std::cout << "[END] Compute features for: " << v_clouds_name[i_f] << std::endl;

			for(size_t i=0; i < points_features.rows; ++i)
			{
				training_data.push_back(points_features.row(i));
				label_data.push_back(cloud_ts.points[i].intensity);

				if (cloud_ts.points[i].intensity == 0)
					total_positives++;
				else
					total_negatives++;

				if (!batch_mode && show_features)
				{
					const int max_dimension = 100;
					pcl::Histogram<max_dimension> histogram_pos_point;
					memcpy(histogram_pos_point.histogram, (float*)points_features.data, points_features.cols * sizeof(float));

					pcl::PointCloud<pcl::Histogram<max_dimension>>::Ptr histogram_cloud(new pcl::PointCloud<pcl::Histogram<max_dimension>>());
					histogram_cloud->points.push_back(histogram_pos_point);

					const std::string title = "POS features";

					plotter_viewer->setTitle(title.c_str());
					plotter_viewer->setXRange(0, points_features.cols);
					plotter_viewer->addFeatureHistogram<pcl::Histogram<max_dimension>>(*histogram_cloud, points_features.cols, "histogram_cloud");

					plotter_viewer->spin();
					while (!plotter_viewer->wasStopped())
					{
						plotter_viewer->plot();
					}
					plotter_viewer->clearPlots();
				}
			}

		}//Views

	}//Models

	std::cout << "Training data: " << total_positives << " positives and " << total_negatives << " negatives." << std::endl;
	std::cout << "Training labels: " << label_data.rows << " points." << std::endl;
	
	//////////////////////////////////////////////////////////////////
	////--------------- TRAINING    FOREST---------------------------
	//////////////////////////////////////////////////////////////////
	if (training_data.rows != label_data.rows)
	{
		perror("Impossible to compute train Random Forest: training data are different from label data.");
	}
	else 
	{
		std::chrono::time_point<std::chrono::system_clock> train_start, train_end;
		train_start = std::chrono::system_clock::now();
		
		int ntrain_samples = (int)(training_data.rows*0.8);
		cv::Ptr<cv::ml::TrainData> t_data = prepare_train_data(training_data, label_data, ntrain_samples);

		const bool is_trained = forest_->train(t_data);
		
		train_end = std::chrono::system_clock::now();
		std::chrono::duration<double> train_elapsed_seconds = train_end - train_start;

		if (is_trained) 
		{
			std::cout << "Forest trained in: " << train_elapsed_seconds.count()/60.0 << " minutes." << std::endl;
			std::cout << "Training error: " << forest_->calcError(t_data,false,cv::noArray()) << std::endl;

			f_training_parameters << "Trained with: " << total_positives << " positives and " << total_negatives  << " negatives."<< std::endl;
			f_training_parameters << "Train duration in seconds: " << boost::lexical_cast<std::string>(train_elapsed_seconds.count()) << std::endl;
		
			forest_->save(path_rf + "\\" + name_rf);
		}
	}

	forest_.release();

	return 0;
}