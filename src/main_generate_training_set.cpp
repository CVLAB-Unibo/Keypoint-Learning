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

#include <iostream>
#include <vector>
#include <unordered_map>

#include <boost\program_options.hpp>

#include <opencv2\core\core.hpp>

#include <pcl\common\geometry.h>
#include <pcl\common\transforms.h>
#include <pcl\io\pcd_io.h>
#include <pcl\io\ply_io.h>
#include <pcl\features\normal_3d.h>
#include <pcl\point_types.h>
#include <pcl\search\kdtree.h>
#include <pcl\visualization\pcl_visualizer.h>

#define MULTITHREAD
#define MULTIVIEW

#if defined(MULTITHREAD) || defined(MULTIVIEW) 
#define NUMBER_OF_THREADS 8
#endif

#include "general_utilities.h"
#include "point_cloud_utilities.h"
#include "overlap_finder.h"
#include "trainingset_generator.h"
#include "view_manager.h"
#include "view.h"

namespace po = boost::program_options;

bool parseCommandLine(int argc, char** argv, po::variables_map & vm)
{
	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("batch,b", "If present, runs in batch mode, no visualization.")
		("distance,d", po::value<float>()->required(), "Euclidean distance to accept points with similar descriptor.")
		("ext,e", po::value<std::string>()->required(), "Dataset file extension.")
		("flipNormals", "If present flip normals, some dataset needs normal re-orientation.")		
		("leaf", po::value<float>(), "Leaf size for subsampling.")		
		("pathDataset", po::value<std::string>()->required(), "Path to dataset.")
		("pathTrainingset", po::value<std::string>()->required(), "Path for generated traning set.")
		("pathDescriptors", po::value<std::string>(), "If present load descriptors from path.")
		("radiusFeatures", po::value<float>(), "Radius for descriptor evaluation.")
		("radiusNegative", po::value<float>()->required(), "Radius for negative generation.")
		("radiusNormals", po::value<float>(), "Radius for normals computation with mode 1.")
		("radiusNms", po::value<float>()->required(), "Radius for non maxima suppression on positive.")
		("normalsMode", po::value<int>()->default_value(0), "0 normals with knn || 1 normals with radius.")
		("nnNormals", po::value<int>()->default_value(10), "Number of nearest neighbor used for normals computation with mode 0.")
		("overlap", po::value<float>()->required(), "Overlapping threshold between views.")
		("showNormals", "If present and not in batch mode, show normals.")
		("showOverlap", "If present and not in batch mode, show overlapping views.")
		("showPositives", "If present and not in batch mode, show learned positives.")
		("showNegatives", "If present and not in batch mode, show learned negatives.")
		("showSupports", "If present and not in batch mode, show in blu support for descriptor estimation.")
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

	if(vm["normalsMode"].as<int>() == 0 && vm.count("nnNormals") == 0)
	{
		std::cout << "Knn normals needs NN."<< std::endl;
		return false;
	}

	if (vm["normalsMode"].as<int>() == 1 && vm.count("radiusNormals") == 0)
	{
		std::cout << "Radius normals needs radius." << std::endl;
		return false;
	}

	if (vm.count("subSampling") > 0 && vm.count("leaf") == 0)
	{
		std::cout << "Subsampling needs leaf." << std::endl;
		return false;
	}

	if (vm.count("pathDescriptors") == 0 && vm.count("radiusFeatures") == 0) 
	{
		std::cout << "Descriptors evaluation needs radius." << std::endl;
		return false;
	}

	return true;
}

typedef pcl::PointXYZ	PointInT;
typedef pcl::Normal		NormalT;
typedef pcl::PointXYZI	PointOutT;

#define SIZE_DESCRIPTOR 352

#define PCL_NO_PRECOMPILE
POINT_CLOUD_REGISTER_POINT_STRUCT(pcl::Histogram<SIZE_DESCRIPTOR>,
(float[SIZE_DESCRIPTOR], histogram, histogram)
)
#undef PCL_NO_PRECOMPILE

typedef pcl::Histogram<SIZE_DESCRIPTOR> DescriptorT;

namespace pcl {
	template <>
	class DefaultPointRepresentation<DescriptorT> : public PointRepresentation<DescriptorT>
	{
	public:
		DefaultPointRepresentation()
		{
			nr_dimensions_ = SIZE_DESCRIPTOR;
		}

		virtual void
		copyToFloatArray(const DescriptorT &p, float * out) const
		{
			for (int i = 0; i < nr_dimensions_; ++i)
				out[i] = p.histogram[i];
		}
	};
}

int main(int argc, char** argv)
{
	////////////////////////////////////////////////////////////////
	//------------------- READ COMMAND LINE ------------------------
	////////////////////////////////////////////////////////////////
	po::variables_map vm;
	if (!parseCommandLine(argc, argv, vm))
		return -1;

	const bool batch_mode = vm.count("batch") > 0;

	const bool show_supports = vm.count("showSupports") > 0;
	const bool show_normals = vm.count("showNormals") > 0;
	const bool show_overlap = vm.count("showOverlap") > 0;
	const bool show_positives = vm.count("showPositives") > 0;
	const bool show_negatives = vm.count("showNegatives") > 0;
	
	const std::string p_dataset = vm["pathDataset"].as<std::string>();
	const std::string ext_dataset = vm["ext"].as<std::string>();
	const std::string p_trainingset = vm["pathTrainingset"].as<std::string>();
	
	std::string p_descriptors = "";
	bool compute_descriptors = true;
	float rad_descriptor = 0.f;

	if(vm.count("pathDescriptors") > 0)
	{
		p_descriptors = vm["pathDescriptors"].as<std::string>();
		compute_descriptors = false;
	}
	else 
	{
		rad_descriptor = vm["radiusFeatures"].as<float>();
	}

	const bool normals_on_radius = vm["normalsMode"].as<int>();

	float rad_normals = 0.0;
	int nn_normals = 0;

	if(normals_on_radius)
	{
		rad_normals = vm["radiusNormals"].as<float>();
	}
	else 
	{
		nn_normals = vm["nnNormals"].as<int>();
	}
	
	const bool sub_sampling = vm.count("subSampling") > 0;
	float leaf_ss = 0.f;
	if (sub_sampling) 
	{
		leaf_ss = vm["leaf"].as<float>();
	}

	const bool flip_normals = vm.count("flipNormals") > 0;

	const float th_overlap = vm["overlap"].as<float>();

	const float th_distance = vm["distance"].as<float>();
	const float rad_nms = vm["radiusNms"].as<float>();

	const float rad_negative = vm["radiusNegative"].as<float>();
	
	//Visualization
	boost::shared_ptr<pcl::visualization::PCLVisualizer> visualizer;
	if (!batch_mode) 
	{
		visualizer.reset(new pcl::visualization::PCLVisualizer("Positive Visualization"));
	}
	
	//View manager singletone for loading and managing views
	kpl::ViewManager<PointInT, NormalT, DescriptorT> *manager_view = kpl::ViewManager<PointInT, NormalT, DescriptorT>::getInstance();

	//Normals
	manager_view->estimateNormals(compute_descriptors);
	manager_view->setNormalsFlipping(flip_normals);
	manager_view->setNormalsOnRadius(normals_on_radius);

	if (normals_on_radius)
	{
		manager_view->setNormalsRadius(rad_normals);
	}
	else
		manager_view->setNormalsNN(nn_normals);

	//Subsampling configuration
	manager_view->setSubSampling(sub_sampling);
	if(sub_sampling)
	{
		manager_view->setLeaf(leaf_ss);
	}
	//
	kpl::TrainingSetGenerator<PointInT, NormalT, PointOutT, DescriptorT> ts_generator;

	boost::filesystem::path path_dataset(p_dataset.c_str());
	//For each model
	for (boost::filesystem::directory_iterator dataset_iterator(path_dataset); dataset_iterator!= boost::filesystem::directory_iterator(); dataset_iterator++)
	{
	
		//Find views for model
		std::map<int, std::string> map_ids_names;
		kpl::findViews(dataset_iterator->path(), ext_dataset, map_ids_names);

		//Select overlapping views for model
		kpl::FileBasedOverlapFinder overlapper(dataset_iterator->path().string() + "\\overlappingAreas.txt");
		overlapper.setThreshold(th_overlap);

		const std::string folder_name = dataset_iterator->path().filename().string();

		const std::string path_positives_folder = p_trainingset + folder_name + "\\positives\\";
		if(!boost::filesystem::create_directories(path_positives_folder))
		{
			std::cerr << "Impossible to create: " << path_positives_folder << std::endl;
		}

		const std::string path_negatives_folder = p_trainingset + folder_name + "\\negatives\\";
		if (!boost::filesystem::create_directories(path_negatives_folder))
		{
			std::cerr << "Impossible to create: " << path_negatives_folder << std::endl;
		}

		//For each views
		std::map<int, std::string>::iterator it = map_ids_names.begin();
		for (it; it != map_ids_names.end(); ++it)
		{
			const int id = it->first;
			std::vector<int> v_ids_overlapping_views = overlapper.findOverlappingViews(id);

			std::cout << "View: " << id << " has " << v_ids_overlapping_views.size() - 1 << " overlapping views." << std::endl;

			std::vector<std::pair<int, std::string>> v_overlapping_views;
			for (size_t i = 0; i < v_ids_overlapping_views.size(); ++i)
			{
				const std::string path_view = dataset_iterator->path().string() + "\\" + map_ids_names[v_ids_overlapping_views[i]];
				v_overlapping_views.push_back(std::pair<int, std::string>(v_ids_overlapping_views[i], path_view));
			}

			const std::string p_groundtruth = dataset_iterator->path().string() + "\\groundTruth.txt";

			//load overlapping views, 0 current view 1-size-1 overlapping
			std::vector<kpl::View<PointInT, NormalT, DescriptorT>> v_views = manager_view->loadViews(v_overlapping_views, p_groundtruth, ext_dataset);

			double cloud_resolution = kpl::computeCloudResolution<PointInT>(v_views[0].c_points_);
			std::cout << "View: " << id << " cloud resolution " << cloud_resolution << std::endl;

			pcl::PointCloud<PointOutT>::Ptr c_positive;
			pcl::PointCloud<PointOutT>::Ptr c_negative;
			if (v_ids_overlapping_views.size() >= 3)
			{
				////////////////////////////////////////////////////////////////
				//---------------- VISUALIZATION BEFORE LEARNING --------------
				///////////////////////////////////////////////////////////////
				if (!batch_mode)
				{
					if (show_overlap)
					{
						for (size_t i_v = 0; i_v < v_views.size(); ++i_v)
						{
							pcl::PointCloud<PointInT>::Ptr c_transformed(new pcl::PointCloud<PointInT>());
							pcl::transformPointCloud(*v_views[i_v].c_points_, *c_transformed, v_views[i_v].matrix_gt_);

							//Reference view
							if (i_v == 0)
							{
								pcl::visualization::PointCloudColorHandlerCustom<PointInT> red(c_transformed, 255, 255, 255);
								visualizer->addPointCloud<PointInT>(c_transformed, red, "Reference");
								visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.0, "Reference");
							}
							else
							{
								pcl::visualization::PointCloudColorHandlerRandom<PointInT> random_color(c_transformed);
								const std::string id = "cloud" + boost::lexical_cast<std::string>(i_v);

								visualizer->addPointCloud<PointInT>(c_transformed, random_color, id);
								visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, id);
							}
						}
					}
					else
					{
						pcl::visualization::PointCloudColorHandlerRandom<PointInT> random_color(v_views[0].c_points_);
						visualizer->addPointCloud<PointInT>(v_views[0].c_points_, random_color, "cloud");

						if (show_normals)
						{
							visualizer->addPointCloudNormals<PointInT, NormalT>(v_views[0].c_points_, v_views[0].c_normals_, 1, 5*cloud_resolution, "cloud_normals", 0);
						}

						if (show_supports)
						{
							const int idx_supp_normals = rand() % v_views[0].c_points_->points.size() - 1;
							const int idx_supp_descriptor = rand() % v_views[0].c_points_->points.size() - 1;

							visualizer->addSphere(v_views[0].c_points_->points[idx_supp_descriptor], rad_descriptor, 0, 0, 1, "support_descriptor", 0);
						}
						visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, "cloud");
					}

					visualizer->spinOnce();
					while (!visualizer->wasStopped())
					{
						visualizer->spinOnce(100);
					}
					visualizer->removeAllPointClouds();
					visualizer->removeAllShapes();
				}

				////////////////////////////////////////////////////////////////
				//----------------- COMPUTE DESCRIPTORS -----------------------
				///////////////////////////////////////////////////////////////			
				if(compute_descriptors)
				{
					std::cout << "START compute descriptors." << std::endl;
					manager_view->computeDescriptors(v_views, rad_descriptor);
					std::cout << "END compute descriptors." << std::endl;
				}
				else
				{
					std::cout << "START load descriptors." << std::endl;
					std::vector<std::pair<int, std::string>> id_file_name;

					std::vector<std::string> splitted_vector;
					boost::split(splitted_vector, dataset_iterator->path().string(), boost::is_any_of("\\"), boost::algorithm::token_compress_on);
					const std::string model_folder = splitted_vector[splitted_vector.size()-1];

					std::map<int, std::string>::iterator it = map_ids_names.begin();
					for (it; it != map_ids_names.end(); ++it)
					{
						std::pair<int, std::string> pair(it->first, p_descriptors + "\\" + model_folder + "\\" + it->second);
						id_file_name.push_back(pair);
					}

					manager_view->loadDescriptors(v_views, id_file_name);
					std::cout << "END load descriptors." << std::endl;
				}
			
				////////////////////////////////////////////////////////////////
				//-------------------- LEARN POSITIVE -------------------------
				///////////////////////////////////////////////////////////////
				std::vector<kpl::View<PointInT, NormalT, DescriptorT>> v_overlap;
				v_overlap.assign(v_views.begin()+1, v_views.end());

				std::cout << "START learnPositive" << std::endl;
				clock_t start = clock();

				c_positive.reset(new pcl::PointCloud<PointOutT>());
				ts_generator.generatePositives(v_views[0], v_overlap, th_distance, rad_nms, c_positive);

				const double elapsed = (clock() - start) /(double) CLOCKS_PER_SEC;
				std::cout << "END learnPositive. Elapsed time: "<< elapsed << std::endl;

				////////////////////////////////////////////////////////////////
				//-------------------- LEARN NEGATIVE -------------------------
				///////////////////////////////////////////////////////////////
				std::cout << "START learn negative" << std::endl;
	
				c_negative.reset(new pcl::PointCloud<PointOutT>());
				ts_generator.generateNegatives(v_views[0], c_positive, rad_negative, c_negative);

				std::cout << "END learnPositive. Elapsed time: " << elapsed << std::endl;

				////////////////////////////////////////////////////////////////
				//-------------------- SAVE------------------------------------
				////////////////////////////////////////////////////////////////
				if (c_positive->points.size() > 0 && c_negative->points.size())
				{
					const std::string f_name_positive = path_positives_folder + it->second;
					const std::string f_name_negative = path_negatives_folder + it->second;

					pcl::io::savePCDFileBinary(f_name_positive, *c_positive);
					pcl::io::savePCDFileBinary(f_name_negative, *c_negative);

					std::cout << f_name_positive + " " + boost::lexical_cast<std::string>(c_positive->points.size()) + " positives saved." << std::endl;
					std::cout << f_name_negative + " " + boost::lexical_cast<std::string>(c_negative->points.size()) + " negatives saved." << std::endl;
				}
			}
			else
			{
				std::cout << "Impossible to generate positive for view: " << id << " too few overlapping views." << std::endl;
			}

			////////////////////////////////////////////////////////////////
			//-------------------- VISUALIZATION ---------------------------
			////////////////////////////////////////////////////////////////
			if (!batch_mode)
			{

				pcl::visualization::PointCloudColorHandlerRandom<PointInT> random_color(v_views[0].c_points_);
				visualizer->addPointCloud<PointInT>(v_views[0].c_points_, random_color, "cloud");
				visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, "cloud");

				if(show_positives)
				{
					for (size_t i_pos = 0; i_pos < c_positive->points.size(); ++i_pos)
					{
						const std::string id = "positive" + boost::lexical_cast<std::string>(i_pos);
						visualizer->addSphere(c_positive->points[i_pos], 1 * cloud_resolution, 0, 255, 0, id, 0);
					}
				}

				if(show_negatives)
				{
					for (size_t i_neg = 0; i_neg < c_negative->points.size(); ++i_neg)
					{
						const std::string id = "negative" + boost::lexical_cast<std::string>(i_neg);
						visualizer->addSphere(c_negative->points[i_neg], 1 * cloud_resolution, 255, 0, 0, id, 0);
					}
				}
				
				visualizer->spinOnce();
				while (!visualizer->wasStopped())
				{
					visualizer->spinOnce(100);
				}
				visualizer->removeAllPointClouds();
				visualizer->removeAllShapes();
			}

		}//views

	}//models

	//Release
	manager_view->releaseInstance();

	return 0;
}