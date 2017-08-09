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
#ifndef VIEW_MANAGER_IMPL_
#define VIEW_MANAGER_IMPL_

#include <pcl\io\ply_io.h>
#include <pcl\features\shot_omp.h>
#include <pcl\filters\filter.h>

#include <omp.h>

#include "point_cloud_utilities.h"
#include "view_manager.h"

//Singleton instance
template<typename PointT, typename NormalT, typename HistogramT>
kpl::ViewManager<PointT, NormalT, HistogramT>* kpl::ViewManager<PointT, NormalT, HistogramT>::instance_ = 0;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename NormalT, typename HistogramT>
kpl::ViewManager<PointT, NormalT, HistogramT>::ViewManager()
{
	 normals_on_radius_ = true;
	 flip_normals_ = false;
	 rad_normals_ = 0.0;
	 nn_normals_ = 10;
	 estimate_normals_ = true;

	 sub_sampling_ = true;
	 ss_leaf_ = 0.0;

	views_.resize(0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename NormalT, typename HistogramT>
kpl::ViewManager<PointT, NormalT, HistogramT>*
kpl::ViewManager<PointT, NormalT, HistogramT>::getInstance()
{
	if (instance_ == 0)
	{
		instance_ = new ViewManager();
	}

	return instance_;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename NormalT, typename HistogramT>void
kpl::ViewManager<PointT, NormalT, HistogramT>::releaseInstance()
{
	delete instance_;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename NormalT, typename HistogramT>std::vector<typename kpl::View<PointT, NormalT, HistogramT>>
kpl::ViewManager<PointT, NormalT, HistogramT>::loadViews(const std::vector<std::pair<int, std::string>>& ids_names, const std::string p_groundtruth, const std::string ext)
{
		//Check on components 
		if (sub_sampling_ && uniform_sampler_ == 0)
			uniform_sampler_.reset(new UniformSampler());

		if (estimate_normals_ && normal_estimator_ == 0)
		{
			normal_estimator_.reset(new NormalEstimator());

			pcl::search::KdTree<PointInT>::Ptr kdtree(new pcl::search::KdTree<PointInT>());
			normal_estimator_->setNumberOfThreads(omp_get_num_threads());
			normal_estimator_->setSearchMethod(kdtree);
		}

		std::vector<kpl::View<PointInT, NormalT, DescriptorT>> views;
		views.resize(ids_names.size());

		unsigned n_views = 0;
		for (size_t i_v=0; i_v < ids_names.size(); ++i_v)
		{
			ViewIn view;

			const int id = ids_names[i_v].first;
			bool found = false;

			for (size_t i_lw = 0; i_lw < views_.size(); ++i_lw)
			{
				if (views_[i_lw].id_ == id)
				{
					view.id_ = views_[i_lw].id_;
					view.c_points_ = views_[i_lw].c_points_;
					view.c_normals_ = views_[i_lw].c_normals_;
					view.c_descriptors_ = views_[i_lw].c_descriptors_;
					view.matrix_gt_ = views_[i_lw].matrix_gt_;

					found = true;
					break;
				}
			}

			if (!found)
			{
				//Read data
				pcl::PointCloud<PointInT>::Ptr c_points(new pcl::PointCloud<PointInT>);
				pcl::PointCloud<NormalT>::Ptr c_normals(new pcl::PointCloud<NormalT>);

				const std::string path = ids_names[i_v].second;

				if (ext == ".pcd")
				{
					if (pcl::io::loadPCDFile<PointInT>(path, *c_points) == -1)
					{
						std::cerr << "Impossible to read point cloud from: " << path << std::endl;
					}
				}
				else if (ext == ".ply")
				{
					if (pcl::io::loadPLYFile(path, *c_points) == -1)
					{
						std::cerr << "Impossible to read point cloud from: " << path << std::endl;
					}
				}

				//Groundtruth Matrix
				Eigen::Matrix4f gt = computeGroundTruth(p_groundtruth, id);

				//Filter for NAN
				c_points->is_dense = false;

				std::vector<int> indices;
				pcl::removeNaNFromPointCloud(*c_points, *c_points, indices);

				//Uniform sampling
				if (sub_sampling_)
				{
					uniform_sampler_->setInputCloud(c_points);
					uniform_sampler_->setRadiusSearch(ss_leaf_);
					uniform_sampler_->filter(*c_points);
				}

				//Normal estimation
				if (estimate_normals_)
				{
					normal_estimator_->setInputCloud(c_points);

					if (normals_on_radius_)
						normal_estimator_->setRadiusSearch(rad_normals_);
					else
						normal_estimator_->setKSearch(nn_normals_);

					normal_estimator_->setNumberOfThreads(omp_get_max_threads());
					normal_estimator_->compute(*c_normals);
				}

				if (flip_normals_)
				{
					//Sensor origin is inside the model, flip normals to make normal sign coherent with scenes
					std::transform(c_normals->points.begin(), c_normals->points.end(), c_normals->points.begin(), [](pcl::Normal p) -> pcl::Normal {auto q = p; q.normal[0] *= -1; q.normal[1] *= -1; q.normal[2] *= -1; return q; });
				}
	
				view.id_ = id;
				view.c_points_ = c_points;
				view.c_normals_ = c_normals;
				view.matrix_gt_ = gt;
			}

			//Save in view
			views[n_views] = view;
			++n_views;
		}
		views_ = views;

 return views_;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename NormalT, typename HistogramT>void
kpl::ViewManager<PointT, NormalT, HistogramT>::loadDescriptors(std::vector<typename kpl::View<PointT, NormalT, HistogramT>>& views, const std::vector<std::pair<int, std::string>>& ids_names)
{
	for(size_t i_v=0;i_v < views.size();++i_v)
	{		
		for (size_t i_p = 0; i_p < ids_names.size(); ++i_p)
		{
			if (views[i_v].id_ == ids_names[i_p].first)
			{
				pcl::PointCloud<HistogramT>::Ptr c_histrogram(new pcl::PointCloud<HistogramT>());

				//Load descriptors
				if (pcl::io::loadPCDFile<HistogramT>(ids_names[i_p].second, *c_histrogram) == -1)
				{
					std::cerr << "Impossible to load descriptors: " << ids_names[i_p].second << std::endl;
					return;
				}

				//Nan removing
				views[i_v].c_descriptors_.reset(new pcl::PointCloud<HistogramT>());
				removeNaNFromHistogram(*c_histrogram, *views[i_v].c_descriptors_, views[i_v].map_point_desc_);

				break;
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*Customizable function to compute descriptors, in this case we use SHOT as descriptor.*/
template<typename PointT, typename NormalT, typename HistogramT>void
kpl::ViewManager<PointT, NormalT, HistogramT>::computeDescriptorsPerView(ViewIn& view, const float radius_descriptor)
{
	pcl::SHOTEstimationOMP<PointInT, NormalT, pcl::SHOT352> shot_estimator;
	shot_estimator.setNumberOfThreads(omp_get_max_threads());
	shot_estimator.setRadiusSearch(radius_descriptor);
	shot_estimator.setLRFRadius(radius_descriptor);
	shot_estimator.setInputCloud(view.c_points_);
	shot_estimator.setInputNormals(view.c_normals_);

	pcl::PointCloud<pcl::SHOT352>::Ptr c_shot_descriptors(new pcl::PointCloud<pcl::SHOT352>());
	shot_estimator.compute(*c_shot_descriptors);

	pcl::PointCloud<HistogramT>::Ptr c_histograms(new pcl::PointCloud<HistogramT>());
	c_histograms->resize(c_shot_descriptors->points.size());
	c_histograms->is_dense = c_shot_descriptors->is_dense;

	for (size_t i_desc = 0; i_desc < c_shot_descriptors->points.size(); ++i_desc)
	{	
		memcpy(c_histograms->points[i_desc].histogram, c_shot_descriptors->points[i_desc].descriptor, c_shot_descriptors->points[i_desc].descriptorSize() * sizeof(float));
	}

	//Nan removing
	pcl::PointCloud<HistogramT>::Ptr c_histograms_filtered(new pcl::PointCloud<HistogramT>());
	std::vector<int> indices;
	removeNaNFromHistogram<HistogramT>(*c_histograms, *c_histograms_filtered, indices);

	view.c_descriptors_ = c_histograms_filtered;
	view.map_point_desc_ = indices;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT, typename NormalT, typename HistogramT>void
kpl::ViewManager<PointT, NormalT, HistogramT>::computeDescriptors(std::vector<typename kpl::View<PointT, NormalT, HistogramT>>& views, const float radius_descriptor)
{
#ifdef MULTITHREAD
	std::vector<boost::thread*> threads;
	for(size_t i_v=0; i_v < views.size(); ++i_v)
	{
		threads.push_back(new boost::thread(&ViewManager::computeDescriptorsPerView,this,boost::ref(views[i_v]),radius_descriptor));
	}

	for (size_t i_th = 0; i_th < threads.size(); ++i_th)
	{
		threads[i_th]->join();
		delete threads[i_th];
	}
#else
	for (size_t i_v=0; i_v< views.size(); ++i_v)
	{
		computeDescriptorsPerView(views[i_v], radius_descriptor);
	}
#endif

}
#endif // !VIEW_MANAGER_IMPL_