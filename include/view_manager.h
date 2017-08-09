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
#ifndef VIEW_MANAGER_H_
#define VIEW_MANAGER_H_

#include "view.h"

#include <pcl\filters\uniform_sampling.h>
#include <pcl\features\normal_3d_omp.h>
#include <pcl\features\shot_omp.h>

#include <unordered_map>

namespace kpl {

	/** \brief Simple implementation of View data structure manager implemented as singleton
	*
	* \author Riccardo Spezialetti
	*/

	template<typename PointInT, typename NormalT, typename HistogramT>
	class ViewManager
	{
		typedef typename pcl::PointCloud<PointInT> PointCloudIn;
		typedef typename PointCloudIn::Ptr PointCloudInPtr;
		typedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;

		typedef typename pcl::PointCloud<NormalT> PointCloudN;
		typedef typename PointCloudN::Ptr PointCloudNPtr;
		typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;

		typedef typename pcl::UniformSampling<PointInT>	UniformSampler;
		typedef typename UniformSampler::Ptr	UniformSamplerPtr;

		typedef typename pcl::UniformSampling<PointInT>	UniformSampler;
		typedef typename UniformSampler::Ptr	UniformSamplerPtr;

		typedef typename pcl::NormalEstimationOMP<PointInT, NormalT> NormalEstimator;
		typedef typename NormalEstimator::Ptr NormalEstimatorPtr;

		typedef typename View<PointInT, NormalT, HistogramT> ViewIn;

		public:

			/** /brief Singleton instance getter
			*/
			static ViewManager* getInstance();

			/** /brief Singleton instance release
			*/
			static void releaseInstance();

			ViewManager(ViewManager& const) = delete;

			void operator=(ViewManager const&) = delete;

			/** \brief Set to true to compute normals for loaded views
			* \param[in] estimate_normals	
			*/
			inline void estimateNormals(bool estimate_normals) { estimate_normals_ = estimate_normals; };

			/** \brief Compute normals with radius search
			* \param[in] normals_on_radius
			*/
			inline void setNormalsOnRadius(bool normals_on_radius) { normals_on_radius_ = normals_on_radius; };

			/** \brief Set radius for normals estimation
			* \param[in] r_normals	radius for normals estimation
			*/
			inline void setNormalsRadius(float r_normals) { rad_normals_ = r_normals; };

			/** \brief Set K-nearest neighbors for normals estimation
			* \param[in] nn_normals		K-nearest neighbors for normals estimation
			*/
			inline void setNormalsNN(int nn_normals) { nn_normals_ = nn_normals; };

			/** \brief Flip normals
			* \param[in] flip_normals 
			*/
			inline void setNormalsFlipping(bool flip_normals) { flip_normals_ = flip_normals; };

			/** \brief Set leaf for pointcloud subsampling
			* \param[in] leaf		leaf for subsampler
			*/
			inline void setLeaf(float leaf) { ss_leaf_ = leaf; };

			/** \brief Subsample loaded views
			* \param[in] sub_sampling	
			*/
			inline void setSubSampling(bool sub_sampling) { sub_sampling_ = sub_sampling; };
			
			/** /brief Load views specified in ids_names
			*	/param[in] ids_names		view ids and names to load
			*	/param[in] p_groundtruth	path to file with groundtruth matrices
			*	/param[in] extension		views extension
				returns a vector with loaded views
			*/
			std::vector<typename View<PointInT, NormalT, HistogramT>> 
			loadViews(const std::vector<std::pair<int, std::string>>& ids_names, const std::string p_groundtruth, const std::string extension);

			/** /brief Compute descriptors for views
			*	/param[out] views				views for which compute descriptors
			*	/param[in] radius_descriptor	radius for descriptor computation
			*/
			void computeDescriptors(std::vector<typename View<PointInT, NormalT, HistogramT>>& views, const float radius_descriptor);

			/** /brief Load descriptors for views
			*	/param[out] views				views for which load descriptors
			*	/param[in]	ids_names			view identifier and descriptor file name
			*/
			void loadDescriptors(std::vector<typename View<PointInT, NormalT, HistogramT>>& views, const std::vector<std::pair<int, std::string>>& ids_names);

		private: 
			
			ViewManager();

			void computeDescriptorsPerView(View<PointInT, NormalT, HistogramT>& view, const float radius_descriptor);

			/* View Manager as singleton*/
			static ViewManager* instance_;
		
			/** /brief Vector of views */
			std::vector<View<PointInT, NormalT, HistogramT>> views_;

			std::vector<int> ids_loaded_views_;

			/** /brief Subsampler*/
			UniformSamplerPtr uniform_sampler_;

			/** /brief Normal estimator*/
			NormalEstimatorPtr normal_estimator_;

			/** /brief Compute normals on radius*/
			bool normals_on_radius_;

			/** /brief Flip normals*/
			bool flip_normals_;

			/** /brief Subsample pointclouds*/
			bool sub_sampling_;

			/** /brief Estimate normals*/
			bool estimate_normals_;

			/** /brief Normals radius for normals estimation*/
			float rad_normals_;

			/** /brief K-nearest neighbors for normals estimation*/
			int nn_normals_;

			/** /brief Subsampler leaf*/
			float ss_leaf_;
	};
}

#include "impl\view_manager.hpp"

#endif // !VIEW_MANAGER_H_
