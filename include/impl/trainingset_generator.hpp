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

#ifndef TRAININGSET_GENERATOR_IMPL_
#define TRAININGSET_GENERATOR_IMPL_

#include <pcl\common\common.h>
#include <pcl\common\distances.h>
#include <pcl\common\io.h>
#include <pcl\point_cloud.h>
#include <pcl\point_types.h>
#include <pcl\io\pcd_io.h>
#include <pcl\kdtree\impl\kdtree_flann.hpp>
#include <pcl\registration\distances.h>

#include "trainingset_generator.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct distances_sort {

	inline bool operator () (const pcl::Correspondence& correspondence1, const pcl::Correspondence& correspondence2)
	{
		return (correspondence1.distance < correspondence2.distance);
	}
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename NormalT, typename PointOutT, typename HistogramT>
kpl::TrainingSetGenerator<PointInT, NormalT, PointOutT, HistogramT>::TrainingSetGenerator()
{

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename NormalT, typename PointOutT, typename HistogramT>int 
kpl::TrainingSetGenerator<PointInT, NormalT, PointOutT, HistogramT>::refineOnView(ViewIn& view, ViewIn& view_for_refinement, const std::vector<int>& indices_positive, const float r_tol, pcl::CorrespondencesPtr correspondences)
{
	correspondences->resize(indices_positive.size());
	unsigned number_of_corr = 0;
	unsigned points_intersection = 0;

	pcl::PointCloud<PointInT>::Ptr c_src_in_rf(new pcl::PointCloud<PointInT>());
	pcl::PointCloud<PointInT>::Ptr c_trg_in_rf(new pcl::PointCloud<PointInT>());

	pcl::transformPointCloud(*view.c_points_, *c_src_in_rf, view.matrix_gt_);
	pcl::transformPointCloud(*view_for_refinement.c_points_, *c_trg_in_rf, view_for_refinement.matrix_gt_);

	pcl::KdTreeFLANN<PointInT> kdtree_trg;
	kdtree_trg.setInputCloud(c_trg_in_rf);

	pcl::KdTreeFLANN<HistogramT> kdtree_trg_descriptors;
	kdtree_trg_descriptors.setInputCloud(view_for_refinement.c_descriptors_);

	for (size_t i_p = 0; i_p < indices_positive.size(); ++i_p)
	{
		const int idx_positive = indices_positive[i_p];
		//intersection
		std::vector<int> indices;
		std::vector<float> distances;
		kdtree_trg.nearestKSearch(c_src_in_rf->points[idx_positive], 1, indices, distances);
	
		PointInT& point_to_refine = c_trg_in_rf->points[indices[0]];

		if (sqrt(distances[0]) < r_tol)
		{
			points_intersection++;
			int idx_hist = idx_positive;
			for (size_t i=0; i< view.map_point_desc_.size(); ++i)
			{
				if (view.map_point_desc_[i] == idx_positive)
				{
					idx_hist = i;
					break;
				}
			}
			//Find nn shot
			HistogramT point_histogram = view.c_descriptors_->points[idx_hist];

			indices.clear();
			distances.clear();
			kdtree_trg_descriptors.nearestKSearch(point_histogram, 1, indices, distances);
			const int idx_point_matched_desc = view_for_refinement.map_point_desc_[indices[0]];

			const float distance = pcl::euclideanDistance(point_to_refine, c_trg_in_rf->points[idx_point_matched_desc]);
			
			if (distance < r_tol)
			{
				pcl::Correspondence corr(idx_positive, idx_point_matched_desc, distances[0]);
				correspondences->at(number_of_corr) = corr;
				number_of_corr++;
			}
		}		
		
	}
	correspondences->resize(number_of_corr);
	
	std::cout << view.id_ << " - " << view_for_refinement.id_ <<" Indices positive:" << indices_positive.size() << " Intersection: " << points_intersection << " Refined: " << number_of_corr << std::endl;

	return static_cast<uint32_t>(correspondences->size());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename NormalT, typename PointOutT, typename HistogramT>int
kpl::TrainingSetGenerator<PointInT, NormalT, PointOutT, HistogramT>::keepUniqueCorrespondences(const std::vector<pcl::CorrespondencesPtr>correspondences_to_filter, pcl::CorrespondencesPtr correspondences_filtered)
{
	int max_id = 0;

	std::vector<pcl::Correspondence> all_correspondences;
	//All in one vector
	for (size_t i_c = 0; i_c < correspondences_to_filter.size(); ++i_c)
	{
		for (size_t j = 0; j < correspondences_to_filter[i_c]->size(); ++j)
		{
			const int index_query = correspondences_to_filter[i_c]->at(j).index_query;
			all_correspondences.push_back(correspondences_to_filter[i_c]->at(j));

			if (index_query > max_id)
				max_id = index_query;
		}
	}
	//Take the correspondences with less distance
	std::sort(all_correspondences.begin(), all_correspondences.end(), distances_sort());
	std::vector<bool> is_present(max_id + 1, false);

	//Keep unique
	for (size_t i_c = 0; i_c < all_correspondences.size(); ++i_c)
	{
		const int idx_query = all_correspondences[i_c].index_query;
		if (!is_present[idx_query])
		{
			correspondences_filtered->push_back(all_correspondences[i_c]);
			is_present[idx_query] = true;
		}
	}
	return static_cast<uint32_t>(correspondences_filtered->size());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename NormalT, typename PointOutT, typename HistogramT>int
kpl::TrainingSetGenerator<PointInT, NormalT, PointOutT, HistogramT>::generatePositives(ViewIn& view, std::vector<ViewIn>& overlapping_views, const float th_EclideanDistance, const float r_nms, PointCloudOutPtr cloud_positive)
{
	std::vector<pcl::CorrespondencesPtr> corr_first_level(overlapping_views.size());
	
	
	for (size_t i=0; i < overlapping_views.size(); i++)
	{

		ViewIn view_learn = overlapping_views[i];
		/*Learn positive with lear view*/
		//Correspondences between descriptors
		std::cout << view.id_ << " on " << view_learn.id_ << std::endl;

		std::vector<pcl::Correspondence> v_sorted_corr = computeSortedCorrespondecesOnDescriptor(view.c_descriptors_, view.map_point_desc_, view_learn.c_descriptors_, view_learn.map_point_desc_);

		//Check on L2 distance
		std::vector<int> v_indices_positive;
		checkCorrespondencesOnL2(v_sorted_corr, view, view_learn, v_indices_positive, th_EclideanDistance, r_nms);

		std::cout << "Learned: " << v_indices_positive.size() << std::endl;
	
		pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());
		corr_first_level[i] = correspondences;

		/*Refine learned positives with others views*/
		std::vector<pcl::CorrespondencesPtr> correspondences_per_refinement(overlapping_views.size() - 1);
		unsigned idx = 0;
#ifdef MULTIVIEW
		std::vector<boost::thread*> threads;
#endif
		for(size_t j=0; j < overlapping_views.size(); j++)
		{
			if(view_learn.id_ != overlapping_views[j].id_)
			{
#ifdef MULTIVIEW
				//Allocate
				pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());
				correspondences_per_refinement[idx] = correspondences;
				//Start thread
				threads.push_back((new boost::thread(&TrainingSetGenerator::refineOnView, this, view, boost::ref(overlapping_views[j]), v_indices_positive, th_EclideanDistance, boost::ref(correspondences_per_refinement[idx]))));
				idx++;
#else
				pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());
				correspondences_per_refinement[idx] = correspondences;

				refineOnView(view, overlapping_views[j], v_indices_positive, th_EclideanDistance, correspondences_per_refinement[idx]);
				idx++;
#endif // MULTIVIEW
			}
		}
#ifdef MULTIVIEW
		for (size_t i_th=0; i_th < threads.size(); ++i_th)
		{
			threads[i_th]->join();
			delete threads[i_th];
		}
#endif
		//Sort and keep unique between single filter on view
		keepUniqueCorrespondences(correspondences_per_refinement, corr_first_level[i]);
		assert(v_indices_positive.size() >= corr_first_level[i]->size());
	}
	//Sort and keep unique correspondence
	pcl::CorrespondencesPtr corr_positive(new pcl::Correspondences());
	keepUniqueCorrespondences(corr_first_level, corr_positive);

	cloud_positive->points.resize(corr_positive->size());
	cloud_positive->height = 1;
	cloud_positive->width = static_cast<uint32_t>(corr_positive->size());
	cloud_positive->is_dense = true;

	for (size_t i_corr=0; i_corr < corr_positive->size(); ++i_corr )
	{
		const int index = corr_positive->at(i_corr).index_query;
		pcl::copyPoint(view.c_points_->points[index], cloud_positive->points[i_corr]);
		
		//Set intensity between 0 and 1
		cloud_positive->points[i_corr].intensity = corr_positive->at(i_corr).distance;
	}
	return cloud_positive->width;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename NormalT, typename PointOutT, typename HistogramT>int
kpl::TrainingSetGenerator<PointInT, NormalT, PointOutT, HistogramT>::generateNegatives(ViewIn& view, const PointCloudOutConstPtr cloud_positive, const float radius_negative, PointCloudOutPtr cloud_negative)
{
	std::vector<int> v_negative_indices;

	std::vector<bool> available(view.c_points_->points.size(), true);

	pcl::search::KdTree<PointInT> kdtree;
	kdtree.setInputCloud(view.c_points_);

	std::vector<int> kdt_indices;
	std::vector<float> kdt_distances;

	for (size_t i_p = 0; i_p < cloud_positive->points.size(); ++i_p)
	{
		kdt_indices.clear();
		kdt_distances.clear();

		PointInT point_to_search;
		pcl::copyPoint<PointOutT, PointInT>(cloud_positive->points[i_p], point_to_search);

		if ((kdtree.radiusSearch(point_to_search, radius_negative, kdt_indices, kdt_distances)) > 0)
		{
			for (size_t i_nn = 0; i_nn < kdt_indices.size(); ++i_nn)
				available[kdt_indices[i_nn]] = false;
		}
	}

	int neg_count = 0;
	int count = 1;
	srand(time(NULL));
	const int guess_number = (static_cast<int>(view.c_points_->points.size())) / 100;

	while ((std::find(available.begin(), available.end(), true) != available.end()) && neg_count < cloud_positive->points.size())
	{
		int neighboor_found = 0;
		int random_indx;
		
		if (count%guess_number != 0)
			random_indx = rand() % view.c_points_->points.size();
		else 
		{
			for (size_t k = 0; k < available.size(); k++)
			{
				if (available[k])
				{
					random_indx = k;
					break;
				}
			}
		}

		if (available[random_indx] == true)
		{
			v_negative_indices.push_back(random_indx);
			neg_count++;
			
			kdt_indices.clear();
			kdt_distances.clear();

			neighboor_found = (kdtree.radiusSearch(view.c_points_->points[random_indx], radius_negative, kdt_indices, kdt_distances));

			if (neighboor_found > 0)
			{
				for (size_t i_nn = 0; i_nn < kdt_indices.size(); ++i_nn)
				{
					available[kdt_indices[i_nn]] = false;
				}
			}
		}
		count++;
	}

	pcl::copyPointCloud<PointInT, PointOutT>(*view.c_points_, v_negative_indices, *cloud_negative);

	return static_cast<uint32_t>(v_negative_indices.size());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename NormalT, typename PointOutT, typename HistogramT>void
kpl::TrainingSetGenerator<PointInT, NormalT, PointOutT, HistogramT>::checkCorrespondencesOnL2(std::vector<pcl::Correspondence>& correspondences, const ViewIn& view_src, const ViewIn& view_trg, std::vector<int>& indices, const float th_EclideanDistance, const float nms_radius) {

	std::vector<int> true_positive;
	std::vector<int> false_positive;

	pcl::PointCloud<PointInT> c_src_transformed, c_trg_transformed;

	pcl::transformPointCloud(*view_src.c_points_, c_src_transformed, view_src.matrix_gt_);
	pcl::transformPointCloud(*view_trg.c_points_, c_trg_transformed, view_trg.matrix_gt_);

	for (size_t i_corr = 0; i_corr < correspondences.size(); i_corr++)
	{
		const int idx_query = correspondences[i_corr].index_query;
		const int idx_match = correspondences[i_corr].index_match;

		const float distance = pcl::euclideanDistance(c_src_transformed.points[idx_query], c_trg_transformed.points[idx_match]);

		if (distance < th_EclideanDistance)
		{
			true_positive.push_back(correspondences[i_corr].index_query);
		}
		else
			false_positive.push_back(correspondences[i_corr].index_query);
	}

	//Keep only maxima
	nonMaximaSuppression<PointInT>(view_src.c_points_, true_positive, nms_radius, indices);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename NormalT, typename PointOutT, typename HistogramT>void 
kpl::TrainingSetGenerator<PointInT, NormalT, PointOutT, HistogramT>::singleThreadFillCorrespondences(PointCloudHistConstPtr cloud_src, pcl::KdTreeFLANN<HistogramT>& kdt_trg, const int start, const int end, const int k, const std::vector<int>& indices_src, const std::vector<int>& indices_trg, std::vector<pcl::Correspondence>& correspondences)
{
	std::vector<int> indices;
	std::vector<float> distances;

	for (unsigned i_point = start; i_point < end; ++i_point)
	{
		kdt_trg.nearestKSearch(cloud_src->points[i_point], 1, indices, distances);

		pcl::Correspondence corr(indices_src[i_point], indices_trg[indices[0]], sqrt(distances[0]));
		correspondences[i_point] = corr;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename NormalT, typename PointOutT, typename HistogramT> std::vector<pcl::Correspondence>
kpl::TrainingSetGenerator<PointInT, NormalT, PointOutT, HistogramT>::computeSortedCorrespondecesOnDescriptor(PointCloudHistConstPtr cloud_src, std::vector<int>& indices_src, PointCloudHistConstPtr cloud_trg, std::vector<int>& indices_trg)
{
	std::vector<pcl::Correspondence> sorted_correspondences;
	sorted_correspondences.resize(cloud_src->points.size());
	//Kdtree on trg
	pcl::KdTreeFLANN<HistogramT> kdt_trg;
	kdt_trg.setInputCloud(cloud_trg);

	const int k = 1;
#ifdef MULTITHREAD
	unsigned n_threads = static_cast<unsigned>(NUMBER_OF_THREADS);
	unsigned n_points = static_cast<unsigned>(cloud_src->points.size());

	int points_per_threads = n_points / NUMBER_OF_THREADS;
	int remaining = n_points % n_threads;

	std::vector<boost::thread*> threads(n_threads);

	for (int i_th = 0; i_th < n_threads; ++i_th)
	{
		int start = (points_per_threads*i_th);
		int end = (i_th == n_threads-1) ? n_points : (points_per_threads*(i_th + 1));

		threads[i_th] = (new boost::thread(&TrainingSetGenerator::singleThreadFillCorrespondences, this, cloud_src, kdt_trg, start, end, k, indices_src, indices_trg, boost::ref(sorted_correspondences)));
	}

	for (int i_th = 0; i_th < n_threads; ++i_th)
	{
		threads[i_th]->join();
		delete threads[i_th];	
	}

#else
	for (size_t i_src = 0; i_src < cloud_src->points.size(); ++i_src)
	{
		std::vector<int> k_index;
		std::vector<float> k_distances;

		if (kdt_trg.nearestKSearch(cloud_src->points[i_src], k, k_index, k_distances) > 0)
		{
			pcl::Correspondence corr(indices_src[i_src], indices_trg[k_index[0]], sqrt(k_distances[0]));
			sorted_correspondences[i_src] = corr;
		}
	}	
#endif
	//sort vector
	std::sort(sorted_correspondences.begin(), sorted_correspondences.end(), distances_sort());
	return 	sorted_correspondences;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename NormalT, typename PointOutT, typename HistogramT>
kpl::TrainingSetGenerator<PointInT, NormalT, PointOutT, HistogramT>::~TrainingSetGenerator()
{

}
#endif //TRAININGSET_GENERATOR_IMPL_
