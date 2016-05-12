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

#ifndef PCL_KEYPOINTS_IMPL_KEYPOINT_LEARNING_H_
#define PCL_KEYPOINTS_IMPL_KEYPOINT_LEARNING_H_

#include "KeypointLearning.h"
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>

void findAnnulusPair(int n_annulus, float radius, float distance, int& annulus_index, int& annulus_index_pair, float& annulus_weight);
void findBinPair(int n_bins, float cosine, int& bin_index, int& bin_index_pair, float& bin_weight);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::keypoints::KeypointLearningDetector<PointInT, PointOutT, NormalT>::setInputCloud(const PointCloudInConstPtr &cloud)
{
	if (this->normals_ && this->input_ && (cloud != this->input_))
	{
		this->normals_.reset();
	}
	this->input_ = cloud;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::keypoints::KeypointLearningDetector<PointInT, PointOutT, NormalT>::setNonMaxima(bool non_maxima)
{
	this->non_maxima_ = non_maxima;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::keypoints::KeypointLearningDetector<PointInT, PointOutT, NormalT>::setNonMaximaDrawsRemove(bool non_maxima_draws_remove)
{
	this->non_maxima_draws_remove_ = non_maxima_draws_remove;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::keypoints::KeypointLearningDetector<PointInT, PointOutT, NormalT>::setNonMaximaDrawsThreshold(float non_maxima_draws_threshold)
{
	this->non_maxima_draws_threshold_ = non_maxima_draws_threshold;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::keypoints::KeypointLearningDetector<PointInT, PointOutT, NormalT>::setNormals(const PointCloudNConstPtr &normals)
{
	this->normals_ = normals;
}
//////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename PointOutT, typename NormalT> void
pcl::keypoints::KeypointLearningDetector<PointInT, PointOutT, NormalT>::setPredictionThreshold(double th)
{
	this->prediction_th_ = th;
}

//////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename PointOutT, typename NormalT> void
pcl::keypoints::KeypointLearningDetector<PointInT, PointOutT, NormalT>::setNonMaxRadius(double non_maxima_radius)
{
	this->non_maxima_radius_ = non_maxima_radius;
}

//////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename PointOutT, typename NormalT> void
pcl::keypoints::KeypointLearningDetector<PointInT, PointOutT, NormalT>::setNAnnulus(int n_annulus)
{
	this->n_annulus_ = n_annulus;
}

//////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename PointOutT, typename NormalT> void
pcl::keypoints::KeypointLearningDetector<PointInT, PointOutT, NormalT>::setNBins(int n_bins)
{
	this->n_bins_ = n_bins;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> bool
pcl::keypoints::KeypointLearningDetector<PointInT, PointOutT, NormalT>::initCompute()
{
	if (!Keypoint<PointInT, PointOutT>::initCompute())
	{
		PCL_ERROR("[pcl::%s::initCompute] init failed!\n", this->name_.c_str());
		return (false);
	}

	if (!this->normals_)
	{
		std::cout << "Computing normals for KPL" << std::endl;
		PointCloudNPtr normals(new PointCloudN());
		normals->reserve(normals->size());
		if (!this->surface_->isOrganized())
		{
			pcl::NormalEstimation<PointInT, NormalT> normal_estimation;
			normal_estimation.setInputCloud(this->surface_);
			//normal_estimation.setKSearch(10);
			normal_estimation.setRadiusSearch(this->search_radius_);
			normal_estimation.compute(*normals);
		}
		else
		{
			IntegralImageNormalEstimation<PointInT, NormalT> normal_estimation;
			normal_estimation.setNormalEstimationMethod(pcl::IntegralImageNormalEstimation<PointInT, NormalT>::SIMPLE_3D_GRADIENT);
			normal_estimation.setInputCloud(this->surface_);
			normal_estimation.setNormalSmoothingSize(5.0);
			normal_estimation.compute(*normals);

		}

		this->normals_ = normals;
	}
	if (this->normals_->size() != this->surface_->size())
	{
		PCL_ERROR("[pcl::%s::initCompute] normals given, but the number of normals does not match the number of input points!\n", this->name_.c_str()/*, method_*/);
		return (false);
	}

	return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename PointOutT, typename NormalT> bool
pcl::keypoints::KeypointLearningDetector<PointInT, PointOutT, NormalT>::loadForest(const std::string & path)
{
	this->forest_.load(path.c_str(), 0);
	if (this->forest_.get_tree_count() == 0)
		std::cerr << "[PROBLEM LOADING FOREST FOR SHOTKPL]: " << path << std::endl;
	return (this->forest_.get_tree_count() != 0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::keypoints::KeypointLearningDetector<PointInT, PointOutT, NormalT>::detectKeypoints(PointCloudOut &output)
{
	boost::shared_ptr<pcl::PointCloud<PointOutT> > response(new pcl::PointCloud<PointOutT>());

	response->points.reserve(this->input_->points.size());

	//Check if normals are provided, else compute it
	runForest(*response);

	if (!this->non_maxima_)
	{
		output = *response;
		// we do not change the denseness in this case
		output.is_dense = this->input_->is_dense;
		for (size_t i = 0; i < response->size(); ++i)
			this->keypoints_indices_->indices.push_back(static_cast<int> (i));
	}
	else
	{
		output.points.clear();
		output.points.reserve(response->points.size());
		std::cout << "Radius non maxima suprresion KPL: " << this->non_maxima_radius_ << std::endl;
		std::vector<int> skipList;

		for (int idx = 0; idx < static_cast<int> (response->points.size()); ++idx)
		{
			if (!isFinite(response->points[idx]) ||
				!pcl_isfinite(response->points[idx].intensity) ||
				response->points[idx].intensity < this->prediction_th_/*threshold_*/)
				continue;

			std::vector<int> nn_indices;
			std::vector<float> nn_dists;

			this->tree_->radiusSearch(idx, this->non_maxima_radius_, nn_indices, nn_dists);
			bool is_maxima = true;
			bool has_draw = false;
			std::vector<int> draws;
			for (std::vector<int>::const_iterator iIt = nn_indices.begin(); iIt != nn_indices.end(); ++iIt)
			{
				if (response->points[idx].intensity < response->points[*iIt].intensity)
				{
					is_maxima = false;
					break;
				}
				else if (response->points[idx].intensity == response->points[*iIt].intensity){
					if (idx != *iIt){
						has_draw = true;
						draws.push_back(*iIt);
					}
				}
			}
			if (is_maxima)
			{
				if (this->non_maxima_draws_remove_ && has_draw){
					if (std::find(skipList.begin(), skipList.end(), idx) == skipList.end()){
						PointInT pointIn = this->input_->points[idx];
						bool survive = false;
						for (int i = 0; i < draws.size(); i++){
							PointInT drawPoint = this->input_->points[draws[i]];
							float distance = (pointIn.getVector3fMap() - drawPoint.getVector3fMap()).norm();
							if (distance < non_maxima_draws_threshold_){
								survive = true;
								skipList.push_back(draws[i]);
							}
						}
						if (survive){
							output.points.push_back(response->points[idx]);
							this->keypoints_indices_->indices.push_back(idx);
						}
					}
				}
				else {
					output.points.push_back(response->points[idx]);
					this->keypoints_indices_->indices.push_back(idx);
				}
			}
		}

		output.height = 1;
		output.width = static_cast<uint32_t> (output.points.size());
		output.is_dense = true;
	}

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::keypoints::KeypointLearningDetector<PointInT, PointOutT, NormalT>::runForest(PointCloudOut &output) const
{
	//  PCL_ALIGN (16) float covar [8];
	//output.reserve (input_->size ());
#ifdef _OPENMP
#pragma omp parallel for shared (output) private (covar) num_threads(threads_)
#endif
	std::cout << "Feature radius for KPL: " << this->search_parameter_ << std::endl;
	for (int pIdx = 0; pIdx < static_cast<int> (this->input_->size()); ++pIdx)
	{
		const PointInT& pointIn = this->input_->points[pIdx];
		//    output [pIdx].intensity = 0.0; //std::numeric_limits<float>::quiet_NaN ();
		if (isFinite(pointIn) && isFinite(this->normals_->points[pIdx]))
		{

			cv::Mat feat = computePointFeatures(pIdx);
			float prob = 1 - (this->forest_.predict_prob(feat, cv::Mat()));
			feat.release();
			
			PointOutT point;
			point.x = pointIn.x;
			point.y = pointIn.y;
			point.z = pointIn.z;
			point.intensity = prob;
			output.push_back(point);

		}

	}
	output.height = 1;
	output.width = static_cast<uint32_t> (output.points.size());
	output.is_dense = true;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> cv::Mat
pcl::keypoints::KeypointLearningDetector<PointInT, PointOutT, NormalT>::computePointFeatures(int point_index) const
{

	cv::Mat features_row(1, n_annulus_ * n_bins_, CV_32F);
	Eigen::MatrixXf histograms = Eigen::MatrixXf::Zero(n_annulus_, n_bins_);
	std::vector<int> indices;
	std::vector<float> distances;
	std::vector<int> annulus(this->n_annulus_);
	std::vector<float> bins(this->n_bins_);
	//std::cout<< "Calcolo features per: " << point_index << std::endl;
	Eigen::Vector3f point_vector = this->normals_->points[point_index].getNormalVector3fMap();

	if (this->searchForNeighbors(point_index, this->search_parameter_, indices, distances) > 0)
	{
		for (int neigh_indx = 1; neigh_indx < indices.size(); neigh_indx++)
		{
			if (pcl::isFinite<pcl::Normal>(this->normals_->points[indices[neigh_indx]]))
			{

				Eigen::Vector3f normal_vector = this->normals_->points[indices[neigh_indx]].getNormalVector3fMap();
				float cosine = 1 - point_vector.dot(normal_vector);
				int annulus_index, annulus_pair;
				float annulus_weight;
				findAnnulusPair(this->n_annulus_, sqrt(distances[neigh_indx]), this->search_radius_, annulus_index, annulus_pair, annulus_weight);
				int bin_index, bin_pair;
				float bin_weight;
				findBinPair(this->n_bins_, cosine, bin_index, bin_pair, bin_weight);

				histograms(annulus_index, bin_index) += ((1 - bin_weight)*(1 - annulus_weight));
				histograms(annulus_index, bin_pair) += ((bin_weight)*(1 - annulus_weight));
				//aggiorno histogram del annulus pair
				histograms(annulus_pair, bin_index) += ((1 - bin_weight)*(annulus_weight));
				histograms(annulus_pair, bin_pair) += ((bin_weight)*(annulus_weight));
			}
			else
				std::cout << neigh_indx << " is NAN" << std::endl;
		}
		for (int i = 0; i < this->n_annulus_; i++)
		{
			if (histograms.row(i).norm() > 0)
			{
				histograms.row(i).normalize();
			}
			for (int k = 0; k < this->n_bins_; k++)
			{

				features_row.at<float>(0, (i*this->n_bins_) + k) = (histograms(i, k));
			}
		}
	}
	else
		std::cerr << "Il punto " << point_index << " non ha vicini" << std::endl;

	return features_row;
}
#define PCL_INSTANTIATE_KeypointLearningDetector(T,U,N) template class PCL_EXPORTS pcl::KeypointLearningDetector<T,U,N>;


#endif //PCL_KEYPOINTS_IMPL_KEYPOINT_LEARNING_H_
