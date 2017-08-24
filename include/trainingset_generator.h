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

#ifndef TRAININGSET_GENERATOR_H_
#define TRAININGSET_GENERATOR_H_

#include "view.h"

namespace kpl
{
	/** \brief Implementation of algorithm for training set generation as presentend in:
	*	 "Learning a Descriptor-Specific 3D Keypoint Detector", Salti S., Tombari F., Spezialetti R., Di Stefano L., The IEEE International Conference on Computer Vision (ICCV), 2015, pp. 2318-2326.
	*
	*	\author Riccardo Spezialetti
	*/
	template<typename PointInT, typename NormalT, typename PointOutT, typename HistogramT>
	class TrainingSetGenerator
	{
	public:
		typedef typename pcl::PointCloud<PointOutT> PointCloudOut;
		typedef typename PointCloudOut::Ptr PointCloudOutPtr;
		typedef typename PointCloudOut::ConstPtr PointCloudOutConstPtr;

		typedef typename pcl::PointCloud<HistogramT> PointCloudHist;
		typedef typename PointCloudHist::Ptr PointCloudHistPtr;
		typedef typename PointCloudHist::ConstPtr PointCloudHistConstPtr;

		typedef typename View<PointInT, NormalT, HistogramT> ViewIn;
		
		/** \brief Empty constructor */
		TrainingSetGenerator();

		/** \brief Empty destructor */
		~TrainingSetGenerator();

		/** \brief Generate positive points
		*	\param[in]	view					view for wich generate positives
		*	\param[in]	overllapping_views		overlapping views for view
		*	\param[in]	th_EclideanDistance		euclidean distance threshold to accept descriptor matching
		*	\param[in]	r_nms					non maxima suppression radius
		*	param[out]	cloud_positive			cloud with generated positives
		*/
		int generatePositives(ViewIn& view, std::vector<ViewIn>& overlapping_views, const float th_EclideanDistance, const float r_nms, PointCloudOutPtr cloud_positive);

		/**	\brief Generate random negative points
		*	\param[in] view					view for wich generate positives
		*	\param[in] cloud_positive		cloud of positive points generated with generatePositives(...)
		*	\param[in] r_negative			radius away from positives in which to look for the negatives
		*	\param[out] cloud_negative		cloud with generated negatives
		*/
		int generateNegatives(ViewIn& view, const PointCloudOutConstPtr cloud_positive, const float r_negative, PointCloudOutPtr cloud_negative);

	private:
		std::vector<pcl::Correspondence> computeSortedCorrespondecesOnDescriptor(PointCloudHistConstPtr cloud_src, std::vector<int>& indices_src, PointCloudHistConstPtr cloud_trg, std::vector<int>& indices_trg);

		void checkCorrespondencesOnL2(std::vector<pcl::Correspondence>& sorted_correspondences, const ViewIn& view_src, const ViewIn& view_trg, std::vector<int>& indices, const float th_EclideanDistance, const float nms_radius);

		int refineOnView(ViewIn& view, ViewIn& view_for_refinement, const std::vector<int>& indices_positive, const float r_tol, pcl::CorrespondencesPtr correspondences);

		int keepUniqueCorrespondences(const std::vector<pcl::CorrespondencesPtr> correspondences_to_filter, pcl::CorrespondencesPtr correspondences_filtered);
		
		void singleThreadFillCorrespondences(PointCloudHistConstPtr cloud_src, pcl::KdTreeFLANN<HistogramT>& kdt_trg, const int start, const int end, const int k, const std::vector<int>& indices_src, const std::vector<int>& indices_trg, std::vector<pcl::Correspondence>& correspondences);

	};
}

#include "impl\trainingset_generator.hpp"

#endif // !TRAININGSET_GENERATOR_H_
