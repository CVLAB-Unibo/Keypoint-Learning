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

#ifndef POINT_CLOUD_UTILITIES_H
#define POINT_CLOUD_UTILITIES_H

#include "opencv2\core.hpp"

#include <pcl\point_types.h>
#include <pcl\impl\point_types.hpp>

namespace kpl
{
	/** \brief Convert a pointcloud of histogram in a cvMat
	* \param[in] cloud	pointcloud to convert
	* returns cvMat with histograms
	*/
	template<typename HistogramT>cv::Mat
	convertPointCloudHistogramToCvMat(typename const pcl::PointCloud<HistogramT>::ConstPtr cloud);

	/** \brief Remove nan from Histogram. 
	*	\param[in]  cloud_in		input pointcloud
	*	\param[out] cloud_out	pointcloud with nan removed
	*	\param[out] indices		is mapping from cloud_in to cloud_out
	*/
	template<typename HistogramT> void
	removeNaNFromHistogram(const pcl::PointCloud<HistogramT>& cloud_in, pcl::PointCloud<HistogramT>& cloud_out, std::vector<int>& indices);

	/** \brief Compute cloud resolution
	*	\param[in] cloud	input pointcloud
	*	returns computed cloud resolution
	*/
	template<typename PointT> double
	computeCloudResolution(typename const pcl::PointCloud<PointT>::ConstPtr &cloud);

	/** \brief Check which of points to suppress is maxima
	*	\param[in] cloud				input pointcloud
	*	\param[in] points_to_suppress	points for which compute nms
	*	\param[in] nms_radius			radius for nms
	*	\param[out] maxima				subset of points_to_suppress that are maxima
	*	returns number of suppressed points
	*/
	template<typename PointT> int
	nonMaximaSuppression(typename const pcl::PointCloud<PointT>::ConstPtr cloud, const std::vector<int>& points_to_suppress, const float nms_radius, std::vector<int>& maxima);
}

#include "impl/point_cloud_utilities.hpp"

#endif //POINT_CLOUD_UTILITIES_H 