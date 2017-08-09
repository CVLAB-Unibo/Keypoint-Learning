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
#ifndef POINT_CLOUD_UTILITIES_IMPL
#define POINT_CLOUD_UTILITIES_IMPL

#include <pcl\common\distances.h>
#include <pcl\common\io.h>
#include <pcl\point_cloud.h>
#include <pcl\point_types.h>

#include "point_cloud_utilities.h"

namespace kpl 
{
	template<typename HistogramT>
	void removeNaNFromHistogram(const pcl::PointCloud<HistogramT>& cloud_in, pcl::PointCloud<HistogramT>& cloud_out, std::vector<int>& indices)
	{

		// If the clouds are not the same, prepare the output
		if (&cloud_in != &cloud_out)
		{
			cloud_out.header = cloud_in.header;
			cloud_out.points.resize(cloud_in.points.size());
		}
		// Reserve enough space for the indices
		indices.resize(cloud_in.points.size());
		size_t j = 0;

		if (cloud_in.is_dense)
		{
			// Simply copy the data
			cloud_out = cloud_in;
			for (j = 0; j < cloud_out.points.size(); ++j)
				indices[j] = static_cast<int>(j);
		}
		else 
		{
			for (size_t i = 0; i < cloud_in.points.size(); ++i)
			{
				bool is_nan = false;

				for (size_t i_h = 0; i_h < cloud_in.points[i_h].descriptorSize(); ++i_h)
				{
					is_nan = !pcl_isfinite(cloud_in.points[i].histogram[i_h]);
					if (is_nan)
						break;
				}

				if (!is_nan)
				{
					cloud_out.points[j] = cloud_in.points[i];
					indices[j] = static_cast<int>(i);
					j++;
				}
			}

			if (j != cloud_in.points.size())
			{
				// Resize to the correct size
				cloud_out.points.resize(j);
				indices.resize(j);
			}

			cloud_out.height = 1;
			cloud_out.width = static_cast<uint32_t>(j);

			// Removing bad points => dense (note: 'dense' doesn't mean 'organized')
			cloud_out.is_dense = true;
		}
	}


	template<typename HistogramT>
	cv::Mat convertPointCloudHistogramToCvMat(typename const pcl::PointCloud<HistogramT>::ConstPtr cloud_in)
	{
		const size_t descriptor_length = cloud_in->points[0].descriptorSize();
		cv::Mat descriptors = cv::Mat::zeros(cloud_in->points.size(), static_cast<int>(descriptor_length), CV_32F);

		for (size_t i_p = 0; i_p < cloud_in->points.size(); i_p++)
		{
			float *const ptr_to_mat = descriptors.ptr<float>(i_p);
			memcpy(ptr_to_mat, cloud_in->points[i_p].histogram, descriptor_length * sizeof(float));
		}
		return descriptors;
	}

	template<typename PointT> double
	computeCloudResolution(typename const pcl::PointCloud<PointT>::ConstPtr &cloud)
	{
		double res = 0.0;
		int n_points = 0;
		int nres;

		std::vector<int> indices(2);
		std::vector<float> sqr_distances(2);
		pcl::search::KdTree<PointT> tree;
		tree.setInputCloud(cloud);

		for (size_t i = 0; i < cloud->size(); ++i)
		{
			if (!pcl_isfinite((*cloud)[i].x))
			{
				continue;
			}
			//Considering the second neighbor since the first is the point itself.
			nres = tree.nearestKSearch((int)i, 2, indices, sqr_distances);
			if (nres == 2)
			{
				res += sqrt(sqr_distances[1]);
				++n_points;
			}
		}
		if (n_points != 0)
		{
			res /= n_points;
		}
		return res;
	}

	template<typename PointT>int
	nonMaximaSuppression(typename const pcl::PointCloud<PointT>::ConstPtr cloud_in, const std::vector<int>& true_positive, const float nms_radius, std::vector<int>& maxima)
	{

		pcl::KdTreeFLANN<PointT> kdtree;
		kdtree.setInputCloud(cloud_in);
		std::vector<int> neighbors_index;
		std::vector<float> neighbors_distances;

		std::vector<bool> is_maxima(true_positive.size(), true);
		std::vector<bool> is_tp(cloud_in->points.size(), false);

		std::vector<int> m_cloud_to_tp(cloud_in->points.size(),0);

		//Set true only points in cloud that are tp
		for (size_t i_tp = 0; i_tp < true_positive.size(); ++i_tp) 
		{
			 is_tp[true_positive[i_tp]] = true;	 
			 m_cloud_to_tp[true_positive[i_tp]] = static_cast<int>( i_tp);
		}

		int suppressed = 0;

		for (size_t i_tp = 0; i_tp < true_positive.size(); ++i_tp)
		{
			int tp_index = true_positive[i_tp];
			if (is_maxima[i_tp] == true)
			{
				if (kdtree.radiusSearch(cloud_in->points[tp_index], nms_radius, neighbors_index, neighbors_distances)>0)
				{
					for (int i = 1; i < neighbors_index.size(); i++)
					{
						if (is_tp[neighbors_index[i]])
						{
							const int indx_on_tp = m_cloud_to_tp[neighbors_index[i]];

							if (is_maxima[indx_on_tp])
							{
								is_maxima[indx_on_tp] = false;
								suppressed++;
							}
						}
					}
				}
			}
		}

		//Keep maxima
		for (size_t i_max = 0; i_max < is_maxima.size(); i_max++)
		{
			if (is_maxima[i_max])
			{
				maxima.push_back(true_positive[i_max]);
			}
		}
		return suppressed;
	}
}
#endif // !POINT_CLOUD_UTILITIES_IMPL
