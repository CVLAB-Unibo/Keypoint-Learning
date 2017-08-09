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
#ifndef VIEW_H_
#define VIEW_H_

#include <Eigen\dense>

#include <pcl\pcl_base.h>
#include <pcl\point_cloud.h>

#include <opencv2\opencv.hpp>

namespace kpl 
{	 
	/** \brief View data structure
    *
    * \author Riccardo Spezialetti
    */
	template<typename PointInT, typename NormalT, typename HistogramT>
	struct View
	{
		/** \brief identifier*/
		unsigned int id_;

		/** \brief pointcloud with points*/
		typename pcl::PointCloud<PointInT>::Ptr c_points_;

		/** \brief pointcloud with descriptors*/
		typename pcl::PointCloud<HistogramT>::Ptr c_descriptors_;

		/** \brief pointcloud with normals*/
		typename pcl::PointCloud<NormalT>::Ptr c_normals_;

		/** \brief groundtruth matrix*/
		Eigen::Matrix4f matrix_gt_;

		/** \brief lut from pointcloud indices to descriptors indices*/
		std::vector<int> map_point_desc_;
	};

}
#endif	//VIEW_H