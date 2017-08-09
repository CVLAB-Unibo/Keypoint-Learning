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
 *   * Neither the name of Willow Garage, Inc. nor the names of its
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
 */
#include <pcl/impl/instantiate.hpp>
#include <pcl/point_types.h>
#include "impl/KeypointLearning.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void findAnnulusPair(int n_annulus,float distance,float support, int& annulus_index, int& annulus_index_pair, float& annulus_weight){

	float annulus_dimension = support / n_annulus;
	
	annulus_index = static_cast<int>(floor(distance/annulus_dimension));
	if (annulus_index == n_annulus)
		annulus_index--;
	
	assert(annulus_index>=0 &&  annulus_index<n_annulus  );
	

	float annulus_center = (annulus_index * annulus_dimension) + (annulus_dimension/2);

	annulus_weight = distance - annulus_center;
	annulus_weight/=annulus_dimension;

	annulus_index_pair = (annulus_weight > 0) ? annulus_index +1 : annulus_index -1;
	
	if(annulus_index_pair == -1)
		annulus_index_pair = 0;
	if(annulus_index_pair == n_annulus)
		annulus_index_pair = annulus_index;

	annulus_weight = abs(annulus_weight);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void findBinPair(int n_bins,float cosine, int& bin_index, int& bin_index_pair, float& bin_weight){

	if(cosine < 0)
		cosine = 0;
	if(cosine > 2)
		cosine = 2;

	float bin_dimension = 2 /(float) n_bins;
	bin_index = static_cast<int>(floor(cosine/bin_dimension));
	if (bin_index == n_bins)
		bin_index--;
	
	assert(bin_index>=0 &&bin_index  <n_bins  );
	

	float bin_center = (bin_index * bin_dimension) + (bin_dimension/2);
	bin_weight = cosine - bin_center;
	bin_weight/=bin_dimension;
	bin_index_pair = (bin_weight > 0) ? bin_index +1 : bin_index -1;
	if(bin_index_pair == -1)
		bin_index_pair = 0;
	if(bin_index_pair == n_bins)
		bin_index_pair = bin_index;
	bin_weight = abs(bin_weight);
}
// Instantiations of specific point types
//PCL_INSTANTIATE (SHOTKeypointLearning, (PCL_XYZ_POINT_TYPES)(PCL_XYZ_POINT_TYPES))
