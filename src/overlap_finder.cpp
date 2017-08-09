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
#ifndef OVERLAP_FINDER_IMPL_
#define OVERLAP_FINDER_IMPL_

#include <iostream>
#include <fstream>
#include <string>

#include <boost\algorithm\string.hpp>
#include <boost\lexical_cast.hpp>

#include "overlap_finder.h"

std::vector<int> kpl::FileBasedOverlapFinder::findOverlappingViews(const int id_view) const
{		
		std::string line;
	
		std::ifstream file(file_overlapping_areas_);

		std::vector<std::string> split_vector;
		std::vector<std::string> v_pairs;
		std::vector<int> v_ids;
		std::vector<float> v_thresholds;

		float threshold;

		bool flag = true;

		if (file.is_open())
		{
			int i = 0;
			while (!file.eof())
			{
				i++;
				std::getline(file, line);

				if (line == "")
					continue;

				if (i >= 6)
				{
					boost::split(split_vector, line, boost::is_any_of(" "), boost::token_compress_on);
					threshold = boost::lexical_cast<float>(split_vector.at(5));

					if (threshold >= th_overlap_)
					{

						if (id_view == boost::lexical_cast<int>(split_vector.at(1)) && flag)
						{
							v_pairs.push_back(split_vector.at(3));
							v_ids.push_back(boost::lexical_cast<int>(split_vector.at(1)));
							flag = false;
						}

						if (id_view == boost::lexical_cast<int>(split_vector.at(2)) && flag)
						{
							v_pairs.push_back(split_vector.at(4));
							v_ids.push_back(boost::lexical_cast<int>(split_vector.at(2)));
							flag = false;
						}

						if (id_view == boost::lexical_cast<int>(split_vector.at(1)))
						{
							v_pairs.push_back(split_vector.at(4));
							v_ids.push_back(boost::lexical_cast<int>(split_vector.at(2)));
							v_thresholds.push_back(threshold);
						}
						if (id_view == boost::lexical_cast<int>(split_vector.at(2)))
						{
							v_pairs.push_back(split_vector.at(3));
							v_ids.push_back(boost::lexical_cast<int>(split_vector.at(1)));
							v_thresholds.push_back(threshold);
						}
					}
				}
			}
		}
		else
			std::cout << "Overlap file not found" << std::endl;

		if (v_ids.size() == 0)
		{
			v_ids.push_back(id_view);
		}

		return v_ids;
}

#endif // !OVERLAP_FINDER_IMPL

