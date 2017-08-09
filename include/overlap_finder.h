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

#ifndef OVERLAP_FINDER_H
#define OVERLAP_FINDER_H

#include <vector>

namespace kpl
{
	/** \brief OverlapFinder is a class with the aim of finding views with a certain percentage of overlapping area
	*
	* \author Riccardo Spezialetti
	*/

	class OverlapFinder
	{
	public:
		/**	\brief Interface for overlap finder
		* \param[in] id		identifier number of view
		* returns a vector with ids of overlapping views
		*/
		virtual std::vector<int> findOverlappingViews(const int id) const = 0;
	};

	/** \brief FileOverlapFinder Simple overlap finder based on file
	*
	* \author Riccardo Spezialetti
	*/

	class FileBasedOverlapFinder : public OverlapFinder
	{
	public:
		/**	\brief Constructor
		* \param[in] file_overlapping_areas		file with list of id view pair and percentage of overlapping area
		* 
		*/
		FileBasedOverlapFinder(std::string file_overlapping_areas) : file_overlapping_areas_(file_overlapping_areas) {};

		/** \brief Set overlapping threshold
		*	\param[in] threshold	overlapping threshold
		*/
		void inline setThreshold(double threshold) { th_overlap_ = threshold; };

		std::vector<int> findOverlappingViews(const int id) const;

	private:
		/** \brief treshold to use in findOverlappingViews */
		double th_overlap_;
		
		/** \brief file file with list of id view pair and percentage of overlapping area*/
		std::string file_overlapping_areas_;
		
	};
}

#endif //OVERLAP_FINDER_H