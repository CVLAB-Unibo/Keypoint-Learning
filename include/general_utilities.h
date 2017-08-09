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

#ifndef GENERAL_UTILITIES_H
#define GENERAL_UTILITIES_H

#include <vector>

#include <boost\filesystem.hpp>

#include <Eigen\Dense>

#include <pcl\point_cloud.h>
#include <pcl\point_types.h>

namespace kpl {

	/** \brief Get all files in a folder with specific extension
	  * \param[in] path_folder	path to folder
	  * \param[in] extension	files extension
	  * \param[out] values		names of file found in folder
	  *	returns number of files
	  */
	int getAllFilesInFolder(const boost::filesystem::path& path_folder, const std::string extension, std::vector<std::string>& values);

	/** \brief Read all lines in a comma separated values file
	* \param[in] file_name		path to file
	* \param[out] values		for each row vector of content
	* returns the number of lines read
	*/
	int readCSVFile(const std::string file_name, std::vector<std::vector<std::string>>& values);

	/** \brief In order to each specific dataset, load groundtruth matrix
	* \param[in] path	path to file
	* \param[in] id		identifier number of view
	* returns Groundtruth matrix
	*/
	Eigen::Matrix4f computeGroundTruth(const std::string path, const int id);

	/** \brief Given a path, find all 2.5 views in the path with extension
	*	/param[in] path_folder		path to folder
	*	/param[in] extension		file extension usually .pcd or .ply
	*	/param[in] ids_view_names	map containing the name and a progressive identifier number for each view
	*/
	void findViews(const boost::filesystem::path& path_folder, const std::string extension, std::map<int, std::string>& ids_view_names);

	/** \brief Given a value between 0 and 1, returns color component
	*	/param[in] intensity	value between 1 and zero
	*	/param[out] r			red component
	*	/param[out] g			green component
	*	/param[out]	b			blue component
	*/
	void getColorForHeatMap(const float intensity, int &r, int &g, int &b);

}

#include <impl\general_utilities.hpp>

#endif // !GENERAL_UTILITIES_H
