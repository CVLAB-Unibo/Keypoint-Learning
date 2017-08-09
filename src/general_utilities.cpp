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

#ifndef GENERAL_UTILITIES_IMPL
#define GENERAL_UTILITIES_IMPL

#include <iostream>
#include <map>

#include <boost\algorithm\string.hpp>
#include <boost\lexical_cast.hpp>

#include "general_utilities.h"

namespace kpl
{
	
	static bool compare_path_by_date(const boost::filesystem::path &p1, const boost::filesystem::path &p2)
	{
		return  boost::filesystem::last_write_time(p1) < boost::filesystem::last_write_time(p2);
	}

	static bool compare_path_by_name(const boost::filesystem::path &p1, const boost::filesystem::path &p2)
	{
		return  p1.string() < p2.string();
	}

	int getAllFilesInFolder(const boost::filesystem::path& path_folder, const std::string extension, std::vector<std::string>& values)
	{

		if (!boost::filesystem::exists(path_folder) || !boost::filesystem::is_directory(path_folder))
		{
			perror("Input directory is not directory.");
			return -1;
		}

		boost::filesystem::directory_iterator it(path_folder);
		boost::filesystem::directory_iterator endit;
		std::vector< boost::filesystem::path> all_path;

		while (it != endit)
		{
			if (boost::filesystem::is_regular_file(*it) && it->path().extension() == extension)
			{

				all_path.push_back(*it);
			}
			++it;
		}
		std::sort(all_path.begin(), all_path.end(), compare_path_by_name);

		values.resize(all_path.size(), "");

		for (int i_path = 0; i_path < all_path.size(); i_path++)
		{
			values[i_path] = (all_path[i_path].filename().string());
		}

		return static_cast<uint32_t>(values.size());
	}

	int readCSVFile(const std::string file_name, std::vector<std::vector<std::string>>& lines)
	{
		std::ifstream file(file_name);

		std::string line;
		std::vector<std::string> split_vector;
		int count = 0;

		if (file.is_open())
		{
			while (!file.eof())
			{
				std::getline(file, line, '\n');
				if (line == "" || count == 0)
				{
					count++;
					continue;
				}

				boost::split(split_vector, line, boost::is_any_of(";"), boost::token_compress_on);
				lines.push_back(split_vector);
			}
		}
		else
		{
			std::cerr << file_name << " file not found" << std::endl;
			return -1;
		}

		file.close();

		return static_cast<uint32_t>(lines.size());
	}

	Eigen::Matrix4f computeGroundTruth(const std::string path, const int id)
	{
		std::string line;

		std::ifstream file(path);
		int offset = 3;
		std::vector<std::string> splitVec;
		int index = id * 5 + offset;
		Eigen::Matrix4f transformationMatrix;

		if (file.is_open())
		{
			int count = 1;
			bool done = false;

			while (count < index)
			{
				std::getline(file, line);
				count++;
			}

			for (int i = 0; i < 4; i++)
			{
				std::getline(file, line);
				boost::split(splitVec, line, boost::is_any_of("\t"), boost::token_compress_on);
				transformationMatrix.row(i) << boost::lexical_cast<float>(splitVec.at(0)),
					boost::lexical_cast<float>(splitVec.at(1)),
					boost::lexical_cast<float>(splitVec.at(2)),
					boost::lexical_cast<float>(splitVec.at(3));
			}
		}
		else
			std::cout << "GroundTruth file not found" << std::endl;

		transformationMatrix.inverse();

		return transformationMatrix;
	}

	void findViews(const boost::filesystem::path& path, const std::string extension, std::map<int, std::string>& ids_view_names)
	{
		std::vector<std::string> views_name;
		if (getAllFilesInFolder(path, extension, views_name))
		{

			int id = 0;
			for (size_t i_v = 0; i_v < views_name.size(); i_v++)
			{

				ids_view_names.insert(std::pair<int, std::string>(id, views_name[i_v]));
				id++;
			}
		}
		else
		{
			std::cerr << "Impossible to find views." << std::endl;
		}
	}

	void getColorForHeatMap(const float intensity, int &r, int &g, int &b)
	{
		const int NUM_COLORS = 5;
		static int colors[NUM_COLORS][3] = { { 0, 0, 255 },{ 0, 255, 255 },{ 0, 255, 0 },{ 255, 255, 0 },{ 255, 0, 0 } };

		int color_start, color_end;

		if (intensity <= 0)
		{
			color_start = color_end = 0;
		}
		else if (intensity >= 1)
		{
			color_start = color_end = NUM_COLORS - 1;
		}
		else
		{
			color_start = floor(intensity * (NUM_COLORS - 1));
			color_end = color_start + 1;
		}

		r = static_cast<int>((colors[color_end][0] - colors[color_start][0]) * intensity + colors[color_start][0]);
		g = static_cast<int>((colors[color_end][1] - colors[color_start][1]) * intensity + colors[color_start][1]);
		b = static_cast<int>((colors[color_end][2] - colors[color_start][2]) * intensity + colors[color_start][2]);

	}
}

#endif // !GENERAL_UTILITIES_IMPL
