#ifndef VIEW_MANAGER_CPP_
#define VIEW_MANAGER_CPP_

#include "impl\view_manager.hpp"

Eigen::Matrix4f computeGroundTruth(const std::string path_groundtruth, const int id)
{
	std::string line;
	std::ifstream file(path_groundtruth);
	int offset = 3;
	std::vector<std::string> splitVec;
	int index = id * 5 + offset;
	Eigen::Matrix4f transformationMatrix;

	if (file.is_open())
	{
		int count = 1;
		bool done = false;

		while (count<index)
		{
			std::getline(file, line);
			count++;
		}

		//start reading matrix from file
		for (int i = 0; i<4; i++)
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

#endif // !VIEW_MANAGER_CPP