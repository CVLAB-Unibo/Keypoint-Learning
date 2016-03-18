cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(Keypoint-Learning)

#pcl
find_package(PCL 1.6 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#opencv
find_package(OpenCV 2.4 REQUIRED )
if(OpenCV_INCLUDE_DIRS)
	if(OpenCV_VERSION VERSION_LESS "3.0")
		include_directories(${OpenCV_INCLUDE_DIRS})
		link_directories(${OpenCV_LIB_DIRS})
	else()
		message( FATAL_ERROR "This sample will not work with opencv>=3.0, please use version 2.4" )
	endif()
endif(OpenCV_INCLUDE_DIRS)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(detector_test 	${PROJECT_SOURCE_DIR}/src/test.cpp
								${PROJECT_SOURCE_DIR}/src/KeypointLearning.cpp
								${PROJECT_SOURCE_DIR}/include/KeypointLearning.h
								${PROJECT_SOURCE_DIR}/include/impl/KeypointLearning.hpp)
target_link_libraries(detector_test ${OpenCV_LIBS} ${PCL_LIBRARIES})