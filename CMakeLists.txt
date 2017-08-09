
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(KeypointLearning)
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)

# postfix, based on type
SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "postfix applied to debug build of libraries")
SET(CMAKE_RELEASE_POSTFIX "" CACHE STRING "postfix applied to release build of libraries")
SET(CMAKE_RELWITHDEBINFO_POSTFIX "rd" CACHE STRING "postfix applied to release-with-debug-information libraries")
SET(CMAKE_MINSIZEREL_POSTFIX "s" CACHE STRING "postfix applied to minimium-size-build libraries")

#Find modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

#Pcl
find_package(PCL 1.8 REQUIRED COMPONENTS common io filters octree keypoints visualization registration surface)
if(PCL_FOUND)
   include_directories(${PCL_INCLUDE_DIRS})
   link_directories(${PCL_LIBRARY_DIRS})
   add_definitions(${PCL_DEFINITIONS})
endif(PCL_FOUND)

#BOOST
find_package(Boost 1.46 REQUIRED COMPONENTS program_options thread system filesystem random)
if(BOOST_FOUND)
  include_directories(${BOOST_INCLUDE_DIR}) # cached
  link_directories(${BOOST_LIBRARY_DIRS}) # cached
  add_definitions(${BOOST_DEFINITIONS})
endif(BOOST_FOUND)

#OpenCV
find_package(OPENCV REQUIRED)
if(OPENCV_FOUND)
	include_directories(${OPENCV_INCLUDE_DIRS})
	link_directories(${OPENCV_LIBRARY_DIRS})
	add_definitions(${OPENCV_DEFINITIONS})
endif(OPENCV_FOUND)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)

ADD_SUBDIRECTORY(cmake/GenerateTrainingSet)
ADD_SUBDIRECTORY(cmake/TrainDetector)
ADD_SUBDIRECTORY(cmake/TestDetector)



	

