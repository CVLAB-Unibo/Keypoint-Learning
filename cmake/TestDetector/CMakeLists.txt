cmake_minimum_required(VERSION 2.8 FATAL_ERROR)


set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "possible configurations" FORCE)
# In case the user does not setup CMAKE_BUILD_TYPE, assume it's Release
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type default to Release" FORCE)
endif("${CMAKE_BUILD_TYPE}" STREQUAL "")

###########################################################################################
# TARGETS

SET(FILES_TD 	${PROJECT_SOURCE_DIR}/src/main_test_detector.cpp
				${PROJECT_SOURCE_DIR}/src/KeypointLearning.cpp
				${PROJECT_SOURCE_DIR}/include/KeypointLearning.h
				${PROJECT_SOURCE_DIR}/include/impl/KeypointLearning.hpp) 
	
ADD_EXECUTABLE(TestDetector ${FILES_TD})
target_link_libraries(TestDetector ${Boost_LIBRARIES})
target_link_libraries(TestDetector ${PCL_LIBRARIES})
target_link_libraries(TestDetector ${OpenCV_LIBS})
