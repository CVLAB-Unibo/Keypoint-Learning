cmake_minimum_required(VERSION 2.8 FATAL_ERROR)


set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "possible configurations" FORCE)
# In case the user does not setup CMAKE_BUILD_TYPE, assume it's Release
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type default to Release" FORCE)
endif("${CMAKE_BUILD_TYPE}" STREQUAL "")

###########################################################################################
# TARGETS

SET(FILES_GTS ${PROJECT_SOURCE_DIR}/src/general_utilities.cpp
				${PROJECT_SOURCE_DIR}/src/main_generate_training_set.cpp
				${PROJECT_SOURCE_DIR}/src/overlap_finder.cpp
				${PROJECT_SOURCE_DIR}/src/view_manager.cpp
				${PROJECT_SOURCE_DIR}/include/general_utilities.h
				${PROJECT_SOURCE_DIR}/include/trainingset_generator.h
				${PROJECT_SOURCE_DIR}/include/point_cloud_utilities.h
				${PROJECT_SOURCE_DIR}/include/overlap_finder.h
				${PROJECT_SOURCE_DIR}/include/view.h
				${PROJECT_SOURCE_DIR}/include/view_manager.h
				${PROJECT_SOURCE_DIR}/include/impl/point_cloud_utilities.hpp
				${PROJECT_SOURCE_DIR}/include/impl/general_utilities.hpp
				${PROJECT_SOURCE_DIR}/include/impl/view_manager.hpp
				${PROJECT_SOURCE_DIR}/include/impl/trainingset_generator.hpp
	 ) 
	
ADD_EXECUTABLE(GenerateTrainingSet ${FILES_GTS})
target_link_libraries(GenerateTrainingSet ${Boost_LIBRARIES})
target_link_libraries(GenerateTrainingSet ${PCL_LIBRARIES})
target_link_libraries(GenerateTrainingSet ${OpenCV_LIBS})
