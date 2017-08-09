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

#ifndef PCL_KEYPOINT_KEYPOINT_LEARNING_H_
#define PCL_KEYPOINT_KEYPOINT_LEARNING_H_

#include <pcl/keypoints/keypoint.h>
#include <ml.h>

namespace pcl
{
  namespace keypoints
  {
    /** \brief Implementation of a simple keypoint learning algorithm.
      * \author Riccardo Spezialetti
	  * \author Samuele Salti
	  * \author Federico Tombari
	  *	\author Alessio Tonioni
      * \ingroup keypoints
	  */
	  template < typename PointInT, typename PointOutT, typename NormalT = pcl::Normal >
	  class KeypointLearningDetector: public Keypoint < PointInT,  PointOutT>
	  {

	  public:
		
		  typedef boost::shared_ptr<KeypointLearningDetector<PointInT, PointOutT, NormalT> > Ptr;
		  typedef boost::shared_ptr<const KeypointLearningDetector<PointInT, PointOutT, NormalT> > ConstPtr;

		  typedef typename Keypoint<PointInT, PointOutT>::PointCloudIn PointCloudIn;
		  typedef typename Keypoint<PointInT, PointOutT>::PointCloudOut PointCloudOut;
		  typedef typename Keypoint<PointInT, PointOutT>::KdTree KdTree;
		  typedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;

		  typedef typename pcl::PointCloud<NormalT> PointCloudN;
		  typedef typename PointCloudN::Ptr PointCloudNPtr;
		  typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;

		  /** \brief Constructor
		  * \param[in] prediction threshold for random forest
		  * \param[in] non maxima suppresion flag
		  * \param[in] flag to remove keypoint with same intensity in non maxina suppression
		  * \param[in] radius for non maxina suppression
		  * \param[in] number of annuli in features computation
		  * \param[in] number of bins in features computation
		  */
		  KeypointLearningDetector(double prediction_th = 0.5f, bool non_maxima = true, bool non_maxima_draws_remove = true, double non_max_radius = 0.0f, int n_annulus = 5, int n_bins = 10)
			  : prediction_th_ (prediction_th)
			  , non_maxima_radius_ (non_max_radius)
			  , non_maxima_(non_maxima)
			  , non_maxima_draws_remove_(non_maxima_draws_remove)
			  , n_annulus_(n_annulus)
			  , n_bins_(n_bins)
		  {
			  this->name_ = "Keypoint_Learnining_Detector";
		  }

		  /** \brief Empty destructor */
		  virtual ~KeypointLearningDetector() 
		  {
			  if(forest_!=0)
				 forest_.release();
		  }

		  /** \brief Provide a pointer to the input dataset
			* \param[in] cloud the const boost shared pointer to a PointCloud message
			*/
		  virtual void
			setInputCloud (const PointCloudInConstPtr &cloud);

		  /** \brief Set the normals if pre-calculated normals are available.
		  * \param[in] normals the given cloud of normals
		  */
		  virtual void
			  setNormals (const PointCloudNConstPtr &normals);

		  /** \brief Set the non maxima suppresion.
		  * \param[in] non_maxima
		  */
		  virtual void
			  setNonMaxima (bool non_maxima);

		  /** \breief Set to true to consider only one point in case of draw between two local maxima
		  * \param[in] non_maxima_pair_remove	
		  */
		  virtual void
			  setNonMaximaDrawsRemove(bool non_maxima_draws_remove);

		  /** \brief Set the maximum distance between two draw points that let one of the two survive.
		  * \param[in] non_maxima_draws_threshold
		  */
		  virtual void
			  setNonMaximaDrawsThreshold(float non_maxima_draws_threshold);

		  /** \brief Load the trained forest. */
		  virtual bool
			  loadForest(const std::string & path);

		  /** \brief Set the minimum forest output score to accept a point as keypoint. Th should be between 0 and 1
		  * \param[in] th the minimum score
		  */
		  virtual void
			  setPredictionThreshold (double th);

		  /** \brief Set the radius for the application of the non maxima supression algorithm.
		  * \param[in] non_max_radius the non maxima suppression radius
		  */
		  virtual void
			  setNonMaxRadius (double non_max_radius);

		  /** \brief Set the number of radial annulus used to compute the forest input features.
		  * \param[in] annulus the number of annulus
		  */
		 virtual void
			  setNAnnulus (int n_annulus);

		  /** \brief Set the number of normal cosine angle histogram bins used to compute the forest input features.
		  * \param[in] bins the number of annulus
		  */
		  virtual void
			  setNBins (int n_bins); 

		  /** \brief Compute features for point
		  * \param[in] point index in input cloud
		  */
		  cv::Mat computePointsForTrainingFeatures(pcl::PointIndicesConstPtr indices);

	  protected:

		  bool initCompute();

		  /** \brief Key point detection method. */
		  virtual void
			  detectKeypoints (PointCloudOut &output);

		  /** \brief Compute the forest response for each input point. */
		  virtual void 
			  runForest (PointCloudOut &output) const;

		  /** \brief Compute features for point
		  * \param[in] point index in input cloud
		  */
		  cv::Mat computePointFeatures(int point_index) const;

		  /** \brief The possibility of non maxima suppresion.*/
		  bool non_maxima_;

		  /** \brief The possibility of remove draw in non maxima suppression.*/
		  bool non_maxima_draws_remove_;

		  /** \brief The maximum distance between two draw points that let one of the two survive.*/
		  float non_maxima_draws_threshold_;

		  /** \brief The minimum forest output score to accept a point as keypoint.*/
		  double prediction_th_;

		  /** \brief The non maxima suppression radius.*/
		  double non_maxima_radius_;

		  /** \brief The number of radial annulus used to compute the forest input features.*/
		  int n_annulus_;

		  /** \brief The number of normal cosine angle histogram bins used to compute the forest input features.*/
		  int n_bins_;

		  /** \brief the Random Forest.*/
		  cv::Ptr<cv::ml::RTrees> forest_;

		  /** \brief Search Surface Cloud Normals.*/
		  PointCloudNConstPtr normals_;

	  };
  }
}

#ifdef PCL_NO_PRECOMPILE
#include "impl/KeypointLearning.hpp"
#endif
#endif    // PCL_KEYPOINT_SHOT_KEYPOINT_LEARNING_H_
