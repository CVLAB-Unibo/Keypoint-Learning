# Keypoint-Learning

Description
-----------
This framework demonstrates the use of a random forest, trained with the method proposed in [1], as a keypoints detector. 
The framework is composed by three different projects:
  * GenerateTrainingSet: implement the training set generation.
  * TrainDetector: starting from samples generated with GenerateTrainingSet, train a random forest for keypoints detection   (monoscale only).
 * TestDetector: demonstrates keypoints extraction (monoscale only).
 
If you use this code please refer to:

[1] **Learning a Descriptor-Specific 3D Keypoint Detector**, *Samuele Salti*, *Federico Tombari*, *Riccardo Spezialetti*, *Luigi Di Stefano*; The IEEE International Conference on Computer Vision (ICCV), 2015, pp. 2318-2326.

http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Salti_Learning_a_Descriptor-Specific_ICCV_2015_paper.pdf

[2] **Learning to Detect Good 3D Keypoints**, *Alessio Tonioni*, *Samuele Salti*, *Federico Tombari*, *Riccardo Spezialetti*, *Luigi Di Stefano*; International Journal of Computer Vision (IJCV), 2017.

http://rdcu.be/uQch


Usage
--------

#### GenerateTrainingSet: implementation of training set generation for random forest train describer in[1].
The algorithm requires a set of calibrated 2.5D views of 3d objects (divided into folders: object_name/2.5D_views/), along with two files, groundTruth.txt containing groundtruth matrix (affine transformation from 2.5D views to full 3d model) and overlappingAreas.txt a list of overalapping areas between 2.5D pairs. For details, refer to the examples in: data/example_groundTruth.txt and data/example_overlappingAreas.txt.

For what concern the descriptor to use, the algorithm is customizable by modifying the function computeDescriptorsPerView() in view_manager.hpp.

To increase efficiency, is possible to enable multithreading defining global variables: MULTITHREAD and MULTIVIEW.
The required console arguments are the following:

*	distance: euclidean distance to accept points with similar descriptor.
*	ext: dataset file extension.
*	pathDataset: path to folder with 3d object 2.5D views.
*	pathTrainingset: path for generated traning set.
* radiusNegative: radius for negative generation.		
* radiusNms: radius for non maxima suppression on positive.
* overlap: overlapping threshold between views.

#### TrainDetector: this sample train and save random forest using features described in [1]. The required console arguments are the following:
* annuli: annuli for features computation.
*	bins: bins for features computation.
* pathDataset: path to dataset (same folder used in GenerateTrainingSet)
* pathTrainingData: path for training data-> Positives in: pathTrainingData\\Model_Name\\positives and Negatives in: pathTrainingData\\Model_Name\\negatives.
* pathRF: path for Random Forest.
* radiusFeatures: radius for features computation.
*	msc: min samples count of Random Forest.
* nameRF: name of YAML file.
* ntrees: number of trees of Random Forest.

#### TestDetector: example of keypoints detection on point cloud. The required console arguments are the following:
* pathCloud: path to point cloud.
* pathRF: path to trained random forest.
* radiusFeatures: features support.
* radiusNMS: non maxima suppression radius.
* threshold: minimum forest output score to accept a point as keypoint. Value between 0 and 1.

Data
--------
This folder contains trained random forest for Laser Scanner dataset:
* FPFH-LaserScanner.yaml.gz: random forest trained with FPFH as descriptor.
* SHOT-LaserScanner.yaml.gz: random forest trained with SHOT as descriptor.
* SPINIMAGES-LaserScanner.yaml.gz: random forest trained with SPINIMAGES as descriptor.

Dependencies
--------
* Point Cloud Library 1.8.0
* OpenCV 3.2.0
* and all the other libraries necessary the compile the previous ones

The code has been tested on Windows 10 and Microsoft Visual Studio 2015.
