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

### TrainDetector: this sample train and save random forest using features described in [1]. The required console arguments are the following:
* annuli: Annuli for features computation.
*	bins: Bins for features computation.
* pathDataset: Path to dataset (same folder used in GenerateTrainingSet)
* pathTrainingData: Path for training data-> Positives in: pathTrainingData\\Model_Name\\positives and Negatives in: pathTrainingData\\Model_Name\\negatives.
* pathRF: Path for Random Forest.
* radiusFeatures: Radius for features computation.
*	msc: Min samples count of Random Forest.
* nameRF: Name of YAML file.
* ntrees: Number of trees of Random Forest.

### TestDetector: example of keypoints detection on point cloud. The required console arguments are the following:
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
