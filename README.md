# HypeLCNN Overview
This repository includes source codes for paper "A Deep Learning Classification Framework with Spectral and Spatial Feature Fusion Layers for Hyperspectral and Lidar Sensor Data"

Developed using Tensorflow 1.x(Tested on version 1.10 to 1.15). This repository includes a complete suite for hyperspectral and lidar, neural net based classification. Primary features :
- Support for hyperparameter estimation
- Plug-in based neural network implementation( via NNModel interface )
- Plug-in based data set integration( via DataLoader interface )
- Data efficient implementations for training( memory efficient/in-memory/record based )
- Ability to use data set integration in classic machine learning methods
- Training, classification and metrics integration for neural networks
- Sample implementations for capsule network and neural networks
- CPU/GPU/TPU(work in progress) based training
- GAN based data augmenter integration
- Cross fold validation support

Source codes can be used for best practices of applying tensorflow in training large data sets, integrating metrics, merging two different neural networks for data augmentation

NOTE : Data set files are too large to commit on repository.

# Requirements
Work with Python 3.6. Library requirements are placed in "requirements.txt".

# Source Files
- PHD.ipynb : A sample notebook for executing the developed neural network application
- NNModel.py : Neural Network definition interface, provides methods for declaring loss function, base parameter values, hyper parameter value range
- CNNModelv2.py : Model declaration source code for CNN Model V2
- CNNModelv4.py : Model declaration source code for CNN Model V4 ( Primary network model(HypeLCNN) for the paper )
- CNNModelv5.py : Model declaration source code for CNN Model V5 ( Dual CNN )
- CAPNModelv2.py : Capsule Network Model v2
- class_ml_trainer : SVM and Random Forest Classifier implementation using scikit-learn
- DataImporter.py : DataImporter interface class
- InMemoryImporter.py : Tensorflow(1.x)-Placeholder based in memory data importer(fastest but memory inefficient implementation) 
- GeneratorImporter.py : Tensorflow(1.x)-Generator based data importer(memory efficient but slower than in memory implementation
- TFRecordImporter.py : Tensorflow(1.x)-TFRecord based data importer. TFRecord files can be created using utilities\tfrecord_writer.py
- DataLoader.py : DataLoader interface class, used for integrating different data sets to the training/inference system
- GRSS2013DataLoader.py : GRSS2013 data set loader implementation
- GRSS2018DataLoader.py : GRSS2018 data set loader implementation
- GULFPORTDataLoader.py : Gulfport data set loader implementation
- deep_classification_gpu.py : Main training execution class
- load_checkpoint_calc_accuracy : Loads a checkpoint file and performs scene classification using a dataset
- monitored_session_runner.py : Tensorflw(1.x) Monitored session runner implementation
- *.json : Parameter value files
- utilities\cycle_gann_inference : Gan inference implementation
- utilities\cycle_gann_training : Gan training runner
- utilities\tfrecord_writer : TF record generator from samples

# In progress
- Data set file commit
- Source code documentation
- Refactoring for cleaner code
- TF 2.x/Pytorch implementation