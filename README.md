# HypeLCNN Overview
This repository includes source codes for hyperspectral and LiDAR fusion system for classification and GAN based hyperspectral sample generation.

Developed using Tensorflow 1.x(Tested on version 1.10 to 1.15). This repository includes a complete suite for hyperspectral and lidar, neural net based classification. Primary features :
- Support for hyperparameter estimation
- Plug-in based neural network implementation( via NNModel interface )
- Plug-in based data set integration( via DataLoader interface )
- Data efficient implementations for training( memory efficient/in-memory/record based )
- Ability to use data set integration in classic machine learning methods
- Training, classification and metrics integration for neural networks
  - Cross fold validation support
- Sample implementations for capsule network and neural networks
- CPU/GPU/TPU(work in progress) based training
- GAN based data augmenter implementation
  - Cycle GAN
  - Vanilla GAN
  - Contrastive Unpaired Translation(CUT) GAN
  - Training and evaluation codes.

Source codes can be used for best practices of applying tensorflow 1.x for;
 - training large data sets
 - integrating custom metrics
 - merging two different neural networks for data augmentation
 - reading summary file

# Data sets
Data set files are too large to commit on repository. Access to data sets can be done using this URL;

https://drive.google.com/drive/folders/1oiNRuPAkQ1MpJEUjKYE8YtXpcM405usE?usp=sharing

Each data set has its own licensing, and you should include them in your paper/work.

# Requirements
Work with Python 3.6 and 3.7. Library requirements are placed in "requirements.txt".

# Source Files
- notebook.ipynb : A sample notebook for executing the developed ML models.
- NNModel.py : Neural Network definition interface, provides methods for declaring loss function, base parameter values, hyper parameter value range.
- CNNModelv4.py : Model declaration source code for CNN Model V4 (Network model HypeLCNN).
- DUALCNNModelv1.py : Model declaration source code for CNN Dual Model.
- CONCNNModelv1.py : Model declaration source code for Context CNN Model.
- CAPNModelv1.py : Capsule Network Model v2.
- classic_ml_trainer : SVM and Random Forest Classifier implementation using scikit-learn.
- DataImporter.py : DataImporter interface class.
- InMemoryImporter.py : Tensorflow(1.x)-Placeholder based in memory data importer(fastest but memory inefficient implementation).
- GeneratorImporter.py : Tensorflow(1.x)-Generator based data importer(memory efficient but slower than in memory implementation.
- TFRecordImporter.py : Tensorflow(1.x)-TFRecord based data importer. TFRecord files can be created using utilities\tfrecord_writer.py.
- DataLoader.py : DataLoader interface class, used for integrating different data sets to the training/inference system.
- GRSS2013DataLoader.py : GRSS2013 data set loader implementation.
- GRSS2018DataLoader.py : GRSS2018 data set loader implementation.
- GULFPORTDataLoader.py : Gulfport data set loader implementation.
- GULFPORTALTDataLoader.py : Gulfport data set alternative loader implementation(shadow data augmentation).
- AVONDataLoader.py : AVON data set loader implementation.
- deep_classification_multigpu.py : Terrain classification training execution class.
- load_checkpoint_calc_accuracy : Loads a checkpoint file and data set and performs scene classification.
- monitored_session_runner.py : Tensorflw(1.x) Monitored session runner implementation.
- modelconfigs/*.json : Parameter values for different models.
- gan/gan_infer_for_shadow.py : Gan inference implementation.
- gan/gan_train_for_shadow.py : Gan training runner.
- gan/*_wrapper.py : Wrappers for various gan archs.
- utilities/tfrecord_writer.py : TF record generator from samples.
- utilities/hsi_rgb_converter.py : HSI to RGB conversion script.
- utilities/read_summary_file.py : Reads tensorflow summary file and extract some statistical information.

# In progress
- Source code documentation
- Refactoring for cleaner code
- TF 2.x/Pytorch implementation