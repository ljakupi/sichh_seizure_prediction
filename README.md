**Author:** Labinot Jakupi 
**Date:** 09.07.2019

# Epileptic seizures prediction
This codebase has implementations of three neural networks (FC, LSTM and TCN) for epileptic seizures prediction. Feature engineering code is also present in this codebase.

#### General project notes
Seizure prediction work-flow/pipeline:
- Features are extracted from raw EEG data (EDF) and are stored in *.dat* files,
- Features are loaded and annotated. For the annotation, first the segments from *.dat* files are loaded and further split in separate binary files (*.npy*) as preictal, ictal and interictal. Then, during the training the segments from binary files (only preictal and interictal) are loaded and are annotated on fly with 0 for interictal and 1 for preictal,
- Cross-validation (using [Stratified 5-fold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)) and hyper-parameters tunning (using [Bayesian Optimization](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)) happen and the chosen parameters are stored under *./CV_results/*.
- Training and evaluation can be performed using the stored parameteres from previous step. The training and evaluation is done using stratified 5-fold cross-validation and as result at the end the mean value from 5 cross-validation classifiers is reported.
    
#### Techincal details
- Linux (GPU)
- Python 3.7.2
- TensorFlow 2.0
- scikit-learn
- Pandas
etc..

#### Directories and files
- *./datasets/* : holds the raw EEG data (EDF files),
- *./data_features/* : holds the extracted features from raw EEG data, saved as *.dat* files. Extracted features (*univariate*, *SPLV*, *max_corr*) are saved in separate files, individually (*univariate.dat*, *SPLV.dat*, *max_correlation.dat*, etc),
- *./processed_datasets/* : holds the pre-processed features files. Features (segments) are split based on the state: preictal, ictal or interictal and are stored as binary files (*.npy*). This is the first step towards annotating the samples/segments, however still there is no class/label (0 or 1) attached to such segments. Files are organized into folders based on segmentation duration used, e.g. for *30 sec* duration, the directory *sec-30* is created,
- *./data_processing/* : holds two files that help on data pre-processing and loading:
  - *./data_processing/data_preparation.py*: loads the features files (*.dat*), splits the segments and stores in separate binary files (preictal, ictal and interictal) in directory *./processed_datasets/*,
  - *./data_processing/data_loader.py*: loads the data from binary files (*./processed_datasets/*) and annotates with 0 and 1. Ictal data are not used in prediction, therefore only preictal and interictal data are loaded. Furthermore, this file deals with grouping the N segments to form a sequence for training and evaluating sequence models such as LSTM and TCN,
- *./CV_results/* : holds the hyper-parameters selected from cross-validation step that are stored as *.json* files,
- *./helpers/* : holds some plotting helper functions to plot ROC AUC, data splits (k-fold vs stratified k-fold) and confusion matrix,
- *./images/* : holds all the images and plots of algorithms,
- *./models/* : holds the implementation of three neural networks FC, LSTM and TCN
- *./process_EEG.py* : loads the raw EEG data, calculates the features and stores the results in *.dat* files in directory *./data_features/*. This script currently works for [CHB-MIT](https://physionet.org/pn6/chbmit/) dataset only,
- *./process_EEG_Dog.py*: NOT COMPLETE. This does the feature extraction for iEEG dogs dataset,
- *./cv_model_selection.py* : runs the neural networks hyper-parameter selection/tuning using stratified 5-fold cross-validation and Bayesian Optimization. The results are saved in directory *./CV_results/*,
- *./train_eval_model.py*: trains and evaluates the neural networks and prints the final results. Plots are also generated from this scrip. 

Data preparation, model selection and training & evaluation are not explicitly run from the terminal, instead they are trigered from bash scripts: *./run_data_preparation.sh*, *./run_cv_model_selection.sh*, *./run_train_eval_model.sh*. E.g. to train a network run the following command in terminal "*./run_train_eval_model.sh*". Inside each *.sh* script there are some config parameters that should be considered before the running. Read the description of these parameteters in corresponding python (e.g. *train_eval_model.py*) files.

##### TODO
- Re-consider the Bayesian Optimization method for hyper-parameter tuning. The current implementation doesn't always provide the best parameters and there is an issue when using more than 8-10 iterations (check the approach details to get a better idea about the iterations)! I used [this](https://scikit-optimize.github.io/) implementation, however it needs to be carefully checked and get improved. Maybe using another approach such as [RandomSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) would be interesting,
- Refactor the code! The codebase is quite dirty, if the project is about to continue, an important step is to refactore the code,
- Decouple the code/logic of feature extraction and neural networks training & evaluation