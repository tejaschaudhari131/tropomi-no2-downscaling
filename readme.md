# Satellite-Based Air Quality Downscaling using Hybrid Deep Learning

This repository contains the code and data for a research project focused on downscaling satellite-based air quality data using a hybrid deep learning approach.  The goal is to improve the spatial resolution of air quality estimations from coarse-resolution satellite data, enabling more accurate assessments of pollution levels at finer scales, particularly at the local level.

## Project Overview

Air pollution is a significant global health concern, and accurate monitoring at high spatial resolution is crucial for effective public health interventions and policy decisions.  While satellite remote sensing offers a large-scale view of air pollution, its inherent limitations in spatial resolution hinder its applicability at local levels where exposure and health effects are most pronounced.  This project addresses this limitation by developing and applying a hybrid deep learning model to downscale satellite-based air quality data.

This hybrid model combines the strengths of Convolutional Neural Networks (CNNs) for spatial feature extraction and Recurrent Neural Networks (RNNs) for capturing temporal dynamics.  The model integrates multiple data sources to enhance accuracy and robustness, including:

* **Satellite Data:** [Specify satellite mission(s) and sensor(s), e.g., Sentinel-5P TROPOMI, MODIS].  Data includes [specify parameters, e.g., NO2 column density, AOD, CO].
* **Ground-Based Measurements:** Data from monitoring stations, providing ground-truth values for training and validation.  [Specify location(s) and parameters].
* **Meteorological Data:**  [Specify data sources, e.g., ERA5 reanalysis, local weather stations].  Includes parameters such as wind speed, direction, temperature, humidity, and precipitation.
* **Land Use/Land Cover Data:**  [Specify data source, e.g., Corine Land Cover, Landsat].  Provides information on urban/rural areas, vegetation, and other land cover types.


## Model Architecture

The core of the downscaling method is a hybrid deep learning architecture.

* **CNN:** A Convolutional Neural Network is used to extract spatial features from the satellite imagery, meteorological data, and land-use data. The CNN learns relevant spatial patterns and correlations.

* **RNN:** A Recurrent Neural Network (e.g., LSTM or GRU) is employed to model the temporal dynamics of air pollution, integrating the time-series information from the meteorological data and potentially ground-based measurements.

* **Fusion:** The outputs of the CNN and RNN are combined (e.g., concatenation or a weighted average) to generate the final high-resolution air quality predictions.

## Data

The data used in this project is available [Specify data access method:  e.g., linked in this repository, available upon request].  This includes:

* **Satellite Data:**  [Provide details on data preprocessing steps, data formats, and file locations].
* **Ground-Based Measurements:** [Provide details on data cleaning, quality control, and file locations].
* **Meteorological Data:** [Provide details on data preprocessing, data formats, and file locations].
* **Land Use/Land Cover Data:** [Provide details on data resolution, projection, and file locations].


## Code

The code is primarily written in Python and uses the following libraries:

* [List libraries used, e.g., TensorFlow/PyTorch, NumPy, Pandas, scikit-learn, rasterio, geopandas].

The main code files are organized as follows:

* `data_processing.py`:  Handles data loading, preprocessing, and cleaning.
* `model.py`:  Defines the CNN-RNN architecture and training process.
* `training.py`:  Runs the model training and evaluation.
* `downscaling.py`:  Applies the trained model to downscale the satellite data.

## Results

[Include a summary of the results, including key findings, figures, and tables. Discuss the performance metrics used (e.g., RMSE, R², MAE). You might include visual comparisons between downscaled results and ground measurements.]


## Future Work

Future improvements and extensions of this research may include:

* Incorporating additional data sources (e.g., traffic data, emission inventories).
* Evaluating the model’s performance in different geographic regions and under varied meteorological conditions.
* Exploring different deep learning architectures and hyperparameter optimization techniques.
* Developing an operational downscaling system for real-time air quality monitoring.

## License



## Contact
tejaschaudhari131@gmail.com
