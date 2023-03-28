# Image Retrieval Repository

This repository contains the code for the Image Retrieval project I worked on during the last semester. The aim of this project was to experiment with various image retrieval techniques and compare their performance. The repository is organized into three main categories: 

1. Classical Methods with Global Features
2. Methods with Local Feature Descriptors (based on SIFT+ORB)
3. Methods with Neural Features (using BiT30k and EfficientNet models)

I am currently working on a blog post detailing my journey and experiences throughout this project. The post will provide more insights into the techniques and results, and it will be available soon. 

## Getting Started

To get started with the project, clone this repository and follow the installation instructions below.

### Prerequisites

Ensure that you have Python 3.x installed on your system.

## Repository Structure

The repository is structured as follows:

```
.
|-- global-features
|   |-- (Python files and notebooks for global features methods)
|-- local-features
|   |-- (Python files and notebooks for SIFT+ORB methods)
|-- neural-features
|   |-- (Python files and notebooks for BiT and EfficientNet models)
```

### Classical Methods with Global Features

This directory contains the code for classical image retrieval methods that use global features such as color histograms, texture, and shape. Additionally, it includes a painstakingly translated color auto correlogram code from MATLAB to Python, which was done by me. These methods are located in the `global-features` directory.

### Methods with Local Feature Descriptors

This directory contains the code for image retrieval methods based on local feature descriptors such as SIFT and ORB. These methods are located in the `local-features` directory.

### Methods with Neural Features

This directory contains the code for image retrieval methods using neural features extracted from pre-trained BiT30k and EfficientNet models. These methods are located in the `neural-features` directory.

## Datasets

For testing and evaluating the performance of all models, the following datasets were used:

1. Flickr30k Dataset
2. Buildings Dataset

Please note that these datasets are **not included** in the repository. If you wish to use them, you will need to obtain them on your own.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).