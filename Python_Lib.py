#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
import cv2
import os
import glob
import pathlib

# TensorFlow and Keras modules
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Flatten, Conv2D, Conv2DTranspose

# Brief explanation for each library or module
# NumPy: Numerical computing library for efficient array operations
# TensorFlow: Machine learning framework
# Matplotlib: Plotting library for data visualization
# ImageIO: Library for reading and writing images
# OpenCV: Computer vision library for image and video processing
# OS: Operating system interaction, used for file operations
# Glob: File path pattern matching
# Pathlib: Object-oriented file system paths

# TensorFlow and Keras modules
# - Sequential: Model composition tool for linear stack of layers
# - layers: Core layers for building neural network models
# - image_dataset_from_directory: Utility function for creating a dataset from image files in a directory
# - plot_model: Function to create a plot of a Keras model
# - Dense, Dropout, BatchNormalization, Activation, Flatten, Conv2D, Conv2DTranspose: Specific layers used in neural network architecture


# In[ ]:




