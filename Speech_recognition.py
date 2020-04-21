# -*- coding: utf-8 -*-

## importando librerias 
import os
import h5py
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import librosa as lr
import dask.array as da
import tensorflow as tf
import keras as k

from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import Dropout, Input, BatchNormalization
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils


import watermark

%reload_ext watermark
%watermark -a Christian_Arcos --iversion

"""
tensorflow 2.1.0
scipy      1.4.1
keras      2.3.1
h5py       2.10.0
watermark  2.0.2
pandas     1.0.3
numpy      1.18.1
Christian_Arcos
"""


#Hiperparametros y carga de datos

in_dim = (192,192,1)
out_dim = 176
batch_size = 32
mp3_path = '../'
tr_path = '../../data1/pol/'
#va_path = 'data/valid/'
te_path = '../../data/testingdata/'
data_size = 66176
tr_size = 52800
va_size = 4576
te_size = 8800