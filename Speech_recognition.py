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
import dask.array.image as dai
import tensorflow as tf
import keras as k
import imageio

from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import Dropout, Input, BatchNormalization
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils


import watermark

#%reload_ext watermark
#%watermark -a Christian_Arcos --iversion

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
tr_path = '..\\..\\data\\trainingdata\\Training\\'
#va_path = 'data/valid/'
te_path = '../../data/testingdata/'
data_size = 8964
tr_size = 5964
va_size = 1000
te_size = 2000



def mp3_to_img(path, height=192, width=192):
    """
    función que convierte cada archivo mp3 en un espectograma

    Parameters
    ----------
    path : carpeta con los archivos mp3.
    height : dimensión del alto de cada imágen
    with : dimensión del ancho de cada imagen del espectrograma.

    Returns
    -------
    imagenes espectrales en escala mel

    """

    signal, sr = lr.load(path, res_type='kaiser_fast') #cargar los audios como waveform 'signal' y su tasa de muestreo como sr
    
    # Recortar el audio en un 5% al inicio y fin de cada uno, para evitar los silencios
    hl = signal.shape[0]//(width*1.1)
    spec = lr.feature.melspectrogram(signal, n_mels=height, hop_length=int(hl))
    img = lr.amplitude_to_db(spec)**2
    start = (img.shape[1] - width) // 2
    return img[:, start:start+width]    

def process_audio(in_folder, out_folder):
    """
    Procesamiento del dataset

    Parameters
    ----------
    in_folder : carpeta qeu contiene los archivos mp3
        
    out_folder : carpeta donde se almacenara las imagenes espectrales en la escala mel
    """
    os.makedirs(out_folder, exist_ok=True)
    files = glob.glob(in_folder+'*.mp3')
    
    start = len(in_folder)
    for file in files:
        print(file)
        img = mp3_to_img(file)
        img_uint8 = img.astype(np.uint8)
        imageio.imsave(out_folder + file[start:] + '.jpg', img_uint8)
        
def process_audio_with_classes(in_folder, out_folder, labels):
    """
    Esta función usara las etiquetas para clasificar todas las imagenes jpg en subcarpetas correspondientes 
    """
    os.makedirs(out_folder, exist_ok=True)
    for i in range(len(labels['Sample Filename'])):
        file = labels['Sample Filename'][i]
        lang = labels['Language'][i]
        os.makedirs(out_folder + lang, exist_ok=True)
        img = mp3_to_img(in_folder+file)
        sp.misc.imsave(out_folder + lang + '/' + file + '.jpg', img)
        

def jpgs_to_h5(source, target, name):
    """
    Convierte un directorio de imágenes en un archivo HDF5 que almacena las imágenes en una matriz que tiene forma
    (img_num, height.width, [channels])
    """
    dai.imread(source + '*.jpg').to_hdf5(target, name)

#convertir los audios a imagenes        
process_audio(tr_path, 'data/jpg/')


#convertir la carpeta de imagenes en un contenedor compacto
jpgs_to_h5('data/jpg/', 'data/data.h5', 'data')

#mezclando los datos para entrenamiento test y validación
y = pd.read_csv('data/train_list.csv')['Language']
y = pd.get_dummies(y)
y = y.reindex_axis(sorted(y.columns), axis=1)
y = y.values
y = da.from_array(y, chunks=1000)
y

x = h5py.File('data/data.h5')['data']
x = da.from_array(x, chunks=1000)
x

shfl = np.random.permutation(data_size)

tr_idx = shfl[:tr_size]
va_idx = shfl[tr_size:tr_size+va_size]
te_idx = shfl[tr_size+va_size:]

x[tr_idx].to_hdf5('data/x_tr.h5', 'x_tr')
y[tr_idx].to_hdf5('data/y_tr.h5', 'y_tr')
x[va_idx].to_hdf5('data/x_va.h5', 'x_va')
y[va_idx].to_hdf5('data/y_va.h5', 'y_va')
x[te_idx].to_hdf5('data/x_te.h5', 'x_te')
y[te_idx].to_hdf5('data/y_te.h5', 'y_te')


#cargando y procesando los datos
x_tr = da.from_array(h5py.File('data/x_tr.h5')['x_tr'], chunks=1000)
y_tr = da.from_array(h5py.File('data/y_tr.h5')['y_tr'], chunks=1000)
print(x_tr.shape, y_tr.shape)

x_va = da.from_array(h5py.File('data/x_va.h5')['x_va'], chunks=1000)
y_va = da.from_array(h5py.File('data/y_va.h5')['y_va'], chunks=1000)
print(x_va.shape, y_va.shape)

x_te = da.from_array(h5py.File('data/x_te.h5')['x_te'], chunks=1000)
y_te = da.from_array(h5py.File('data/y_te.h5')['y_te'], chunks=1000)
print(x_te.shape, y_te.shape)

x_tr /= 255.
x_va /= 255.
x_te /= 255.

#muestra
# test_img = x_tr[0, :, :, 0]
# plt.imshow(test_img)
# plt.show()

#Creando el modelo
i = Input(shape=in_dim)
m = Conv2D(16, (3, 3), activation='elu', padding='same')(i)
m = MaxPooling2D()(m)
m = Conv2D(32, (3, 3), activation='elu', padding='same')(m)
m = MaxPooling2D()(m)
m = Conv2D(64, (3, 3), activation='elu', padding='same')(m)
m = MaxPooling2D()(m)
m = Conv2D(128, (3, 3), activation='elu', padding='same')(m)
m = MaxPooling2D()(m)
m = Conv2D(256, (3, 3), activation='elu', padding='same')(m)
m = MaxPooling2D()(m)
m = Flatten()(m)
m = Dense(512, activation='elu')(m)
m = Dropout(0.5)(m)
o = Dense(out_dim, activation='softmax')(m)

model = Model(inputs=i, outputs=o)
model.summary()


model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=1e-3), metrics=['accuracy'])
model.fit(x_tr, y_tr, epochs=2, verbose=1, validation_data=(x_va, y_va))

# model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=1e-4), metrics=['accuracy'])
# model.fit(x_tr, y_tr, epochs=3, verbose=1, validation_data=(x_va, y_va))

# model = load_model('speech_v9.h5')

model.evaluate(x_te, y_te)