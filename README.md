# Speech_Recognition
Reconocimiento y clasificación de audio usando CNNs

En este repositorio se entrena una red convolucional para la clasificación de archivos de aúdio de diferentes idiomas.
El conjunto de datos utilizado contiene 65.000 archivos con 176 idiomas y esta disponible para descargar enla pagína de competiciones 
TopCoder (https://goo.gl/G5XBJl).

##Problemas con librosa
Despues de instalar librosa en un ambiente anaconda es posible que se presenten algunos problemas para cargar archivos o incluso
para importar la libreria algunos de ellos son:

1 -> problemas relacionados con scipy
la solucion es usar pip install scipy e instalarlo de nuevo con pip install scipy

2 -> problema relacionado con el metodo load de librosa.
Si se usa un ambiente anaconda en windows es necesario instalar ffmpeg para poder cargar archivos mp3
conda install -c conda-forge ffmpeg
O busque su sistema operativo en la pagina oficial de librosa 
https://github.com/librosa/librosa#audioread
