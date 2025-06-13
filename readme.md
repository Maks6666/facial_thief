# Real time face generator
Model, based on VAE, which is able to find and track human faces with YOLO and SORT and generate some copies of them 
with VAE custom model. 

## Generator
Generative part is based on variational autoencoder (VAE) model, where is possible to input an image
(human face in this case) and it will output a little bit changed version of it with some different 
features. Basic idea of VAE is, that unlike usual autoencoder it encodes data not into single vector, 
but into distribution of features, which could be defined with two values: 
μ - center (mean) value of each distribution and σ - standard deviation (or logarithm of the standard deviation), 
which defines the spread (variance) of each element in the distribution. Look for more 
details in 'vae-learning.ipynb' file. 



## Tracker
Program is able to track human faces with YOLOv11 and SORT algorithms. Running main.py file, 
program will start tracking. Tracked faces will be saved into 'generations' directory with a 
specific ID-number. When image is saved, it will be passed to model, which will generate some versions
of it. Amount of copies could be set before running. 



## To be continued... 