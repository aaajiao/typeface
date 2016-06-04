import sys
sys.path.append('..')

import numpy as np
import scipy
import os
from os import listdir
from os.path import isfile, join
from time import time
from collections import Counter
import random
from PIL import Image
#from matplotlib import pyplot as plt

from lib.data_utils import shuffle
from lib.config import data_dir

def mnist():
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28*28)).astype(float)
    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000))
    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28*28)).astype(float)
    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000))
    trY = np.asarray(trY)
    teY = np.asarray(teY)
    return trX, teX, trY, teY

def mnist_with_valid_set():
    trX, teX, trY, teY = mnist()
    trX, trY = shuffle(trX, trY)
    vaX = trX[50000:]
    vaY = trY[50000:]
    trX = trX[:50000]
    trY = trY[:50000]
    return trX, vaX, teX, trY, vaY, teY
  
def get_image_vector(path, sizex, sizey):
  img = scipy.misc.imread(path)
  img = scipy.misc.imresize(img, [sizex, sizey, img.shape[2]], 'bicubic')
  if img.shape[2] == 4:
    img = img[:, :, 0:3]
  if len(img.shape) > 2:
    img = np.average(img, 2)
  img = img.reshape(sizex * sizey)
  return img

def get_hitor3c(num, size_x, size_y, force_include_classes):
  allX = []
  allY = []
  trX = []
  vaX = []
  teX = []
  trY = []
  vaY = []
  teY = []
  base_path = os.path.join(os.getcwd(),'data')
  new_class_idx = []
  for c in (range(num) + force_include_classes):
    class_idx = c
    if c in new_class_idx:
      class_idx = num + new_class_idx.index(c)
    elif c > num:
      new_class_idx.append(c)
      class_idx = num + new_class_idx.index(c)
    class_path = os.path.join(base_path,str(c))
    files = [f for f in listdir(class_path) if isfile(join(class_path, f))]
    print "class %d (index %d) : %d files " % (c, class_idx, len(files))
    for f in files:
      allX.append(get_image_vector(os.path.join(class_path,f), size_x, size_y))
      allY.append(class_idx)
  allX = np.asarray(allX)
  allY = np.asarray(allY)
  allX, allY = shuffle(allX, allY)
  numX = allY.shape[0]
  trX = allX[:int(numX*0.7)]
  vaX = allX[int(numX*0.7):int(numX*0.85)]
  teX = allX[int(numX*0.85):]
  trY = allY[:int(numX*0.7)]
  vaY = allY[int(numX*0.7):int(numX*0.85)]
  teY = allY[int(numX*0.85):]
  return trX, vaX, teX, trY, vaY, teY, (num+len(new_class_idx))
