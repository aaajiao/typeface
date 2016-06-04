# - getting data
# - training data
# - saving models
# -------
# - loading models
# - make 1 big random
# - make 1 set of fakes + set of reals
# - make zloops
# - make interpolations
#   - make zloops
#   - test permutations
#   - variable speed + phase
#   - reverse v's from 0.5 (loops)
# - make interpolations w/ radicals
# -------
# - organize terminal + scripts

import sys
sys.path.append('..')

import os
import json
from PIL import Image
import pickle
import math
from time import time
import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib
from itertools import izip
import itertools

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from lib import activations
from lib import updates
from lib import inits
from lib.vis import grayscale_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data
from lib.metrics import nnc_score, nnd_score


# this directory
home_dir = os.getcwd()


# dataset parameters
npx = 112          # pixels width/height of images
npy = 112
ny = 10          # num classes (for the model that we are loading)

### learning parameters
k = 1             # num of discrim updates for each gen update
l2 = 2.5e-5       # l2 weight decay
b1 = 0.5          # momentum term of adam
nc = 1            # num of channels in image
nbatch = 128      # num of examples in batch
nz = 100          # num of dim for Z
ngfc = 1024       # num of gen units for fully connected layers
ndfc = 1024       # num of discrim units for fully connected layers
ngf = 64          # num of gen filters in first conv layer
ndf = 64          # num of discrim filters in first conv layer
nx = npx*npy*nc   # num of dimensions in X
lr = 0.00022       # initial learning rate for adam

# setup network
relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
bce = T.nnet.binary_crossentropy

gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)

npx_ = npx / 4

gw  = gifn((nz+ny, ngfc), 'gw')
gw2 = gifn((ngfc+ny, ngf*2*npx_*npx_), 'gw2')
gw3 = gifn((ngf*2+ny, ngf, 5, 5), 'gw3')
gwx = gifn((ngf+ny, nc, 5, 5), 'gwx')

dw  = difn((ndf, nc+ny, 5, 5), 'dw')
dw2 = difn((ndf*2, ndf+ny, 5, 5), 'dw2')
dw3 = difn((ndf*2*npx_*npx_+ny, ndfc), 'dw3')
dwy = difn((ndfc+ny, 1), 'dwy')

gen_params = [gw, gw2, gw3, gwx]
discrim_params = [dw, dw2, dw3, dwy]

def transform(X):
    return (floatX(X)/255.).reshape(-1, nc, npx, npy)

def inverse_transform(X):
    X = X.reshape(-1, npx, npy)
    return X

def gen(Z, Y, w, w2, w3, wx):
    yb = Y.dimshuffle(0, 1, 'x', 'x')
    Z = T.concatenate([Z, Y], axis=1)
    h = relu(batchnorm(T.dot(Z, w)))
    h = T.concatenate([h, Y], axis=1)
    h2 = relu(batchnorm(T.dot(h, w2)))
    h2 = h2.reshape((h2.shape[0], ngf*2, npx_, npx_))
    h2 = conv_cond_concat(h2, yb)
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2))))
    h3 = conv_cond_concat(h3, yb)
    x = sigmoid(deconv(h3, wx, subsample=(2, 2), border_mode=(2, 2)))
    return x

def discrim(X, Y, w, w2, w3, wy):
    yb = Y.dimshuffle(0, 1, 'x', 'x')
    X = conv_cond_concat(X, yb)
    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
    h = conv_cond_concat(h, yb)
    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2))))
    h2 = T.flatten(h2, 2)
    h2 = T.concatenate([h2, Y], axis=1)
    h3 = lrelu(batchnorm(T.dot(h2, w3)))
    h3 = T.concatenate([h3, Y], axis=1)
    y = sigmoid(T.dot(h3, wy))
    return y

def load_model(model_path):
  gen_params_values = joblib.load(model_path+'_gen_params.jl')
  for p, v in izip(gen_params, gen_params_values):
      p.set_value(v)
  discrim_params_values = joblib.load(model_path+'_discrim_params.jl')
  for p, v in izip(discrim_params, discrim_params_values):
      p.set_value(v)

X = T.tensor4()
Z = T.matrix()
Y = T.matrix()

gX = gen(Z, Y, *gen_params)

p_real = discrim(X, Y, *discrim_params)
p_gen = discrim(gX, Y, *discrim_params)

d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()
g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean()

d_cost = d_cost_real + d_cost_gen
g_cost = g_cost_d

cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

lrt = sharedX(lr)
d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
d_updates = d_updater(discrim_params, d_cost)
g_updates = g_updater(gen_params, g_cost)
updates = d_updates + g_updates

print 'COMPILING'
t = time()
_train_g = theano.function([X, Z, Y], cost, updates=g_updates)
_train_d = theano.function([X, Z, Y], cost, updates=d_updates)
_gen = theano.function([Z, Y], gX)
print '%.2f seconds to compile theano functions'%(time()-t)

def gen_samples(n, nbatch=128):
    samples = []
    labels = []
    n_gen = 0
    for i in range(n/nbatch):
        ymb = floatX(OneHot(np_rng.randint(0, ny, nbatch), ny))
        zmb = floatX(np_rng.uniform(-1., 1., size=(nbatch, nz)))
        xmb = _gen(zmb, ymb)
        samples.append(xmb)
        labels.append(np.argmax(ymb, axis=1))
        n_gen += len(xmb)
    n_left = n-n_gen
    ymb = floatX(OneHot(np_rng.randint(0, ny, n_left), ny))
    zmb = floatX(np_rng.uniform(-1., 1., size=(n_left, nz)))
    xmb = _gen(zmb, ymb)
    samples.append(xmb)    
    labels.append(np.argmax(ymb, axis=1))
    return np.concatenate(samples, axis=0), np.concatenate(labels, axis=0)

def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def get_z(zmb_, steps, indexes):
  indexes_ = []
  for i, index in enumerate(indexes):
    indexes_.append(index + i * steps)
  return zmb_[indexes_]

def gen_image(name, ymb, zmb, cols, indexes):
  path = '%s/%s.png' % (sample_dir, name)
  samples = np.asarray(_gen(zmb, ymb))
  grayscale_grid_vis(inverse_transform(samples[indexes]), (samples[indexes].shape[0]/cols, cols), path)
  return samples[indexes]

def gen_image_set(name, ymb, zmb, indexes):
  folder = '%s/%s_frames' % (sample_dir, name)
  if not os.path.exists(folder):
    os.makedirs(folder)
  samples = np.asarray(_gen(zmb, ymb))
  for i in range(len(indexes)):
    path = '%s/%04d.png' % (folder, i)
    grayscale_grid_vis(inverse_transform(samples[indexes[i]]), (1, 1), path)

def get_buffer_y(steps,num_buffer_samples=480,num_buffer_steps=3):
  num_buffer_rows = int(math.ceil(float(num_buffer_samples) / steps))
  targets = np.asarray([[int(round(i*steps+_/num_buffer_steps)) for _ in range(steps)] for i in range(num_buffer_rows)])
  ymb = floatX(OneHot(targets.flatten(), ny))
  return ymb

def get_buffer_z(steps,num_buffer_samples=480,num_buffer_steps=3):
  num_buffer_rows = int(math.ceil(float(num_buffer_samples) / steps))
  zmb = floatX(np_rng.uniform(-1., 1., size=(num_buffer_rows * steps, nz)))
  return zmb

def make_frac(steps, dim):
  frac = np.zeros(steps * dim).reshape(steps, dim)
  e = []
  for i in range(2 * dim):
    # this was supposed to make loops more interesting by interpolating each
    # z-dim by a different amount, but wasn't really working right so
    # now the exponent does nothing (set to 1)
    if np.random.random() > 0.5:
      #e.append(1 + 5 * np.random.random())
      e.append(1)
    else:
      #e.append(np.random.random())
      e.append(1)
  for j in range(steps):
    for k in range(dim):
      f = j / (steps - 1.0)
      if f > 0.5:
        f = np.power(2.0 * (1.0 - f), e[k])
      else:
        f = np.power(2.0 * f, e[dim + k])
      frac[j][k] = f
  return frac

def setup_z(zmb, offset, classes, numtargets, steps, start=None):
  if start == None:
    start = 0
  if numtargets == 1:
    z1 = zf[classes[0]][0]
    z2 = zf[classes[0]][1]
    frac = make_frac(steps, nz)
    for j in range(steps):
      for k in range(nz): 
        idx = offset + j
        f = frac[j][k]
        zmb[idx][k] = z1[k] * (1.0 - f) + z2[k] * f
  elif np.var(classes) == 0:
    for i in range(numtargets):
      c1 = classes[0]
      c2 = c1
      idxz1 = (start + i) % len(zf[c1])
      idxz2 = (start + 1 + i) % len(zf[c1])
      if i == numtargets-1:
        idxz2 = start
      print "interp %d, %d -> %d, %d" % (c1, idxz1, c2, idxz2)
      z1 = zf[c1][idxz1]
      z2 = zf[c2][idxz2]
      frac = make_frac(2*steps-1, nz)
      for j in range(steps):
        f = j / (steps - 1.0)
        for k in range(nz): 
          idx = offset + i*steps + j
          f = frac[j][k]
          zmb[idx][k] = z1[k] * (1.0 - f) + z2[k] * f
  else:
    for i in range(numtargets):
      idx1 = i
      idx2 = i+1
      c1 = classes[idx1]
      c2 = classes[idx2 % numtargets]
      idxz1 = start
      idxz2 = start
      print "interps %d, %d -> %d, %d" % (c1, idxz1, c2, idxz2)
      z1 = zf[c1][idxz1]
      z2 = zf[c2][idxz2]
      frac = make_frac(2*steps-1, nz)
      for j in range(steps):
        f = j / (steps - 1.0)
        for k in range(nz): 
          idx = offset + i*steps + j
          f = frac[j][k]
          zmb[idx][k] = z1[k] * (1.0 - f) + z2[k] * f
  return zmb

def gen_classes(name, steps, classes, interpolate=False, start=None):
  bymb = get_buffer_y(steps, num_buffer_classes, 3)  # dont know why but samples better when 
  bzmb = get_buffer_z(steps, num_buffer_classes, 3)  # included is a buffer of common classes
  offset = bymb.shape[0]
  numtargets = len(classes)
  targets = np.asarray([[classes[i] for _ in range(steps)] for i in range(numtargets)])
  ymb = floatX(OneHot(targets.flatten(), ny))
  zmb = floatX(np_rng.uniform(-1., 1., size=(numtargets * steps, nz)))
  ymb = np.vstack((bymb, ymb))
  zmb = np.vstack((bzmb, zmb))
  if interpolate:
    if numtargets > 1:
      for i in range(numtargets):
        y1 = classes[i]
        y2 = classes[(i+1) % numtargets]
        for j in range(steps):
          y = offset + steps * i + j
          ymb[y] = np.zeros(ny)
          if y1 == y2:
            ymb[y][y1] = 1.0
          else:
            ymb[y][y1] = 1.0 - j / (steps-1.0)
            ymb[y][y2] = j / (steps-1.0)
    zmb = setup_z(zmb, offset, classes, numtargets, steps, start)
  indexes = range(offset, ymb.shape[0])
  samples = gen_image(name, ymb, zmb, steps, indexes)
  gen_image_set(name, ymb, zmb, indexes)
  return ymb[offset:], zmb[offset:], samples

def gen_classes_random(name, steps, classes):
  ymb, zmb, samples = gen_classes(name, steps, classes, False)
  return ymb, zmb, samples

def gen_classes_interpolated(name, steps, classes, start=None):
  ymb, zmb, samples = gen_classes(name, steps, classes, True, start)
  return ymb, zmb, samples

def gen_classes_arithmetic(name, steps, classes, weights):
  bymb = get_buffer_y(steps, num_buffer_classes, 3)  # dont know why but samples better when 
  bzmb = get_buffer_z(steps, num_buffer_classes, 3)  # included is a buffer of common classes
  offset = bymb.shape[0]
  numtargets = len(classes)+1
  targets = np.asarray([[classes[i % (numtargets-1)] for _ in range(steps)] for i in range(numtargets)])
  ymb = floatX(OneHot(targets.flatten(), ny))
  zmb = floatX(np_rng.uniform(-1., 1., size=(numtargets * steps, nz)))
  ymb = np.vstack((bymb, ymb))
  zmb = np.vstack((bzmb, zmb))
  for i in range(numtargets):
    for j in range(steps):
      y_idx = offset + steps * i + j
      ymb[y_idx] = np.zeros(ny)
      if i == numtargets-1:
        for k, c in enumerate(classes):
          ymb[y_idx][c] = weights[k]
      else:
        ymb[y_idx][classes[i]] = 1.0
      frac = j / (steps-1.0)
      if frac > 0.5:
        frac = 2.0 * (1.0 - frac)
      else:
        frac = 2.0 * frac
      if (i == numtargets-1):
        z1 = zf[classes[0]][0]
        z2 = zf[classes[0]][1]
      else:
        z1 = zf[classes[i]][0]
        z2 = zf[classes[i]][1]
      for k in range(nz): 
        z = (1.0 - frac) * z1[k] + frac * z2[k]
        #z = min(z, z2 - z)
        zmb[y_idx][k] = z
  indexes = range(offset, ymb.shape[0])
  samples = gen_image(name, ymb, zmb, steps, indexes)
  gen_image_set(name, ymb, zmb, indexes)
  return ymb[offset:], zmb[offset:], samples


################################################################
################################################################
###### load our data


# this is where the samples will go
sample_dir = os.path.join(home_dir,'images')
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# load the model. there's a few in there, but the one i ended up using was
# 56x56 samples, learning rate 0.00022, 250 epochs (decay started after 150)
load_model(os.path.join(os.getcwd(),'models','500'))

# this is a hack to correct for the "dead-zone" in Z when generating
# samples, before i figured out how to fix it. when generating samples
# it generates a bunch of throw-away ones in a big buffer in the 
# beginning, and this improves the quality. otherwise quality seems low.
# there is a much better way to do this but i never got around to it :)
# if you are running out of memory generating large samples try lowering
# this number

num_buffer_classes = 10


# this dict holds our favorite z-values which are saved in the file zf.p
# these were hand-picked out of a bunch of random generations of characters
# as the ones that looked the best. when generating characters from class
# X, it will look for zf[X] to get the best values for z. you can just use
# random ones also, but the quality (in terms of how much the images 
# resemble actual characters) may suffer. 

zf = {}


# load our favorites for zf and a lookup table for character class indexes
# to the corresponding word in english.
# note that word_lut is a (very bad) translation because it is just pulled
# straight from google translate, and was never corrected. chinese 
# characters do not always (or even usually) translate straightforwardly
# to english words, especially single characters which refer to more abstract
# things, so don't rely on them for accuracy

zf = pickle.load(open(os.path.join(home_dir,"zf.p"), "rb" ))
word_lut = pickle.load( open(os.path.join(home_dir,"word_lut.p"), "rb"))



################################################################
################################################################
###### get favorites

def saveFavorites(zmb, classes, steps, indexes):
  if len(classes) != len(indexes):
    print("classes != indexes")
    return
  for i, c in enumerate(classes):
    zf[c] = []
    for k, idx in enumerate(indexes[i]):
      index = i * steps + idx
      zf[c].append(zmb[index])
  


# if you want to rewrite zf, you can do so in the following way:

# pick classes to rewrite favorites for
classes_to_test = [1,2,3,4,5,6,7,8,9]

# generate samples into a file images/test.png
steps = 24
ymb, zmb, samples = gen_classes_random('test', steps, classes_to_test)

# pick your N favorites for each row (e.g. 4 is the 5th image in test from the left). you need at least two to run some of the methods here
#indexes = [[19,9,28], [1,2,3], [5,6,8]]

# then save them to zf
#saveFavorites(zmb, classes, steps, indexes)



##############
## this will produce the first animation at the top of the page

# top = [6,32,10,21,4,11,27,15,8,5,17,29,19,7,25,12,15,13,18,23,6]
# gen_classes_interpolated("top_all", 20, top)


################################################################
################################################################
###### make sample matrix

# classes = [4,5,6,7,8,10,11,12,13,15,17,18,19,21,23,25,14,27,29,32]
# steps = 30
#ymb, matrixZ = gen_classes_random('matrixAll', steps, classes)
#
# if not os.path.exists(os.path.join(home_dir,'images/matrix')):
#     os.makedirs(os.path.join(home_dir,'images/matrix'))
#
#
# for i1,c1 in enumerate(classes):
#   for i2,c2 in enumerate(classes):
#     name_i = 'matrix/interp_%d_%d' % (c1, c2)
#     if c1 == c2:
#       steps_i = 120
#       gen_classes_interpolated(name_i, steps_i, [c1])
#     else:
#       val = np.random.random()
#       if val < 0.5:
#         steps_i = 60
#       else:
#         steps_i = 120
#       gen_classes_interpolated(name_i, steps_i, [c1, c2])



################################################################
################################################################
###### make bottom banner 
#
#
# # this is a bit hacky, but it will generate the very last image in the bottom of the page
#
# if not os.path.exists(os.path.join(home_dir,'images/banner')):
#     os.makedirs(os.path.join(home_dir,'images/banner'))
#
#
# steps = 7
# top=[6,32,10,21,4,11,27,15,8,5,17,29,19,7,25,12,15,13,18,23,6]
# classes = []
# for z in zf:
#   if z not in top and z not in [0]:
#     classes.append(z)
#
# for t in top:
#   classes.append(t)
#   classes.append(t)
#
# classes = top + np.ndarray.tolist(shuffle(classes))
#
# all_samples = []
# for i in range(0,16):
#   ymb, zmb, samples = gen_classes_interpolated("banner/banner%d"%i, steps, classes[10*i:10*i+11])
#   all_samples.append(samples[0:11*steps])
#
# all_samples = np.vstack(all_samples)
# grayscale_grid_vis(inverse_transform(all_samples[0:1200]), (30, 40), sample_dir+"banner.png")




# ################################################################
# ################################################################
# ###### interpolation series
#
# if not os.path.exists(os.path.join(home_dir,'images/interpolations')):
#     os.makedirs(os.path.join(home_dir,'images/interpolations'))
#
# if not os.path.exists(os.path.join(home_dir,'images/rad_interpolations')):
#     os.makedirs(os.path.join(home_dir,'images/rad_interpolations'))
#
#
# class_groups = {}
#
# # setup class groups
# class_groups['ymd1'] = {'name': 'ymd1', 'classes': [17,102,292,40] }
# class_groups['ymd2'] = {'name': 'ymd2', 'classes': [17,102,292,154] }
# class_groups['shehe'] = {'name': 'shehe', 'classes': [328,41] }
# class_groups['oldnew'] = {'name': 'oldnew', 'classes': [296,31] }
# class_groups['wantneed'] = {'name': 'wantneed', 'classes': [34,404] }
# class_groups['peoplecountry'] = {'name': 'peoplecountry', 'classes': [14,10] }
# class_groups['comego'] = {'name': 'comego', 'classes': [38,21] }
#
# #class_groups['redgreen'] = {'name': 'redgreen', 'classes': [56,467] }
# class_groups['largesmall'] = {'name': 'largesmall', 'classes': [101,13] }
# class_groups['malefemale'] = {'name': 'malefemale', 'classes': [444,167] }
# class_groups['beforeafter'] = {'name': 'beforeafter', 'classes': [53,37] }
# class_groups['recog_remem_learn'] = {'name': 'recog_remem_learn', 'classes': [144,138,32] }
# class_groups['reason_say'] = {'name': 'reason_say', 'classes': [67, 68] }
# class_groups['openclose'] = {'name': 'openclose', 'classes': [87,126] }
# class_groups['beijing_city_capital_country_world'] = {'name': 'beijing_city_capital_country_world', 'classes': [385,27,90,10,397] }
# class_groups['city_capital_country_world'] = {'name': 'city_capital_country_world', 'classes': [27,90,10,397] }
# class_groups['beijing_capital'] = {'name': 'beijing_capital', 'classes': [385,90] }
# class_groups['city_country'] = {'name': 'city_country', 'classes': [27,10] }
# class_groups['timecurrent'] = {'name': 'timecurrent', 'classes': [23,51] }
# class_groups['airwaterground'] = {'name': 'airwaterground', 'classes': [347,447,60] }
# class_groups['airgroundwater'] = {'name': 'airgroundwater', 'classes': [347,60,447] }
# class_groups['they_i_you_he_she'] = {'name': 'they_i_you_he_she', 'classes': [63,66,402,41,328] }
# class_groups['they_she_he'] = {'name': 'they_she_he', 'classes': [63,328,41] }
# class_groups['they_he_she'] = {'name': 'they_he_she', 'classes': [63,41,328] }
# class_groups['they_you_i'] = {'name': 'they_you_i', 'classes': [63,402,66] }
# class_groups['they_i_you'] = {'name': 'they_i_you', 'classes': [63,66,402] }
# class_groups['i_you_he_she'] = {'name': 'i_you_he_she', 'classes': [66,402,41,328] }
# class_groups['she_he_you_i'] = {'name': 'she_he_you_i', 'classes': [328,31,402,66] }
# class_groups['she_you_he_i'] = {'name': 'she_you_he_i', 'classes': [328,402,41,66] }
# class_groups['i_he_she_you'] = {'name': 'i_he_she_you', 'classes': [66,41,328,402] }
# class_groups['i_he_she'] = {'name': 'i_he_she', 'classes': [66,41,328] }
# class_groups['she_he_i'] = {'name': 'she_he_i', 'classes': [328,41,66] }
# class_groups['you_i'] = {'name': 'you_i', 'classes': [402,66] }
# class_groups['people1_culture_family'] = {'name': 'people1_culture_family', 'classes': [14,171,173] }
# class_groups['people2_culture_family'] = {'name': 'people2_culture_family', 'classes': [169,171,173] }
# class_groups['people1_culture'] = {'name': 'people1_culture', 'classes': [14,171] }
# class_groups['people2_culture'] = {'name': 'people2_culture', 'classes': [169,171] }
# class_groups['people1_family'] = {'name': 'people1_family', 'classes': [14,173] }
# class_groups['people2_family'] = {'name': 'people2_family', 'classes': [169,173] }
# class_groups['culture_family'] = {'name': 'culture_family', 'classes': [171,173] }
# class_groups['positive_negative'] = {'name': 'positive_negative', 'classes': [211,399] }
# class_groups['n1234'] = {'name': 'n1234', 'classes': [2,295,152,415] }
# class_groups['eye_face_body'] = {'name': 'eye_face_body', 'classes':[84,127,131] }
#
#
# # setup zloops
# for z in zf:
#   name = 'z_%s' % word_lut[z]
#   class_groups[name] = {'name': name, 'classes': [z,z] }
#
#
# # setup radical groups
# rad_09_classes = [14,20,41,47,63,128,179,250,274,289,301,340,341,356,430]
# rad_30_classes = [37,49,81,126,130,135,195,237,288,441]
#
# rad_09_0 = [14,179,63,356,301,47,274]
# rad_09_1 = [14,179,47,274,356,301,250]
# rad_09_2 = [14,356,301,63,47,274,179]
# rad_30_0 = [37,288,135]
# rad_30_1 = [37,135,288]
# rad_01_0 = [7,11,33,44]
# rad_01_1 = [11,33,7,44]
# rad_01_2 = [11,33,7,44]
# rad_class_groups = {}
# rad_class_groups['rad9_0'] = {'name': 'rad9_0', 'classes': rad_09_0 }
# rad_class_groups['rad9_1'] = {'name': 'rad9_1', 'classes': rad_09_1 }
# rad_class_groups['rad9_2'] = {'name': 'rad9_2', 'classes': rad_09_2 }
# rad_class_groups['rad30_0'] = {'name': 'rad30_0', 'classes': rad_30_0 }
# rad_class_groups['rad30_1'] = {'name': 'rad30_1', 'classes': rad_30_1 }
# rad_class_groups['rad01_0'] = {'name': 'rad01_0', 'classes': rad_01_0 }
# rad_class_groups['rad01_1'] = {'name': 'rad01_1', 'classes': rad_01_1 }
# rad_class_groups['rad01_2'] = {'name': 'rad01_2', 'classes': rad_01_2 }
#
#
# for c in class_groups:
#   print 'i_%s' % class_groups[c]['name']
#   steps_i = int(30 + np.random.random() * 10)
#   gen_classes_interpolated('interpolations/i_%s' % class_groups[c]['name'], steps_i, class_groups[c]['classes'], 0)
#
# for r in rad_class_groups:
#   print 'ri_%s' % rad_class_groups[r]['name']
#   steps_i = int(40 + np.random.random() * 10)
#   gen_classes_interpolated('rad_interpolations/ri_%s' % rad_class_groups[r]['name'], steps_i, rad_class_groups[r]['classes'], 0)
#
#
# ################################################################
# ################################################################
# ###### arithmetic
#
# # [king, male, female], [+1, -1, +1]
# gen_classes_arithmetic("arithmetic", 75, [357,444,167], [1,-1,1])


print "done generating samples!!! see the images folder"
