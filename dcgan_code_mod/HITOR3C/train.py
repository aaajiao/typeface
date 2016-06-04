import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from tqdm import tqdm
#from matplotlib import pyplot as plt
from sklearn.externals import joblib
from itertools import izip

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

from load import get_hitor3c


# parameters
npx = 112          # pixels width/height of images
npy = 112
ny = 10            # num classes
force_include_classes = []


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
niter = 100       # num of iter at starting learning rate
niter_decay = 401 # num of iter to linearly decay learning rate to zero
lr = 0.00022       # initial learning rate for adam

# load data
trX, vaX, teX, trY, vaY, teY, ny = get_hitor3c(ny, npx, npy, force_include_classes)
vaX = floatX(vaX)/255.
ntrain, nval, ntest = len(trX), len(vaX), len(teX)
print "size of training set: %d" % len(trX)

def transform(X):
    return (floatX(X)/255.).reshape(-1, nc, npx, npy)

def inverse_transform(X):
    X = X.reshape(-1, npx, npy)
    return X


model_dir = 'models/'
samples_dir = 'samples/'

if not os.path.exists('logs/'):
    os.makedirs('logs/')

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

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
  gen_params_values = joblib.load(model_path + '_gen_params.jl')
  for p, v in izip(gen_params, gen_params_values):
      p.set_value(v)
  discrim_params_values = joblib.load(model_path + '_discrim_params.jl')
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

cols = 10
tr_idxs = np.arange(len(trX))
trX_vis = np.asarray([[trX[i] for i in py_rng.sample(tr_idxs[trY==y], cols)] for y in range(ny)]).reshape(ny * cols, -1)
trX_vis = inverse_transform(transform(trX_vis))
grayscale_grid_vis(trX_vis, (ny, cols), 'samples/test.png')


############
# set up targets normally
steps = 6
numtargets = 9 #This is how many letter you will count
start = 1
targets = np.asarray([[i+start for _ in range(steps)] for i in range(numtargets)])
sample_ymb = floatX(OneHot(targets.flatten(), ny))

# set up random z
sample_zmb = floatX(np_rng.uniform(-1., 1., size=(numtargets * steps, nz)))



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

f_log = open('logs/log.ndjson', 'wb')
log_fields = ['n_epochs', 'n_updates', 'n_examples', 'n_seconds','1k_va_nnc_acc', '10k_va_nnc_acc', '100k_va_nnc_acc','1k_va_nnd','10k_va_nnd','100k_va_nnd','g_cost','d_cost']


n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()
for epoch in range(1, niter+niter_decay+1):
    trX, trY = shuffle(trX, trY)
    for imb, ymb in tqdm(iter_data(trX, trY, size=nbatch), total=ntrain/nbatch):
        imb = transform(imb)
        ymb = floatX(OneHot(ymb, ny))
        zmb = floatX(np_rng.uniform(-1., 1., size=(len(imb), nz)))
        if n_updates % (k+1) == 0:
            cost = _train_g(imb, zmb, ymb)
        else:
            cost = _train_d(imb, zmb, ymb)
        n_updates += 1
        n_examples += len(imb)
    if (epoch-1) % 5 == 0 & False:
        g_cost = float(cost[0])
        d_cost = float(cost[1])
        gX, gY = gen_samples(100000)
        gX = gX.reshape(len(gX), -1)
        va_nnc_acc_1k = nnc_score(gX[:1000], gY[:1000], vaX, vaY, metric='euclidean')
        va_nnc_acc_10k = nnc_score(gX[:10000], gY[:10000], vaX, vaY, metric='euclidean')
        va_nnc_acc_100k = nnc_score(gX[:100000], gY[:100000], vaX, vaY, metric='euclidean')
        va_nnd_1k = nnd_score(gX[:1000], vaX, metric='euclidean')
        va_nnd_10k = nnd_score(gX[:10000], vaX, metric='euclidean')
        va_nnd_100k = nnd_score(gX[:100000], vaX, metric='euclidean')
        log = [n_epochs, n_updates, n_examples, time()-t, va_nnc_acc_1k, va_nnc_acc_10k, va_nnc_acc_100k, va_nnd_1k, va_nnd_10k, va_nnd_100k, g_cost, d_cost]
        print '%.0f %.2f %.2f %.2f %.4f %.4f %.4f %.4f %.4f'%(epoch, va_nnc_acc_1k, va_nnc_acc_10k, va_nnc_acc_100k, va_nnd_1k, va_nnd_10k, va_nnd_100k, g_cost, d_cost)
        f_log.write(json.dumps(dict(zip(log_fields, log)))+'\n')
        f_log.flush()
    n_epochs += 1
    print "done epoch "+str(n_epochs);
    if (-1+n_epochs) % 10 == 0:
      samples = np.asarray(_gen(sample_zmb, sample_ymb))
      grayscale_grid_vis(inverse_transform(samples), (samples.shape[0] / steps, steps), 'samples/%d.png'%(n_epochs))
    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
    if n_epochs in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
        joblib.dump([p.get_value() for p in gen_params], 'models/%d_gen_params.jl'%(n_epochs))
        joblib.dump([p.get_value() for p in discrim_params], 'models/%d_discrim_params.jl'%(n_epochs))
    



#load_model('/root/dcgan_code/mnist/models/cond_dcgan/242_lr2') 
#samples = np.asarray(_gen(sample_zmb, sample_ymb))
#grayscale_grid_vis(inverse_transform(samples), (samples.shape[0]/steps, steps), 'samples/%s/hi2%d.png'%(n_epochs))

  












