### DCGAN trained on HIT-OR3C (handwritten Chinese characters)

Original code [here](https://github.com/Newmu/dcgan_code). Training code is mostly unchanged.

To generate all images found [here](genekogan.com/works/a-book-from-the-sky.html) and more, run the following:
  
  cd dcgan_code/HITOR3C/
  THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True' python generate.py

It will take around 10 minutes (mostly from generating the 20x20 interpolations matrix from the bottom of the page), and all images will be placed into the images directory.

To retrain the dataset:

  THEANO_FLAGS='floatX=float32,device=gp
u0,nvcc.fastmath=True' python train.py
