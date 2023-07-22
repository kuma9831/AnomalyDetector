import mxnet as mx
from mxnet import gluon
import utils
import glob
import scipy
from scipy import signal
from matplotlib import pyplot as plt
from keras.models import model_from_json
from matplotlib import colors
import glob
import numpy as np
import os
from PIL import Image

class ConvolutionalAutoencoder(gluon.nn.HybridBlock):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        with self.name_scope():
            self.encoder = gluon.nn.HybridSequential(prefix="")
            with self.encoder.name_scope():
                self.encoder.add(gluon.nn.Conv2D(32, 5, padding=0, activation='relu'))
                self.encoder.add(gluon.nn.MaxPool2D(2))
                self.encoder.add(gluon.nn.Conv2D(32, 5, padding=0, activation='relu'))
                self.encoder.add(gluon.nn.MaxPool2D(2))
                self.encoder.add(gluon.nn.Dense(2000))
            self.decoder = gluon.nn.HybridSequential(prefix="")
            with self.decoder.name_scope():
                self.decoder.add(gluon.nn.Dense(32*22*22, activation='relu'))
                self.decoder.add(gluon.nn.HybridLambda(lambda F, x: F.UpSampling(x, scale=2, sample_type='nearest')))
                self.decoder.add(gluon.nn.Conv2DTranspose(32, 5, activation='relu'))
                self.decoder.add(gluon.nn.HybridLambda(lambda F, x: F.UpSampling(x, scale=2, sample_type='nearest')))
                self.decoder.add(gluon.nn.Conv2DTranspose(1, kernel_size=5, activation='sigmoid'))


    def hybrid_forward(self, F, x):
        x = self.encoder(x)
        x = self.decoder[0](x)
        x = x.reshape((-1,32,22,22))
        x = self.decoder[1:](x)

        return x

def train(batch_size, ctx, num_epochs, path, lr=1e-4, wd=1e-5, params_file="autoencoder_ucsd_convae.params"):

  dataloader = utils.create_dataset(path, batch_size, shuffle=True)

  model = ConvolutionalAutoencoder()
  model.hybridize()

  model.collect_params().initialize(mx.init.Xavier('gaussian'), ctx=ctx)
  
  l2loss = gluon.loss.L2Loss()
  optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})

  for epoch in range(num_epochs):
    for image in dataloader:
        image = image.as_in_context(ctx)

        with mx.autograd.record():
            reconstructed = model(image)
            loss = l2loss(reconstructed, image)

            loss.backward()
            optimizer.step(batch_size)
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, mx.nd.mean(loss).asscalar()))

  model.save_parameters(params_file)
  return model, params_file