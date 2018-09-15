'''
Code of Mike Greenwood, please contact MGreenwood1994@gmail.com for questions or comments.

Relevant Papers:
Original GAN : https://arxiv.org/abs/1406.2661
Least Squares GAN: https://arxiv.org/abs/1611.04076
Convolutional GAN : https://arxiv.org/abs/1511.06434
Generative Adversarial Autoencoders : https://arxiv.org/abs/1511.05644
CoordConv : arxiv.org/pdf/1807.03247.pdf
SELU : arxiv.org/pdf/1706.02515.pdf

The following code is an attempt to combine a set of improvements to the
original generative adversarial network, which are the covolutional GAN,
the Autoencoding GAN and the LSGAN, along with a variety of further changes
made for training stability and imporoved representation. Convolution has a
tendency to improve the desired expressiveness for image and wave
outputs, while using autoencoding to improve training is similar to the
improvements found when a GAN has been supplied with ground truth labels in
addition to a randomly generated input state. Using autoencoding also
opens this project up to extension via the many improvements being made in
this area for more optimal unsupervised learning.
'''

import argparse
import torch as th
import torch.nn as nn
import numpy as np
import os
import copy
from scipy import misc
import random


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='number of epochs used in for training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate across optimizers')
parser.add_argument('--cuda', type=bool, default=True, help='Use of GPU-enabled device')
parser.add_argument('--latent_dimension_size', type=int, default=30, help='latent space dimensionality')
parser.add_argument('--sample_period', type=int, default=50, help='Epochs between samples')
opt = parser.parse_args()


IMAGE_SIZE = 432*432*5
RAW_IMAGE_SIDE_LENGTH = 424
IMAGE_SIDE_LENGTH = 432/3
IMAGE_CHANNELS = 3
IMAGE_SCALE_FACTOR = 8
DEVICE = th.device("cuda" if opt.cuda else "cpu")


class Reshape(nn.Module):
    '''
    A simple additional layer to translate between the dimensionality
    of inputs for convolutional and linear layers.
    '''
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


'''
AvgPool2d layers are used in place of MaxPool layers. A sparse gradient 
tends to be less stable for generative architectures. SELU is also used here 
in place of ReLU, again sparse gradients are not helpful, and the 
self-normalizing property is attractive for the stability of high-depth 
networks, an advantage over PrELU or Leaky ReLU. Shallow networks are 
usually recommended for generative networks due to instability with more layers. 
Batchnorm has been used throughout all convolutional layers as was used in 
the original convolutional GAN paper for cleaner error curves in training 
and less noisy outputs.
'''

'''
The encoder has the job of creating a representation of an image sample in 
LATENT dimensions, interpretable by the decoder.
'''
encoder = nn.Sequential(
    nn.Conv2d(5, 16, 3, stride=1, padding=1),
    nn.BatchNorm2d(16, 0.8),
    nn.SELU(),
    nn.AvgPool2d(2, stride=2),
    nn.Conv2d(16, 32, 3, stride=1, padding=1),
    nn.BatchNorm2d(32, 0.8),
    nn.SELU(),
    nn.AvgPool2d(2, stride=2),
    nn.Conv2d(32, 64, 3, stride=1, padding=1),
    nn.BatchNorm2d(64, 0.8),
    nn.SELU(),
    nn.AvgPool2d(2, stride=2),
    nn.Conv2d(64, 8, 3, stride=1, padding=1),
    nn.BatchNorm2d(8, 0.8),
    nn.SELU(),
    Reshape(opt.batch_size, -1),
    nn.Linear((IMAGE_SIDE_LENGTH**2)/8, opt.latent_dimension_size),
    nn.Tanh()
)

'''
The job of the decoder is to take a latent representation of an image in 
LATENT number of dimensions, and transform it into an approximation of the 
image the input represents. In this context it also has the job of learning 
to generate realistic-looking images from noise.
'''
decoder = nn.Sequential(
    nn.Linear(opt.latent_dimension_size, ((IMAGE_SIDE_LENGTH/IMAGE_SCALE_FACTOR)**2)*64),
    Reshape(opt.batch_size, 64, IMAGE_SIDE_LENGTH/IMAGE_SCALE_FACTOR, IMAGE_SIDE_LENGTH/IMAGE_SCALE_FACTOR),
    nn.BatchNorm2d(64),
    nn.Upsample(scale_factor=2),
    nn.Conv2d(64, 32, 3, stride=1, padding=1),
    nn.BatchNorm2d(32, 0.8),
    nn.SELU(),
    nn.Upsample(scale_factor=2),
    nn.Conv2d(32, 16, 3, stride=1, padding=1),
    nn.BatchNorm2d(16, 0.8),
    nn.SELU(),
    nn.Upsample(scale_factor=2),
    nn.Conv2d(16, 8, 3, stride=1, padding=1),
    nn.BatchNorm2d(8, 0.8),
    nn.SELU(),
    nn.Conv2d(8, IMAGE_CHANNELS, 3, stride=1, padding=1),
    nn.Tanh()
)

'''
The latent discriminator learns the difference between the output of the 
encoder and the generated noise. This discrimination is used to bias the 
autoencoding by pushing the outputs from a sample of random real images 
from the data towards resembling a LATENT dimensional vector of gaussian 
noise. 
'''
latent_discriminator = nn.Sequential(
    nn.Linear(opt.latent_dimension_size, 100),
    nn.Tanh(),
    nn.Linear(100, 100),
    nn.Tanh(),
    nn.Linear(100, 1),
    nn.Tanh()
)

'''
The image discriminator is trained to distinguish real images from images
constructed by the decoder from input noise. This is used to cause the generator
(the decoder) to smoothly represent realistic-looking galaxies from any 
realistically gaussian input by supplying gradients, as is the aim of any 
generative adversarial architecture.
'''
image_discriminator = nn.Sequential(
    nn.Conv2d(5, 16, 3, stride=1, padding=1),
    nn.BatchNorm2d(16, 0.8),
    nn.SELU(),
    nn.AvgPool2d(2, stride=2),
    nn.Conv2d(16, 32, 3, stride=1, padding=1),
    nn.BatchNorm2d(32, 0.8),
    nn.SELU(),
    nn.AvgPool2d(2, stride=2),
    nn.Conv2d(32, 64, 3, stride=1, padding=1),
    nn.BatchNorm2d(64, 0.8),
    nn.SELU(),
    nn.AvgPool2d(2, stride=2),
    nn.Conv2d(64, 8, 3, stride=1, padding=1),
    nn.BatchNorm2d(8, 0.8),
    nn.SELU(),
    Reshape(opt.batch_size, -1),
    nn.Linear((IMAGE_SIDE_LENGTH**2)/8, 100),
    nn.Tanh(),
    nn.Linear(100, 1),
    nn.Tanh()
)

'''
Pixelwise loss is defined here as the least squares error, as described in 
the LSGAN paper.
'''
adversarial_loss = nn.L1Loss()
pixelwise_loss = nn.MSELoss()

'''
For the sake of readability, some optimizers have been separated out for 
the parameters of different models. The image generation optimizer has been 
kept separate from the autoencoding optimizer for different momentum terms,
allowing generation optimizer to gain momentum without being impaired. 
'''
autoencoding_optimizer = th.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=opt.learning_rate)
image_generation_optimizer = th.optim.Adam(decoder.parameters(), lr=opt.learning_rate)
latent_discrimination_optimizer = th.optim.Adam(latent_discriminator.parameters(), lr=opt.learning_rate)
image_discrimination_optimizer = th.optim.Adam(image_discriminator.parameters(), lr=opt.learning_rate)


class GalaxyHandler:
    def __init__(self, location):
        self.image_location = location
        self.original_files_list = os.listdir(self.image_location)
        self.current_files = copy.copy(self.original_files_list)
        self.padding_layer = nn.ReflectionPad2d(4).to(DEVICE)
        self.downsizing_layer = nn.AvgPool2d(3, stride=3).to(DEVICE)

    '''
    A DEVICE-neutral function for taking a 3 channel vector of an RGB image in 
    the form of a torch tensor and adds the location as per CoordConv, a paper 
    from Uber AI I recommend strongly to machine learning specialists not 
    familiar with it, as this small change can make a great deal of difference.
    '''
    def add_image_locations(self, image):
        image_location = np.asarray([range(IMAGE_SIDE_LENGTH) for _ in range(IMAGE_SIDE_LENGTH)])
        image_location = np.stack((image_location, image_location.T), axis=0)
        image_location = image_location.astype(np.float16)/IMAGE_SIDE_LENGTH
        image_location = np.repeat(np.expand_dims(image_location, axis=0), opt.batch_size, 0)
        image = th.cat((image, th.Tensor(image_location).to(DEVICE)), dim=1)
        return image

    '''
    A simple function for retrieving real images from the image location and 
    transforming them into a useful format by sampling down to a smaller image 
    (width and height-wise), and adding image locations via add_image_locations().
    '''
    def galaxy_samples(self):
        sample_names = []
        for _ in range(opt.batch_size):
            if len(self.current_files) < 1:
                self.current_files = copy.copy(self.original_files_list)
            sample_names.append(self.current_files.pop(random.randint(0, len(self.current_files))))
        galaxies = [misc.imread(self.image_location + name) for name in sample_names]
        galaxies = np.asarray([np.swapaxes(galaxy, 0, 2) for galaxy in galaxies])
        galaxies = galaxies.astype(np.float16)/300
        galaxies = self.padding_layer.forward(th.Tensor(galaxies).to(DEVICE))
        galaxies = self.downsizing_layer.forward(galaxies)
        galaxies = self.add_image_locations(galaxies)
        return galaxies


'''
Instantiating and moving to optimal devices as necessary.
'''
image_handler = GalaxyHandler(location='../Images/')
encoder.to(DEVICE)
decoder.to(DEVICE)
latent_discriminator.to(DEVICE)
image_discriminator.to(DEVICE)
log = []


'''
Setting up directory structure for results.
'''


def check_directory(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

check_directory('samples')
check_directory('samples/images')
check_directory('samples/models')


'''
Model training loop.
'''


for epoch in range(opt.epochs):
    '''
    First, we take a sample of real images, and set up our definitions for what 
    denotes a real image, and a fake image.
    '''
    real_galaxies = image_handler.galaxy_samples()
    valid = th.ones(()).new_full(size=[real_galaxies.shape[0]], fill_value=0.5, requires_grad=False)
    fake = th.ones(()).new_full(size=[real_galaxies.shape[0]], fill_value=-0.5, requires_grad=False)

    '''
    Second, we cover the autoencoding optimization. As with all further machine 
    learning optimization steps, we start by zeroing the gradient of the 
    optimizer, and finish by calculating the error wrt. our optimizers 
    parameters by calling backward(), then moving our parameters in the 
    direction of the sum of our gradients with step(). Here, the sample of 
    images is used to create a latent encoded representation, which is used to 
    attempt to reconstruct the original galaxies. 
    
    The difference between the original image and the reconstruction is minimized. 
    The difference here from ordinary autoencoding is the addition of the latent 
    discriminator to force the latent encoding to resemble an unpredictable 
    random gaussian sample. The influence of the latent discriminator has been 
    left at three orders of magnitude less than the error from image 
    reconstruction as in the original generative adversarial autoencoder paper.
    '''
    autoencoding_optimizer.zero_grad()
    latent_galaxies = encoder(real_galaxies)
    reconstructed_galaxies = decoder(latent_galaxies)
    is_latent_real = latent_discriminator(latent_galaxies)
    reconstruction_error = pixelwise_loss(reconstructed_galaxies, real_galaxies[:, :3])
    autoencoding_error = 0.001 * adversarial_loss(is_latent_real.view(-1), valid) + \
                         0.999 * reconstruction_error
    autoencoding_error.backward(retain_graph=True)
    autoencoding_optimizer.step()

    '''
    Third, we train latent vector discrimination. A random gaussian sample is 
    taken in the shape we desire, z. We minimize the error of the estimates of 
    the discriminator for the real distribution z, then minimize the error of 
    the estimates of the falsified galaxies latent_galaxies. If the vector 
    latent_galaxies regularly shows patterns not expected in z, 
    latent_discriminator will use this to learn to label those patterns as 
    closer to invalid. As is often recommended, for discriminator training, 
    minibatches are kept exclusively either valid examples or invalid examples 
    with no shuffling.
    '''
    latent_discrimination_optimizer.zero_grad()
    z = th.Tensor(np.random.normal(0, 0.2, (real_galaxies.shape[0], opt.latent_dimension_size))).to(DEVICE)
    real_latent_error = adversarial_loss(latent_discriminator(z).view(-1), valid) * 0.5
    real_latent_error.backward(retain_graph=True)
    latent_discrimination_optimizer.step()
    latent_discrimination_optimizer.zero_grad()
    fake_latent_error = adversarial_loss(latent_discriminator(latent_galaxies).view(-1), fake) * 0.5
    fake_latent_error.backward(retain_graph=True)
    latent_discrimination_optimizer.step()
    latent_discrimination_error = real_latent_error + fake_latent_error

    '''
    Fourth, we train the image discrimination in a similar way. Errors on 
    estimates of the validity of real images is minimized separately from the 
    error on the estimates of the invalidity of images generated by the decoder 
    from z.
    '''
    image_discrimination_optimizer.zero_grad()
    real_image_error = adversarial_loss(image_discriminator(real_galaxies).view(-1), valid) * 0.5
    real_image_error.backward(retain_graph=True)
    image_discrimination_optimizer.step()
    image_discrimination_optimizer.zero_grad()
    generated_image = decoder(z)
    generated_image_error = adversarial_loss(image_discriminator(image_handler.add_image_locations(
        generated_image)).view(-1), fake) * 0.5
    generated_image_error.backward(retain_graph=True)
    image_discrimination_optimizer.step()
    image_discrimination_error = real_image_error + generated_image_error

    '''
    Fifth the generator (the decoder) is trained to generate more realistic 
    images as far as the image discriminator is concerned. Fake images are 
    created from z by the decoder, and the error with respect to the image is 
    minimized, with the error of the generated image being given via backprop 
    through the image discriminator. Note that here, a fake image is being 
    generated, and a valid label is being used, as here the parameters of the 
    decoder is being optimized towards creating images more likely to be 
    labelled as valid.
    '''
    image_generation_optimizer.zero_grad()
    generated_image_error = adversarial_loss(image_discriminator(image_handler.add_image_locations(
        decoder(z))).view(-1), valid)
    generated_image_error.backward()
    image_generation_optimizer.step()

    log.append('Reconstruction error:, %.5f, ' % reconstruction_error.item() +
               'Latent discrimination error:, %.5f, ' % latent_discrimination_error.item() +
               'Image discrimination error:, %.5f, ' % image_discrimination_error.item() +
               'False image error:, %.5f' % generated_image_error.item())
    print(log[-1])

    '''
    Samples of false images and parameters are saved to the samples folder periodically.
    '''
    if epoch % opt.sample_period == 0:
        misc.imwrite('samples/images/sample%06d.jpg' % epoch, th.transpose(generated_image[0], 0, 2).numpy())
        encoder.save_state_dict('samples/models/encoder_%06d.pt' % epoch)
        decoder.save_state_dict('samples/models/decoder_%06d.pt' % epoch)
