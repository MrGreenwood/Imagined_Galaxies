#  Imagined Galaxies

The following code is an attempt to combine a set of improvements to the
original generative adversarial network, which are the covolutional GAN,
the Autoencoding GAN and the LSGAN, along with a variety of further changes
detailed in the code.

## Prerequisites

Install virtualenv

```
sudo apt-get install virtualenv
```

## Installation

First, clone the repository and cd to the project root, then create a new
environment and install the requirements.

```
virtualenv --python=/usr/bin/python2.7 venv
source venv/bin/activate
pip install -r requirements.txt
pip install kaggle
```

If you don't have unzip installed, install unzip.

```
sudo apt-get install unzip
```

Next, download and unzip the galaxy images.

```
kaggle competitions download -p ../ -f images_training_rev1.zip galaxy-zoo-the-galaxy-challenge --force
unzip ../images_training_rev1.zip -d ../
rm ../Images_training_rev1.zip
```

## Usage

Run the main script from terminal. Make sure that your source is still the virtual environment in the
current directory.

```
python Autoencoded_Gan.py
```

Optional arguments:

```
--learning_rate             Controls learning rate for all optimizers
--epochs                    Total number of training loops
--batch_size                Size of each batch (recommended 64+)
--cuda                      Whether a CUDA-enabled device will be used for training
--latent_dimension_size     Size of the latent dimension representing image samples
```

The script places sampled images and models into the results folder.