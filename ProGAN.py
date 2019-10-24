
import argparse
import yaml
import numpy as np
import os
import time
import datetime
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from skimage.io import imsave

from layers import EqualizedConv2d, EqualizedDeconv2d, PixelwiseNorm, MinibatchStdDev
from utils import ImageDataset, Logger, update_average
import losses


class Generator(nn.Module):
    """ Generator of the GAN network """

    def __init__(self, depth=6, channel_size=512, latent_dim=128):
        """
        constructor for the Generator class
        :param depth: required depth of the Network
        :param latent_size: size of the latent manifold
        :param use_eql: whether to use equalized learning rate
        """

        super(Generator, self).__init__()

        # state of the generator:
        self.depth = depth
        self.channel_size = channel_size

        # register the modules required for the GAN
        self.initial_block = GenInitialBlock(latent_dim, channel_size)

        # create a module list of the other required general convolution blocks
        self.layers = nn.ModuleList([])  # initialize to empty list

        # create the ToRGB layers for various outputs:
        self.rgb_converters = nn.ModuleList([EqualizedConv2d(channel_size, 3, (1, 1), bias=False)])

        # create the remaining layers
        for _ in range(self.depth - 1):

            layer = GenGeneralConvBlock(channel_size, channel_size//2)
            #rgb = self.toRGB(latent_size//2)
            rgb = EqualizedConv2d(channel_size//2, 3, (1, 1), bias=False)
            if channel_size > 32:
                channel_size = channel_size // 2
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

        # register the temporary upsampler
        self.temporaryUpsampler = lambda x: F.interpolate(x, scale_factor=2)
        
        self.final_output = torch.nn.Tanh()

    def forward(self, x, depth, alpha):
        """
        forward pass of the Generator
        :param x: input noise
        :param depth: current depth from where output is required
        :param alpha: value of alpha for fade-in effect
        :return: y => output
        """

        assert depth < self.depth, "Requested output depth cannot be produced"

        y = self.initial_block(x)

        if depth > 0:
            for block in self.layers[:depth - 1]:
                y = block(y)

            residual = self.rgb_converters[depth - 1](self.temporaryUpsampler(y))
            straight = self.rgb_converters[depth](self.layers[depth - 1](y))
            out = alpha * torch.tanh(straight) + (1 - alpha) * torch.tanh(residual)

        else:
            out = torch.tanh(self.rgb_converters[0](y))

        return out

class Discriminator(nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, height=6, feature_size=512):
        """
        constructor for the class
        :param height: total height of the discriminator (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param use_eql: whether to use equalized learning rate
        """

        super(Discriminator, self).__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if height >= 4:
            assert feature_size >= np.power(2, height - 4), "feature size cannot be produced"

        # create state of the object
        self.height = height
        self.feature_size = feature_size

        self.final_block = DisFinalBlock(self.feature_size)

        # create a module list of the other required general convolution blocks
        self.layers = nn.ModuleList([])  # initialize to empty list

        # create the fromRGB layers for various inputs:

        self.rgb_to_features = nn.ModuleList([EqualizedConv2d(3, self.feature_size, 1, bias=False)])

        # create the remaining layers
        for _ in range(self.height - 1):
            layer = DisGeneralConvBlock(self.feature_size//2, self.feature_size)
            rgb = EqualizedConv2d(3, self.feature_size//2, 1, bias=False)
            if self.feature_size > 32:
                self.feature_size = self.feature_size // 2

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # register the temporary downSampler
        self.temporaryDownsampler = nn.AvgPool2d(2)

    def forward(self, x, height, alpha):
        """
        forward pass of the discriminator
        :param x: input to the network
        :param height: current height of operation (Progressive GAN)
        :param alpha: current value of alpha for fade-in
        :return: out => raw prediction values (WGAN-GP)
        """

        assert height < self.height, "Requested output depth cannot be produced"

        if height > 0:
            residual = self.rgb_to_features[height - 1](self.temporaryDownsampler(x))
            straight = self.layers[height - 1](
                self.rgb_to_features[height](x)
            )

            y = (alpha * straight) + ((1 - alpha) * residual)

            for block in reversed(self.layers[:height - 1]):
                y = block(y)
        else:
            y = self.rgb_to_features[0](x)

        out = self.final_block(y)

        return out
    

class GenInitialBlock(nn.Module):
    """ Module implementing the initial block of the input """

    def __init__(self, in_channels, out_channels):
        """
        constructor for the inner class
        :param in_channels: number of input channels to the block
        :param use_eql: whether to use equalized learning rate
        """

        super(GenInitialBlock, self).__init__()

        self.conv_1 = EqualizedDeconv2d(in_channels, in_channels, (4, 4), bias=False)
        self.conv_2 = EqualizedConv2d(in_channels, in_channels, (3, 3),
                                        pad=1, bias=False)
        self.conv_3 = EqualizedConv2d(in_channels, out_channels, (3, 3),
                                pad=1, bias=False)

        # Pixelwise feature vector normalization operation
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input to the module
        :return: y => output
        """
        # convert the tensor shape:
        y = torch.unsqueeze(torch.unsqueeze(x, -1), -1)

        # perform the forward computations:
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.pixNorm(self.conv_2(y)))
        y = self.lrelu(self.pixNorm(self.conv_3(y)))

        return y


class GenGeneralConvBlock(nn.Module):
    """ Module implementing a general convolutional block """

    def __init__(self, in_channels, out_channels):
        """
        constructor for the class
        :param in_channels: number of input channels to the block
        :param out_channels: number of output channels required
        :param use_eql: whether to use equalized learning rate
        """ 

        super(GenGeneralConvBlock, self).__init__()

        self.upsample = lambda x: F.interpolate(x, scale_factor=2)


        self.conv_1 = EqualizedConv2d(in_channels, out_channels, (3, 3),
                                        pad=1, bias=False)
        self.conv_2 = EqualizedConv2d(out_channels, out_channels, (3, 3),
                                        pad=1, bias=False)

        # Pixelwise feature vector normalization operation
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        #self.lrelu = LeakyReLU(0.2)

        self.relu = nn.ReLU()
    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        y = self.upsample(x)
        y = self.relu(self.pixNorm(self.conv_1(y)))
        y = self.relu(self.pixNorm(self.conv_2(y)))
        return y


class DisFinalBlock(nn.Module):
    """ Final block for the Discriminator """

    def __init__(self, in_channels):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param use_eql: whether to use equalized learning rate
        """

        super(DisFinalBlock, self).__init__()

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()
        self.conv_1 = EqualizedConv2d(in_channels + 1, in_channels, (3, 3), pad=1, bias=False)
        self.conv_2 = EqualizedConv2d(in_channels, in_channels, (4, 4), bias=False)
        # final conv layer emulates a fully connected layer
        self.conv_3 = EqualizedConv2d(in_channels, 1, (1, 1), bias=False)


        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the FinalBlock
        :param x: input
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)

        # define the computations
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # fully connected layer
        y = self.conv_3(y)  # This layer has linear activation

        # flatten the output raw discriminator scores
        return y.view(-1)


class DisGeneralConvBlock(nn.Module):
    """ General block in the discriminator  """

    def __init__(self, in_channels, out_channels):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param use_eql: whether to use equalized learning rate
        """

        super(DisGeneralConvBlock, self).__init__()

        self.conv_1 = EqualizedConv2d(in_channels, in_channels, (3, 3), pad=1, bias=False)
        self.conv_2 = EqualizedConv2d(in_channels, out_channels, (3, 3), pad=1, bias=False)


        #self.bn_1 = BatchNorm2d(out_channels)
        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)
        
        self.downSampler = nn.AvgPool2d(2)

    def forward(self, x):
        """
        forward pass of the module
        :param x: input
        :return: y => output
        """
        # define the computations
        #y = self.lrelu(self.bn_1(self.conv_1(x)))
        y = self.lrelu(self.conv_1(x))
        y = self.lrelu(self.conv_2(y))
        y = self.downSampler(y)
        return y


class Trainer():
    def __init__(self, params):
        self.params = params
        training_params = params["training"]
        self.num_epochs = training_params["epoch_list"]
        self.batch_sizes = training_params["batch_sizes"]
        self.fade_ins = [50] * len(self.num_epochs)
        self.depth = len(self.num_epochs)
        
        self.learning_rate = training_params["learning_rate"]
        self.betas = training_params["betas"]

        self.latent_dim = training_params["latent_dim"]

        self.checkpoint_root = "./train_result/"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, generator, discriminator, train_data_dir):
        
        # convert device
        generator.to(self.device)
        discriminator.to(self.device)

        # create a shadow copy of the generator
        generator_shadow = copy.deepcopy(generator)

        # initialize the gen_shadow weights equal to the
        # weights of gen
        update_average(generator_shadow, generator, beta=0)
        generator_shadow.train()
        
        optimizer_G = Adam(generator.parameters(),
                            lr=self.learning_rate,
                            betas=self.betas)
        optimizer_D = Adam(discriminator.parameters(),
                            lr=self.learning_rate, 
                            betas=self.betas)

        image_size = 2 ** (self.depth + 1)
        print("Construct dataset")
        train_dataset = ImageDataset(train_data_dir, image_size)
        print("Construct optimizers")

        now = datetime.datetime.now()
        checkpoint_dir = os.path.join(self.checkpoint_root, f"{now.strftime('%Y%m%d_%H%M')}-progan")
        sample_dir = os.path.join(checkpoint_dir, "sample")
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

        # training roop
        loss = losses.WGAN_GP(discriminator)
        print("Training Starts.")
        start_time = time.time()
        
        logger = Logger()
        iterations = 0
        fixed_input = torch.randn(16, self.latent_dim).to(self.device)
        fixed_input /= torch.sqrt(torch.sum(fixed_input*fixed_input + 1e-8, dim=1, keepdim=True))

        for current_depth in range(self.depth):
            current_res = 2 ** (current_depth + 1)
            print(f"Currently working on Depth: {current_depth}")
            current_res = np.power(2, current_depth + 2)
            print(f"Current resolution: {current_res} x {current_res}")

            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_sizes[current_depth],
                                        shuffle=True, drop_last=True, num_workers=4)
            
            ticker = 1
            
            for epoch in range(self.num_epochs[current_depth]):
                print(f"epoch {epoch}")
                total_batches = len(train_dataloader)
                fader_point = int((self.fade_ins[current_depth] / 100)
                                * self.num_epochs[current_depth] * total_batches)
                

                for i, batch in enumerate(train_dataloader):
                    alpha = ticker / fader_point if ticker <= fader_point else 1

                    images = batch.to(self.device)

                    gan_input = torch.randn(images.shape[0], self.latent_dim).to(self.device)
                    gan_input /= torch.sqrt(torch.sum(gan_input*gan_input + 1e-8, dim=1, keepdim=True))

                    real_samples = progressive_downsampling(images, current_depth, self.depth, alpha)
                        # generate a batch of samples
                    fake_samples = generator(gan_input, current_depth, alpha).detach()
                    d_loss = loss.dis_loss(real_samples, fake_samples, current_depth, alpha)

                    # optimize discriminator
                    optimizer_D.zero_grad()
                    d_loss.backward()
                    optimizer_D.step()

                    d_loss_val = d_loss.item()

                    gan_input = torch.randn(images.shape[0], self.latent_dim).to(self.device)
                    gan_input /= torch.sqrt(torch.sum(gan_input*gan_input + 1e-8, dim=1, keepdim=True))
                    fake_samples = generator(gan_input, current_depth, alpha)
                    g_loss = loss.gen_loss(real_samples, fake_samples, current_depth, alpha)
                    
                    optimizer_G.zero_grad()
                    g_loss.backward()
                    optimizer_G.step()

                    update_average(generator_shadow, generator, beta=0.999)
                    
                    g_loss_val = g_loss.item()

                    elapsed = time.time() - start_time

                    logger.log(depth=current_depth,
                               epoch=epoch,
                               i=i,
                               iterations=iterations,
                               loss_G=g_loss_val,
                               loss_D=d_loss_val,
                               time=elapsed)
                    
                    ticker += 1
                    iterations += 1

                logger.output_log(f"{checkpoint_dir}/log.csv")
                
                generator.eval()
                with torch.no_grad():
                    sample_images = generator(fixed_input, current_depth, alpha)
                save_samples(sample_images, current_depth, sample_dir, current_depth, epoch, image_size)
                generator.train()
                if current_depth == self.depth:
                    torch.save(generator.state_dict(), f"{checkpoint_dir}/{current_depth}_{epoch}_generator.pth")


def progressive_downsampling(real_batch, current_depth, depth, alpha):
    # downsample the real_batch for the given depth
    down_sample_factor = int(np.power(2, depth - current_depth - 1))
    prior_downsample_factor = max(int(np.power(2, depth - current_depth)), 0) 
    ds_real_samples = nn.AvgPool2d(down_sample_factor)(real_batch)

    if current_depth > 0:
        prior_ds_real_samples = F.interpolate(nn.AvgPool2d(prior_downsample_factor)(real_batch),
                                            scale_factor=2)
    else:
        prior_ds_real_samples = ds_real_samples

    # real samples are a combination of ds_real_samples and prior_ds_real_samples
    real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

    # return the so computed real_samples
    return real_samples


def save_samples(image_tensor, current_depth, sample_dir, depth, epoch, image_size):
    img = image_tensor.to("cpu").detach()
    img = img.numpy().transpose(0, 2, 3, 1)
    result = np.zeros((image_size*4, image_size*4, 3))
    img = (img + 1) * 127.5
    img = img.repeat(image_size//img.shape[1], axis=1).repeat(image_size//img.shape[2], axis=2)
    
    for i in range(4):
        for j in range(4):
            result[i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size, :] = img[4*i+j, :, :, :]
    result = result.astype(np.uint8)
    imsave(f"{sample_dir}/{depth}_{epoch}.png", result)

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--param_file', default="./params/progan.yml")
    parser.add_argument('--train_data', required=True)

    args = parser.parse_args()
    param_file = args.param_file
    train_data_dir = args.train_data

    with open(param_file, "r") as f:
        params = yaml.load(f)
    
    #network_params = params["network"]

    generator = Generator()
    discriminator = Discriminator()

    trainer = Trainer(params)

    trainer.train(generator, discriminator, train_data_dir)


