# -*- coding: utf-8 -*-

import os
from glob import glob
import random
import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np
import pandas as pd
import skimage.transform
import skimage.io

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, data_dir, image_size):

        color_jitter = [transforms.ColorJitter(brightness=0.04,
                                              contrast=0.02,
                                              saturation=0.02,
                                              hue=0.02)]
        
        self.transform = transforms.Compose([transforms.RandomCrop(image_size),
                                             transforms.RandomHorizontalFlip(0.5),
                                             transforms.RandomApply(color_jitter, 0.3),
                                             transforms.ToTensor()])
        print("preparing datasets")
        #self.images = [self.transform_1(Image.open(path)) for path in glob(data_dir + '/*.*')]
        self.images = glob(data_dir + '/*.*')
        print("preparing complete")

    def __getitem__(self, index):
        #image = self.transform_2(self.images[index])
        image = self.transform(Image.open(self.images[index]))
        image = (image - 0.5) * 2
        return image

    def __len__(self):
        return len(self.images)

    def show_transformations(self, num_transformant=10):
        result_list = []
        image = Image.open(self.images[0])
        for _ in range(num_transformant):
            result_list.append(self.transform_2(image))
        return result_list


def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)


class Logger():
    def __init__(self):
        self.log_df = pd.DataFrame(columns=["depth" ,"epoch", "i", "iterations", "loss_G", "loss_D", "time"])


    def log(self, depth=None, epoch=None, i=None, iterations=None, loss_G=None, loss_D=None, time=None):
        temp_df = pd.DataFrame({"depth": [depth],
                                "epoch": [epoch],
                                "i": [i],
                                "iterations": [iterations],
                                "loss_G": [loss_G],
                                "loss_D": [loss_D],
                                "time": [time]})
        self.log_df = pd.concat([self.log_df, temp_df], axis=0)
        print(f"epoch {epoch}, iter {i} (total {iterations}), loss_G: {loss_G:.4f}, loss_D: {loss_D:.4f}, time: {time:.2f}")
    
    def output_log(self, path):
        self.log_df.to_csv(path, index=False)


def update_average(model_tgt, model_src, beta):
    """
    update the model_target using exponential moving averages
    :param model_tgt: target model
    :param model_src: source model
    :param beta: value of decay beta
    :return: None (updates the target model)
    """

    # utility function for toggling the gradient requirements of the models
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    # turn off gradient calculation
    toggle_grad(model_tgt, False)
    toggle_grad(model_src, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    # turn back on the gradient calculation
    toggle_grad(model_tgt, True)
    toggle_grad(model_src, True)


