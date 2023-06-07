# import the library
import sys
import argparse
import os
import PIL 
import torch
import pickle
import random
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.parallel
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.utils import save_image

# The architecture of the generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
 
        self.conv1 = nn.ConvTranspose2d(64, 2048, kernel_size = 4, stride = 1, padding = 0, bias = False)
        self.conv2 = nn.ConvTranspose2d(2048, 512, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.conv3 = nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.conv4 = nn.ConvTranspose2d(256, 64, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.conv6 = nn.ConvTranspose2d(32, 3, kernel_size = 4, stride = 2, padding = 1, bias = False)

        
        self.batch1 = nn.BatchNorm2d(2048)
        self.batch2 = nn.BatchNorm2d(512)
        self.batch3 = nn.BatchNorm2d(256)
        self.batch4 = nn.BatchNorm2d(64)

        self.activation = nn.ReLU(True)
        self.dropout = nn.Dropout(p = 0.15)
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.batch2(self.conv2(x)))
        x = self.activation(self.batch3(self.conv3(x)))
        x = self.dropout(x)
        x = self.activation(self.batch4(self.conv4(x)))
        x = self.activation(self.conv5(x))
        x = self.conv6(x)

        return nn.Tanh()(x)

# The architecture of the Classifier 
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(20, 16, kernel_size = 4, stride = 2, padding = 1, bias = False)

        self.linear1 = nn.Linear(16 * 8 * 8, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 18)

        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p = 0.35)

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(-1, 16 * 8 * 8)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.linear4(x)
        return x

# The function below is to generate certain type of Pokemon image
def generate_image(type_num):
    # First, delete all files generated before
    for previous_image in os.listdir('./'):
        if previous_image.endswith('.png'):
            os.remove(previous_image)

    # The location of the trained model
    generator_name = "website/static/pokemonmodel/pokemon_generator.pth"
    classifier_name = "website/static/pokemonmodel/pokemon_classifier.pth"

    # Activate the saved classifier model
    classifier = Classifier()
    classifier.load_state_dict(torch.load(classifier_name, map_location=torch.device('cpu')))
    classifier.eval()

    # Activate the saved generator model
    generator = Generator()
    generator.load_state_dict(torch.load(generator_name, map_location=torch.device('cpu')))
    generator.eval() 

    # The dictionary stored each number's corresponding type
    type_dict = {0: 'Bug', 1: 'Dark', 2: 'Dragon', 3: 'Electric', 4: 'Fairy', 5: 'Fighting', 6: 'Fire', 7: 'Flying', 8: 'Ghost', 9: 'Grass', 10: 'Ground', 11: 'Ice', 12: 'Normal', 13: 'Poison', 14: 'Psychic', 15: 'Rock', 16: 'Steel', 17: 'Water'}
    print("Type is: ", type_dict[type_num])

    image_index = 0

    # Generate 20 x 32 = 640 images in all
    for x in range(20):

        # Make a random noise and feed it to the generator
        random_noise = torch.randn(32, 64, 1, 1)
    
        # The image tensor
        image = generator(random_noise).detach()
        
        # Get the type of each image
        types = classifier(image)
        types = types.max(1)[1]
        types = types.tolist()


        for i in range(len(image)):
            # If the image type is the target type, then we transform the image tensor to the real image
            # Reference of changing image from tensor to png image:
            # https://pytorch.org/vision/stable/generated/torchvision.utils.save_image.html
            if types[i] == type_num:
                # Change the image from tensor to png format image
                # Choose the image with the correct type
                final_image = image[i]

                # After checking the minimum and maximum value of the image, the range is from -1 to 1
                # Need to normalize the data so that the range is from 0 to 1
                # Otherwise, the color of the image would be strange
                final_image = (final_image + 1) / 2.0

                # print(image.min(), image.max())
                
                # Use save_image to save the image
                save_image(final_image, 'website/static/pokemonmodel/{}{}.png'.format(type_dict[type_num], image_index))

                image_index += 1