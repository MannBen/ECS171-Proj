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

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
 
        self.conv1 = nn.ConvTranspose2d(64, 1024, 4, 1, 0, bias = False)
        self.conv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias = False)
        self.conv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False)
        self.conv4 = nn.ConvTranspose2d(256, 64, 4, 2, 1, bias = False)
        self.conv5 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias = False)
        self.conv6 = nn.ConvTranspose2d(32, 3, 4, 2, 1, bias = False)

        
        self.batch1 = nn.BatchNorm2d(1024)
        self.batch2 = nn.BatchNorm2d(512)
        self.batch3 = nn.BatchNorm2d(256)
        self.batch4 = nn.BatchNorm2d(64)
        self.batch5 = nn.BatchNorm2d(32)

        self.activation = nn.ReLU(True)
        self.dropout = nn.Dropout(p = 0.15)
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.batch2(self.conv2(x)))
        x = self.activation(self.batch3(self.conv3(x)))
        x = self.dropout(x)
        x = self.activation(self.batch4(self.conv4(x)))
        x = self.activation(self.batch5(self.conv5(x)))
        x = self.conv6(x)

        return nn.Tanh()(x)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 18)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def generate_image(type_num):
    # First, delete all files generated before
    for previous_image in os.listdir('./'):
        if previous_image.endswith('.png'):
            os.remove(previous_image)

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

    #type_num = random.randint(0, 17)
    # type_num = int(sys.argv[1])
    type_dict = {0: 'Bug', 1: 'Dark', 2: 'Dragon', 3: 'Electric', 4: 'Fairy', 5: 'Fighting', 6: 'Fire', 7: 'Flying', 8: 'Ghost', 9: 'Grass', 10: 'Ground', 11: 'Ice', 12: 'Normal', 13: 'Poison', 14: 'Psychic', 15: 'Rock', 16: 'Steel', 17: 'Water'}
    print("Type is: ", type_dict[type_num])

    image_index = 0

    # Generate 20 x 32 = 640 images in all
    for x in range(20):
        # Make a random noise and feed it to the generator
        random_noise = torch.randn(32, 64, 1, 1)
    
        # [32, 3, 128, 128]
        image = generator(random_noise).detach()

        # [32, 18]
        types = classifier(image)
        types = types.max(1)[1]
        types = types.tolist()

        image = image.numpy()

        for i in range(len(image)):
            if types[i] == type_num:
                numpy_image = np.transpose(image[i], (1, 2, 0))
                numpy_image = (numpy_image + 1) / 2.0

                stored_image = np.clip(numpy_image, 0, 1)
                stored_image = (stored_image * 255).astype(np.uint8)

                final_image = Image.fromarray(stored_image)
                final_image.save('website/static/pokemonmodel/{}{}.png'.format(type_dict[type_num], image_index))
                image_index += 1
# pickle.dump(modelname, open(picklemodel.pkl))

