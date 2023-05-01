import sys
import os
print("exe: ",sys.executable)


import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def methodRun():
    print("test")
    return -1, -1
class Run:
    if 1:
        np.random.seed(12)
        np.set_printoptions(threshold=10_000)
        torch.manual_seed(12)


        print("Is avail: ",torch.cuda.is_available())
        print("Device count: ",torch.cuda.device_count())
        print("curr active:", torch.cuda.current_device())
        #print(torch.cuda.device(0))
        print(torch.cuda.get_device_name(0))  

        device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
        print("curr device",device)

        
        dir = '../ECS171-Proj/images/'
        dirOs = os.listdir(dir + '/pokemon2') 
        numImages = len(dirOs)
        print("num images: ",numImages)
        print(os.getcwd())

        dataset = dset.ImageFolder(dir,
                            transform=transforms.Compose([
                                transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        #dataset = dset.ImageFolder(dir)

        dataObj = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
        


        aBatch = next(iter(dataObj))
        print(aBatch[:1])
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(aBatch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()

        print('************ Start ************')
        meanScore, stdScore = methodRun()
        print('************ Overall Performance ************')
        print('GAN STATS: ' + str(meanScore) + ' +/- ' + str(stdScore))
        print('************ Finish ************')