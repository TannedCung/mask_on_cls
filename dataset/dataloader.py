import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import os
import glob
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F
import PIL


class SaltAndPepperNoise(nn.Module):
    def __init__(self,
                 imgType:str = "PIL",
                 lowerValue:int = 5,
                 upperValue:int = 250,
                 noiseType:str = "RGB"):
        self.imgType = imgType
        self.lowerValue = lowerValue # 255 would be too high
        self.upperValue = upperValue # 0 would be too low
        self.noiseType = noiseType
        super(SaltAndPepperNoise, self).__init__()

    def __call__(self, img):
        if self.imgType == "PIL":
            img = np.array(img)
        
        threshold = np.random.uniform(0,0.008)
        if self.noiseType == "SnP":
            random_matrix = np.random.rand(img.shape[0],img.shape[1])
            img[random_matrix>=(1-threshold)] = self.upperValue
            img[random_matrix<=threshold] = self.lowerValue
        elif self.noiseType == "RGB":
            random_matrix = np.random.random(img.shape)      
            img[random_matrix>=(1-threshold)] = self.upperValue
            img[random_matrix<=threshold] = self.lowerValue
        if self.imgType == "cv2":
            return img
        elif self.imgType == "PIL":
            # return as PIL image for torchvision transforms compliance
            return PIL.Image.fromarray(img)

class ResNetDataset(Dataset):
    def __init__(self, path, transforms=None, repl=("", ""), type="train"):
        self.repl = repl
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        if type=="train":
            self.train_path = os.path.join(path, "train")
            self.transform = T.Compose([T.Resize((128,128), interpolation=2),
                                        T.RandomRotation(5),
                                        T.RandomApply(torch.nn.ModuleList([
                                            T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 5)),
                                            SaltAndPepperNoise()]),
                                            p=0.3
                                            ),
                                        T.RandomVerticalFlip(),
                                        T.RandomGrayscale(),
                                        T.RandomSizedCrop((112,112)),
                                        T.ColorJitter(brightness=0.4, contrast=0.4,saturation=0.4),
                                        T.GaussianBlur(kernel_size=5),
                                        T.ToTensor(),
                                        T.Normalize(self.mean, self.std)])
        elif type=="test":
            self.train_path = os.path.join(path, "test")
            self.transform = T.Compose([T.Resize((112,112), interpolation=2),
                                        T.ToTensor(),
                                        T.Normalize(self.mean, self.std)])
        else:
            print(f"[ERR]: Unknown dataloader type")
            return -1

        self.classes = []
        for d in os.listdir(self.train_path):
            if ".txt" not in d:
                self.classes.append(d)
                
        self.paths = []
        for c in self.classes:
            for file in list(glob.glob(os.path.join(self.train_path, os.path.join(c, "*.*")))):
                self.paths.append(file)

        self.classes.sort()
        print("Dataset loaded with {} images of {} classes".format(len(self.paths),len(self.classes)))

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
         
        X_path = self.paths[idx].replace(self.repl[0], self.repl[1])
        X = Image.open(X_path).convert("RGB")
        Y = self.paths[idx][:-1].split(os.path.sep)[-2]
        X = self.transform(X)
        Y = self.classes.index(Y)
        # Y = torch.tensor(Y, dtype=torch.long)
        data = [X, Y]
        return data