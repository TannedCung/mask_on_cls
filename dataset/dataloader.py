import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import os
import glob
from PIL import Image
import torch.nn.functional as F

class ResNetDataset(Dataset):
    def __init__(self, path, transforms=None, repl=("", ""), type="train"):
        self.repl = repl
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        if type=="train":
            self.train_path = os.path.join(path, "train")
            self.transform = T.Compose([T.Resize((128,128), interpolation=2),
                                        # T.RandomRotation(45),
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