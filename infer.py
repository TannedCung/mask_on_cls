from PIL import Image
import cv2
import torch
import torch.nn as nn
from torchvision import transforms as T
import torchvision.models as models
import os
import time
import numpy as np
import argparse
from network.models import *
from network.metrics import *
# from torchsampler import ImbalancedDatasetSampler


# ---------------------Parsing options---------------------
parser = argparse.ArgumentParser(description="Infer face_mask, face_no_mask classification")
#--------------------Data options--------------------
parser.add_argument("--alt-data-dir", type=str, default='data', required=False)
parser.add_argument("--base-data-dir", type=str, default='data')
parser.add_argument("--data-path", type=str, default='data/train_MAFA')
#--------------------model options--------------------
parser.add_argument("--start-switch", type=str, default=None)
parser.add_argument("--model", type=str, default="MobileFaceNetUltraLite")
parser.add_argument("--model-restore", type=str, default=None)
parser.add_argument("--head-restore", type=str, default=None)
parser.add_argument("--model-save", type=str, default="checkpoints/mBFN.pth")
parser.add_argument("--head-save", type=str, default="checkpoints/mBFN_head.pth")
#--------------------ArcLoss options --------------------
parser.add_argument("--m", type=float, default=10)
parser.add_argument("--s", type=float, default=0.3)

args = parser.parse_args()
print(args)


# save_path = "checkpoints/model.pth"

def load_model(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if args.with_arc == False:
        if args.model == 'resnet18':
            Net = models.resnet18(num_classes=1000, pretrained=True)
            Net.fc = nn.Sequential(nn.Dropout(0.2),
                            nn.ReLU(),
                            nn.BatchNorm1d(512),
                            nn.Linear(512, args.num_classes)
                            )           

        elif args.model == 'resnet34':
            Net = models.resnet34(num_classes=1000, pretrained=True)
            Net.fc = nn.Sequential(nn.Dropout(0.2),
                            nn.ReLU(),
                            nn.BatchNorm1d(512),
                            nn.Linear(512, args.num_classes)
                            )           

        elif args.model == "MobileFaceNet":
            Net = MobileFaceNet(embedding_size=args.num_classes)
        
        elif args.model == "MobileFaceNetLite":
            Net = MobileFaceNetLite(embedding_size=args.num_classes)
        
        elif args.model == "MobileFaceNetSuperLite":
            Net = MobileFaceNetSuperLite(embedding_size=args.num_classes)
        
        elif args.model == "MobileFaceNetUltraLite":
            Net = MobileFaceNetUltraLite(embedding_size=args.num_classes)

        else:
            print("Model not implemented {}".format(args.model))

        if args.model_restore is not None:
            if os.path.exists(args.model_restore):
                # Net.load_state_dict(torch.load(NET_PTH))
                Net = torch.load(args.model_restore)
                print("Model loaded from {}".format(args.model_restore))
            else:
                print("Model restore not exist")
    else:
        if args.model == "resnet18" :
            Net = models.resnet18(num_classes=1000, pretrained=True)
            Net = nn.Sequential(*(list(Net.children())[:-2]))
            Net.avgpool = nn.Sequential(nn.Flatten(),
                        nn.Linear(512*7*7, 512))
            head = ArcMarginModel(emb_size=512, num_classes=args.num_classes, margin_s=args.m, margin_m=args.s)

        elif args.model == "MobileFaceNet":
            if args.start_switch is not None:
                Net = torch.load(args.start_switch)
                Net = nn.Sequential(*(list(Net.children())[:-2]))
                Net.flat = nn.Flatten()
                Net.linear = nn.Sequential(nn.Linear(512*8*8, 2048),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(2048))
                print("ArcFace model loaded from swtich")
            else:
                Net = MobileFaceNet(embedding_size=3)
                Net = nn.Sequential(*(list(Net.children())[:-2]))
                Net.flat = nn.Flatten()
                Net.linear = nn.Sequential(nn.Linear(512*8*8, 2048),
                                nn.ReLU(),
                                nn.BatchNorm1d(2048))
            head = ArcMarginModel(emb_size=2048, num_classes=args.num_classes, margin_s=args.m, margin_m=args.s)

        elif args.model == "MobileFaceNetLite":
            head = ArcMarginModel(emb_size=64, num_classes=args.num_classes, margin_s=args.m, margin_m=args.s)
            if args.start_switch is not None:
                Net = torch.load(args.start_switch)
                print(Net)
                head.weight = Net.linear.weight
                Net = nn.Sequential(*(list(Net.children())[:-1]))
                Net.flat = nn.Flatten()
                print("ArcFace model loaded from swtich")
            else:
                Net = MobileFaceNetLite(embedding_size=3)
                Net = nn.Sequential(*(list(Net.children())[:-1]))
                Net.flat = nn.Flatten()
        
        elif args.model == "MobileFaceNetSuperLite":
            head = ArcMarginModel(emb_size=64, num_classes=args.num_classes, margin_s=args.m, margin_m=args.s)
            if args.start_switch is not None:
                Net = torch.load(args.start_switch)
                print(Net)
                head.weight = Net.linear.weight
                Net = nn.Sequential(*(list(Net.children())[:-1]))
                Net.flat = nn.Flatten()
                print("ArcFace model loaded from swtich")
            else:
                Net = MobileFaceNetSuperLite(embedding_size=3)
                Net = nn.Sequential(*(list(Net.children())[:-1]))
                Net.flat = nn.Flatten()
        
        elif args.model == "MobileFaceNetUltraLite":
            head = ArcMarginModel(emb_size=64, num_classes=args.num_classes, margin_s=args.m, margin_m=args.s)
            if args.start_switch is not None:
                Net = torch.load(args.start_switch)
                print(Net)
                head.weight = Net.linear.weight
                Net = nn.Sequential(*(list(Net.children())[:-1]))
                Net.flat = nn.Flatten()
                print("ArcFace model loaded from swtich")
            else:
                Net = MobileFaceNetUltraLite(embedding_size=3)
                Net = nn.Sequential(*(list(Net.children())[:-1]))
                Net.flat = nn.Flatten()

        else:
            print("Model not implemented {}".format(args.model))

        if args.model_restore is not None and args.start_switch is None:
            if os.path.exists(args.model_restore):
                # Net.load_state_dict(torch.load(NET_PTH))
                Net = torch.load(args.model_restore)
                print("Model loaded from {}".format(args.model_restore))
            else:
                print("Model restore not exist")
        
        if args.head_restore is not None:
            if os.path.exists(args.head_restore):
                # Net.load_state_dict(torch.load(NET_PTH))
                head.weight = torch.load(args.head_restore)
                print("Head loaded from {}".format(args.head_restore))
            else:
                print("Head restore not exist")

    Net.to(device)
    
    return Net


def infer(Net, input):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = ["covered", "uncovered"]
    def _preprocess(input):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]


        transform = T.Compose([T.ToTensor(),
                                T.Normalize(mean, std)])
        return transform(input)
    input = _preprocess(input)
    X = input.to(device)
    out = Net(X)
    label = classes[int(out[0].data.cpu().numpy().argmax())]
    return label
                

   