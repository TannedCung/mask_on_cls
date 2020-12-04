from PIL import Image
import cv2
import torch
from dataset.dataloader import ResNetDataset
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import os
import time
import numpy as np
import argparse
from network.models import *
from network.metrics import *
# from torchsampler import ImbalancedDatasetSampler


# ---------------------Parsing options---------------------
parser = argparse.ArgumentParser(description="Train face_mask, face_no_mask classification")
#--------------------Traing strategy--------------------
parser.add_argument("--optimizer", type=str, default="Adam", help="Adam, SGD")
parser.add_argument("--learning-rate", type=float, default=0.001)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--num-classes", type=int, default=3)
parser.add_argument("--start-epoch", type=int, default=0)
parser.add_argument("--end-epoch", type=int, default=400)
parser.add_argument("--gamma", type=float, default=0.05)
parser.add_argument("--step", nargs='+', type=int, default=[6,12,24,48,96,192])
parser.add_argument("--with-arc", type=bool, default=False)
parser.add_argument("--dataloader", type=int, default=1)
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


criterion = nn.CrossEntropyLoss()
# save_path = "checkpoints/model.pth"
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

    Net.to(device)
    Net.train()
    params = [*Net.parameters()]

    if args.optimizer == "Adam":
        opt = optim.Adam(params, lr=args.learning_rate)
    elif args.optimizer == "SGD":
        opt = optim.SGD(params, lr=args.learning_rate, momentum=args.momentum)
    else:
        print("Not implemented Optimizer {}".format(args.optimizer))

    my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.step, gamma=args.gamma)
    Uta = ResNetDataset(path=args.data_path, repl=(args.base_data_dir, args.alt_data_dir))
    data = torch.utils.data.DataLoader(Uta, batch_size=args.batch_size, shuffle=True)
    best = 0
    print("init net done")
    
    def save_progress(state, epoch, train_loss, train_acc):
        file = open("./progress.txt", "a")
        txt = "{}  epoch: {}       loss: {:.5}      acc: {:.5} ".format(state, epoch, train_loss, train_acc)
        file.write(txt+"\n")
        file.close()
    
    for epoch in range(args.start_epoch, args.end_epoch):
        running_loss = 0
        train_loss = 0
        valid_loss = 0
        start = time.time()
        total = 0
        correct = 0
        vtotal = 0
        vcorrect = 0
        for i, d in enumerate(data):
            [X, Y] = d[0].to(device), d[1].to(device)
    
            opt.zero_grad()
    
            out = Net(X)
            loss = criterion(out, Y)
            loss.backward()
            opt.step()
            idx = np.zeros(len(Y))
            # print(out[0].shape)
            # print(Y.shape)
            for j, o in enumerate(out):
                idx[j] = o.data.cpu().numpy().argmax()
            idx = torch.from_numpy(idx).type(torch.int64)
            
            for j in range((len(idx))):
                if idx[j] - Y[j] == 0:
                    correct += 1
            total += len(idx)
    
            running_loss += loss.item()
            train_loss += loss.item()*len(Y)
            if i % 10 == 9:
                print("[{}, {}], loss {:.10} in {:.5}s acc: {:.5}%".format(epoch+1, i+1, running_loss/10, time.time()-start, 100*correct/total))
                save_progress(state="   BATCH [{}, {}]".format(i-10, i), epoch= epoch+1, train_loss=running_loss/10, train_acc=100*correct/total)
                start = time.time()
                running_loss = 0.0

        print ("====== Epoch {} Loss: {:.5}, acc: {:.5}% ======".format(epoch+1, train_loss/len(data.sampler), 100*correct/total))
        my_lr_scheduler.step()
        if best <= 100*correct/total:
            torch.save(Net, args.model_save)
            # torch.save(metric_fc.weight, HEAD_PTH)
            print("model saved to {}".format(args.model_save))
            best = 100*correct/total
            save_progress(state="SAVED   ", epoch= epoch+1, train_loss=train_loss/len(data.sampler), train_acc=best)

        else:
            print("model not saved as best >= acc, current best : {}".format(best))
            save_progress(state="FAIL    ", epoch= epoch+1, train_loss=train_loss/len(data.sampler), train_acc=100*correct/total)

# Train with arcloss
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

    Net.cuda()
    Net.train()
    params = [*Net.parameters()] + [head.weight]

    if args.optimizer == "Adam":
        opt = optim.Adam(params, lr=args.learning_rate)
    elif args.optimizer == "SGD":
        opt = optim.SGD(params, lr=args.learning_rate, momentum=args.momentum)
    else:
        print("Not implemented Optimizer {}".format(args.optimizer))

    my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.step, gamma=args.gamma)

    Uta = ResNetDataset(path=args.data_path, repl=(args.base_data_dir, args.alt_data_dir))
    data = torch.utils.data.DataLoader(Uta, batch_size=args.batch_size, shuffle=True)
    # Valid = ResNetDataset(path="./data/valid")
    # valid_data = torch.utils.data.DataLoader(Valid, batch_size=4, shuffle=True)
    best = 0
    print("init net done")
    
    def save_progress(state, epoch, train_loss, train_acc):
        file = open("./progress.txt", "a")
        txt = "{}  epoch: {}       loss: {:.5}      acc: {:.5} ".format(state, epoch, train_loss, train_acc)
        file.write(txt+"\n")
        file.close()
    
    for epoch in range(args.start_epoch, 400):
        running_loss = 0
        train_loss = 0
        valid_loss = 0
        start = time.time()
        total = 0
        correct = 0
        vtotal = 0
        vcorrect = 0
        for i, d in enumerate(data):
            [X, Y] = d[0].to(device), d[1].to(device)
            opt.zero_grad()
    
            out = Net(X)
            out = head(out, Y)
            loss = criterion(out, Y)
            loss.backward()
            opt.step()
            idx = np.zeros(len(Y))
            # print(out[0].shape)
            # print(Y.shape)
            for j, o in enumerate(out):
                idx[j] = o.data.cpu().numpy().argmax()
            idx = torch.from_numpy(idx).type(torch.int64)
            
            for j in range((len(idx))):
                if idx[j] - Y[j] == 0:
                    correct += 1
            total += len(idx)
    
            running_loss += loss.item()
            train_loss += loss.item()*len(Y)
            if i % 10 == 9:
                print("[{}, {}], loss {:.10} in {:.5}s acc: {:.5}%".format(epoch+1, i+1, running_loss/10, time.time()-start, 100*correct/total))
                save_progress(state="   BATCH [{}, {}]".format(i-10, i), epoch= epoch+1, train_loss=running_loss/10, train_acc=100*correct/total)
                start = time.time()
                running_loss = 0.0

        print ("====== Epoch {} Loss: {:.5}, acc: {:.5}% ======".format(epoch+1, train_loss/len(data.sampler), 100*correct/total))
        my_lr_scheduler.step()
        if best <= 100*correct/total:
            torch.save(Net, args.model_save)
            # torch.save(metric_fc.weight, HEAD_PTH)
            print("model saved to {}".format(args.model_save))
            best = 100*correct/total
            save_progress(state="SAVED   ", epoch= epoch+1, train_loss=train_loss/len(data.sampler), train_acc=best)

        else:
            print("model not saved as best >= acc, current best : {}".format(best))
            save_progress(state="FAIL    ", epoch= epoch+1, train_loss=train_loss/len(data.sampler), train_acc=100*correct/total)