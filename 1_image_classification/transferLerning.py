import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import  tqdm
import matplotlib.pyplot as plt
import pprint as pr

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import models, transforms

rndnum = 1234
torch.manual_seed(rndnum)
np.random.seed(rndnum)
random.seed(rndnum)

epoch_loss_plt = []
epoch_acc_plt = []
epoch_plt_count = []

size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
            ),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):

        return self.data_transform[phase](img)

def make_datapath_list(phase="train"):

    rootpath = "./pytorch_advanced/1_image_classification/data/hymenoptera_data/"
    target_path = osp.join(rootpath+phase+'/**/*.jpg')
    print(target_path)

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list

train_list = make_datapath_list(phase="train")
val_list = make_datapath_list(phase="val")

class HymenopteraDataset(data.Dataset):

    def __init__(self, file_list, transform=None, phase="train"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(
            img, self.phase
        )
        if self.phase == "train":
            wordPosition = int(img_path.find("train"))+6
            label = img_path[wordPosition:wordPosition+4]
        elif self.phase == "val":
            wordPosition = int(img_path.find("val"))+4
            label = img_path[wordPosition:wordPosition+4]


        if label == "ants":
            label = 0

        elif label == "bees":
            label = 1

        return img_transformed, label


train_dataset = HymenopteraDataset(
    file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')

val_dataset = HymenopteraDataset(
    file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')

batch_size = 16

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}


use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)

net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

net.train()

print('ネットワーク設定完了：学習済みの重みをロードし、訓練モードに設定しました')

criterion = nn.CrossEntropyLoss()
params_to_update = []

update_param_names = ["classifier.6.weight", "classifier.6.bias"]

for name, param in net.named_parameters():
    if name in update_param_names:
        param.requires_grad = True
        params_to_update.append(param)
        print(name)
    else:
        param.requires_grad = False

print("-----------")
print(params_to_update)

optimizer = optim.SGD(params=params_to_update, lr=000.1, momentum=0.9)


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                print("モデルを訓練モードに")
            else:
                net.eval()
                print("モデルを検証モードに")

            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            epoch_loss_plt.append(epoch_loss)
            epoch_acc_plt.append(epoch_acc)
            epoch_plt_count.append(epoch)

num_epochs=3
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

# save_path = 'pytorch_advanced/1_image_classification/save_path/'
# torch.save(net.state_dict(), save_path)

plt.plot(epoch_plt_count, epoch_loss_plt, label='epoch loss')
plt.plot(epoch_plt_count, epoch_acc_plt, label = 'epoch accuracy')

plt.title('Transfer Learning')
plt.xlabel('# of Epoch')
plt.ylabel('Epoch_Accuracy')

plt.show()