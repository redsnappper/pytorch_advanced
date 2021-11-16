from random import shuffle
import torch
from torch.cuda import is_available
from torch.utils.data.dataset import T
from torchvision.transforms.transforms import ToTensor
from utils.dataloader_image_classification import ImageTransform, HymenopteraDataset, make_datapath_list
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

train_list = make_datapath_list(phase='train')
val_list = make_datapath_list(phase='val')

size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
train_dataset = HymenopteraDataset(
    file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')

val_dataset = HymenopteraDataset(
    file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')

batch_size = 32

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

dataLoaders_dict = {"train": train_dataloader, "val": val_dataloader}

use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)

net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

net.train()

print('ネットワーク設定完了:学習済みの重みをロードし、訓練モードに設定しました')

criterion = nn.CrossEntropyLoss()

params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

update_param_names_1 = ["features"]
update_param_names_2 = ["classifier.0.weight",
                        "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]


for name, param in net.named_parameters():
    if update_param_names_1[0] in name:
        param.requires_grad = True
        params_to_update_1.append(param)
        print("params_to_update_1に格納：", name)

    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        print("params_to_update_2に格納：", name)

    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
        print("params_to_update_3に格納：", name)

    else:
        param.requires_grad = False
        print("勾配計算なし。学習なし", name)

optimiser = optim.SGD([
    {'params': params_to_update_1, 'lr': 1e-4,
    'params': params_to_update_2, 'lr': 5e-4,
    'params': params_to_update_3, 'lr': 1e-3}
], momentum=0.9)

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス: ", device)

    net.to(device)

    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch+1, num_epochs))
        print('------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_correct = 0

            if(epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):

                # GPUが使えるならGPUにデータを送る
                inputs = inputs.to(device)
                labels = labels.to(device)

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)  # 損失を計算
                    _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_correct += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_correct.double() / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

num_epochs = 2
train_model(net, dataLoaders_dict, criterion, optimiser, num_epochs)

save_path = 'pytorch_advanced/1_image_classification/save_path'
torch.save(net.state_dict(), save_path)