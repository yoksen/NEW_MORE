import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.datasets as vdsets

from timm.models import deit_small_patch16_224
import pickle
import os
import argparse
parser = argparse.ArgumentParser(description='CIFAR100 training with DeiT-Small')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--device', default=0, type=int, help='device')

args = parser.parse_args()

train_trsf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),#cancel colorjitter so far
        transforms.ColorJitter(brightness=63/255),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
test_trsf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

device = torch.device('cuda:2')

num_epochs = 200
cp_epoch = 50
batch_size = 128
learning_rate = args.lr

# trsf = transforms.Compose(transforms.ToTensor()])

train_dir=os.path.join("/root/autodl-tmp/data/seed32_pre_mini_imagenet/xuchen/seed32_pre_mini_imagenet/miniImageNet", "train")
test_dir=os.path.join("/root/autodl-tmp/data/seed32_pre_mini_imagenet/xuchen/seed32_pre_mini_imagenet/miniImageNet", "val")
# test_dir=os.path.join(os.environ["MINIIMAGENETDATASET"], "test")


train_dataset = vdsets.ImageFolder(train_dir, transform=train_trsf)
test_dataset = vdsets.ImageFolder(test_dir, transform=test_trsf)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)

# # 加载模型
model = deit_small_patch16_224(pretrained=False, num_classes=50)
model.to(device)
print("模型类型：", type(model))
print("模型是否在GPU上：", next(model.parameters()).is_cuda)
def compute_acc(model, loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# # 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)

# # 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # 将数据移到GPU上
        images = images.to(device)
        labels = labels.to(device)
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    scheduler.step()
    train_acc = compute_acc(model, train_loader)
    test_acc = compute_acc(model, test_loader)
    print('Epoch [{}/{}], Current LR: {:.6f}'.format(epoch+1, num_epochs, scheduler.get_last_lr()[0]))
    print ('Epoch {}/{}, train acc:{}, test acc:{}'.format(epoch + 1, num_epochs, train_acc, test_acc))
    
    if  (epoch + 1) % cp_epoch == 0:
        torch.save(model.state_dict(), '/root/autodl-tmp/MORE/checkpoint/deit_seed32_i100_epoch_{}_lr_{}.ckpt'.format(epoch + 1, learning_rate))
# # 测试模型


torch.save(model.state_dict(), 'deit_seed32_i100_lr_{}.ckpt'.format(learning_rate))
