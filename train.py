# Author: Jingxiao Gu
# Baidu Account: Seigato
# Description: Train Code for Lane Segmentation Competition

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset,ToTensor,ImageAug,DeformAug,ScaleAug,CutOut
from models.deeplabv3p import Res_Deeplab
from tqdm import tqdm
from torchvision import transforms

def mean_iou(pred, target, n_classes = 8):
  ious = []
  pred = torch.argmax(pred, axis=1)
  pred = pred.view(-1)
  target = target.view(-1)

  for cls in torch.arange(1, n_classes):
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]
    union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(max(union, 1)))
  return np.mean(ious)

# Get Loss Function
no_grad_set = []

class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)  # 二分类问题，sigmoid等价于softmax
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

def create_loss(predict, label, num_classes):
    predict = predict.permute(0, 2, 3, 1)
    predict = predict.view(-1, num_classes)
    predict = torch.softmax(predict)
    label = label.view(-1, 1)

    bce_loss = BCELoss2d()(predict,label)
    dice_loss = SoftDiceLoss()(predict,label)
    no_grad_set.append(label.name)
    loss = bce_loss + dice_loss
    miou = mean_iou(predict, label, num_classes)
    return torch.mean(loss), miou


def train_model(epoch,model,optimizer,train_loader,history=None):
    model.train()
    for batch_idx, (img_batch) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        predict, label = model(img_batch)
        loss = create_loss(predict,label,8)
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        loss.data))


def evaluate_model(epoch,model,dev_loader, history=None):
    model.eval()
    loss = 0
    with torch.no_grad():
        for img_batch in dev_loader:
            predict, label = model(img_batch)
            loss = create_loss(predict, label, 8)
            loss += loss
    loss /= len(dev_loader.dataset)

    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()

    print('Dev loss: {:.4f}'.format(loss))

def main():
    BATCH_SIZE = 2
    train_dataset = LaneDataset("train.csv", transform=transforms.Compose([
        ImageAug(), DeformAug(), ScaleAug(), CutOut(32,0.5), ToTensor()
    ]))
    val_dataset = LaneDataset("val.csv", transform=transforms.Compose([
        ImageAug(), DeformAug(), ScaleAug(), CutOut(32,0.5), ToTensor()
    ]))
    # Create data generators - they will produce batches
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    net = Res_Deeplab(8)
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net)
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
    for epoch in range(4):
        train_model(epoch,net,optimizer,train_loader)
        evaluate_model(epoch, net, optimizer, dev_loader)
    torch.save(net.state_dict(), './model.pth')

# Main
if __name__ == "__main__":
    main()
