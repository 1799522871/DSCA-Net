import math
from copy import deepcopy
from itertools import cycle
import argparse
import numpy as np
import torch
import torch.optim as optim
from collections import Counter
import os
from datetime import datetime

from torch.utils.data import SequentialSampler, DataLoader, RandomSampler

from dataset.UCM import get_UCM
from loader import get_dataloaders
import metric
from utils import AverageMeter, set_model, DrawCluster, visualization, result_display, display_predictions, convert_to_color, DrawResult, create_dir_str
import utils
import torch.nn.functional as F

import logging
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


def train(model,
          optim,
          lr_schdlr,
          args,
          selected_label,
          classwise_acc,
          labeled_train_loader,
          unlabeled_train_loader,
          ):
    cls_losses = AverageMeter()
    self_training_losses = AverageMeter()
    # define loss function
    cls_criterion = utils.CrossEntropyLoss()
    selected_label = selected_label.cuda()

    model.train()
    for batch_idx, data in enumerate(zip(cycle(labeled_train_loader), unlabeled_train_loader)):
        x_l, labels_l = data[0][0], data[0][1]      # 有标签数据的图像，标签
        x_u, x_u_strong, labels_u = data[1][0], data[1][1], data[1][2]   #  无标签图像的弱增强、强增强、标签
        x_l = x_l.cuda()
        x_u = x_u.cuda()
        x_u_strong = x_u_strong.cuda()
        labels_l = labels_l.cuda()
        batch_size = x_l.size(0)
        t = list(range(batch_size*batch_idx, batch_size*(batch_idx+1), 1)) # 每次得出一个batch的id
        t = (torch.from_numpy(np.array(t))).cuda()   # 将刚才的t转化为张量

        # --------------------------------------
        x = torch.cat((x_l, x_u, x_u_strong), dim=0)
        y, y_pseudo = model(x)   
        # cls loss on labeled data
        y_l = y[:args.batch_size]
        cls_loss = cls_criterion(y_l, labels_l.long())
        # self training loss on unlabeled data
        y_u, _ = y[args.batch_size:].chunk(2, dim=0)   #弱无标签数据的预测值  ，不用chunk(2,dim=0)的话，应该也可以用y[args.batch_size:args.batch_size*2]
        _, y_u_strong = y_pseudo[args.batch_size:].chunk(2, dim=0)   #强无标签数据的预测值

        #
        confidence, pseudo_labels = torch.softmax(y_u.detach(), dim=1).max(dim=1)   # 每个类的置信度
        mask = confidence.ge(0.95 * ((-0.3) * (torch.pow((classwise_acc[pseudo_labels] - 1), 2)) + 1)).float()
        #     self_training_loss = (F.cross_entropy(y_u_strong, pseudo_labels, reduction='none') * mask).mean()
        # else:
        self_training_loss = (F.cross_entropy(y_u_strong, pseudo_labels, reduction='none') * mask).mean()
        # if batch_idx == 100:
        #     print(t_p)
        #     print(confidence.mean())
        if t[mask == 1].nelement() != 0:
            selected_label[t[mask == 1]] = pseudo_labels[mask == 1]

        loss = cls_loss + self_training_loss

        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schdlr.step()

        cls_losses.update(cls_loss.item())
        self_training_losses.update(self_training_loss.item())
    return cls_losses.avg, self_training_losses.avg, selected_label


# test for one epoch
def test(model, test_loader):
    model.eval()
    total_accuracy,  total_num = 0.0, 0.0
    prediction = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            logits, _ = model(data)
            out = torch.softmax(logits, dim=1)
            pred_labels = out.argsort(dim=-1, descending=True)
            total_num += data.size(0)
            total_accuracy += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            for num in range(len(logits)):
                prediction.append(np.array(logits[num].cpu().detach().numpy()))

    return total_accuracy / total_num * 100, prediction

# 余弦退火学习率
def get_cosine_schedule_with_warmup(
    optimizer,
    num_training_steps,
    num_cycles=7.0 / 16.0,
    num_warmup_steps=0,
    last_epoch=-1,
):
    """
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    """

    def _lr_lambda(current_step):
        """
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        """

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch)


def main():
    dataset_names = ['UCM','']
    parser = argparse.ArgumentParser(description='Pseudo label for RSIC')
    parser.add_argument('--dataset', type=str, default='UCM', choices=dataset_names)
    parser.add_argument('--model', default='resnet50', type=str, choices=['Network','Supervisednetwork','WideResnet','resnet50'])
    parser.add_argument('--feature_dim', default=256, type=int, help='Feature dim for last conv')
    parser.add_argument('--batch_size', default=4, type=int, help='Number of data in each mini-batch')
    parser.add_argument('--epoches', default=1000, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate for training') # 原来为1e-3
    parser.add_argument('--runs', type=int, default=1, help='number of training times')
    parser.add_argument('--num-labeled', type=int, default=504, help='number of labeled data')
    parser.add_argument('--num-workers', type=int, default=1, help='number of workers')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
    parser.add_argument('--num_classes', type=int, default=21, help='该数据集的类别数')
    parser.add_argument('--train_dir', type=str, default='/root/data/DSCA-Net/data/UCM_split/train/')
    parser.add_argument('--test_dir', type=str, default='/root/data/DSCA-Net/data/UCM_split/test/')
    parser.add_argument('--save_name', type=str, default='results')
    args = parser.parse_known_args()[0]
    batch_size, epochs = args.batch_size, args.epoches
    
    #保存文件夹的名称
    dir_name = create_dir_str(args)
    # 创建新的文件夹
    folder_name = os.path.join(args.save_name, dir_name)
    os.makedirs(folder_name)
    logger_file = os.path.join(folder_name, "log.txt")

    # 创建一个Logger实例
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # 创建一个文件处理器
    file_handler = logging.FileHandler(logger_file)
    file_handler.setLevel(logging.INFO)
    # 创建控制台处理程序
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # 创建一个格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # 将格式化器添加到处理程序
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # 将文件处理器添加到 Logger 实例中
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('args:{}'.format(args))

    # 创建一个TensorBoard的SummaryWriter实例
    writer = SummaryWriter(folder_name)

    # data prepare
    for n in range(0, args.runs):
        logger.info(f"----Now begin the {format(n)} run----")
        train_labeled_dataset, unlabeled_dataset, test_dataset = get_UCM(args, args.train_dir, args.test_dir)

        labeled_train_loader = DataLoader(
            train_labeled_dataset,
            sampler=RandomSampler(train_labeled_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True)

        unlabeled_train_loader = DataLoader(
            unlabeled_dataset,
            sampler=RandomSampler(unlabeled_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True)

        test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers)

        # labeled_train_loader, _, unlabeled_train_loader, _, _, _, unlabeled_dataset = get_dataloaders(batchsize=batch_size, n=n)
        # _, test_loader, _, TestLabel, TestPatch, pad_width_data, _ = get_dataloaders(batchsize=batch_size, n=n)

        args.bands = 3  # 应该就是高光谱的通道数 103  ---> 输入的是rgb，所以是3
        #args.num_classes = len(np.unique(TestLabel))  # 在参数设置时已经处理
        model = set_model(args)   # 创建模型
        logger.info('model打印的结果:{}'.format(model))
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info("Total params: {:.2f}M".format(total_params))
        model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=True)
        #optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], 0.2, last_epoch=-1)
        # 总训练步数
        #num_training_steps = epochs * (len(train_labeled_dataset) // batch_size)
        #print('训练数据的长度:{}'.format(len(train_labeled_dataset)))
        #lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps, num_warmup_steps=num_training_steps * 0.1)

        # args.patchsize = 24   # 切割后输入的图片大小
        args.threshold = 0.95   # 全局 阈值
        # training loop
        best_acc, best_epoch = 0.0, 0
        selected_label = torch.ones((len(unlabeled_dataset),), dtype=torch.long, ) * -1   # 每个值都为1
        classwise_acc = torch.zeros((args.num_classes, )).cuda()

        for epoch in range(1, epochs+1):
            pseudo_counter = Counter(selected_label.tolist())   # 记录每种标签的总个数 ----> [{-1:1680}]
            # print('pseudo_counter:', pseudo_counter)
            if max(pseudo_counter.values()) < len(unlabeled_dataset):   # 初始时不满足这个条件
                wo_negative_one = deepcopy(pseudo_counter)
                if -1 in wo_negative_one.keys():    # 排出key为-1的字典
                    wo_negative_one.pop(-1)
                for i in range(args.num_classes):
                    classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())
            loss_x, loss_u, selected_label = \
                train(model=model, optim=optimizer, lr_schdlr=lr_scheduler, classwise_acc=classwise_acc,
                      selected_label=selected_label, labeled_train_loader=labeled_train_loader,
                      unlabeled_train_loader=unlabeled_train_loader,  args=args)
            test_acc, predictions = test(model, test_loader)
            logger.info('Epoch: [{}/{}] | classify loss_x:{:.4f} | classify loss_u:{:.4f} | Test Acc:{:.2f} | lr:{:.4f}'
                  .format(epoch, epochs, loss_x, loss_u, test_acc, lr_scheduler.get_last_lr()[0]))
            writer.add_scalar('train/train_loss_x', loss_x,  epoch)
            writer.add_scalar('train/train_loss_u', loss_u,  epoch)
            writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0],  epoch)
            writer.add_scalar('test/test_acc', test_acc,  epoch)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                torch.save(model, os.path.join(folder_name,'best_acc_result1.pth'))
        logger.info('Best test_acc_1: {:.2f} at runs:{} epoch {}'.format(best_acc, n, best_epoch))

        model = torch.load(os.path.join(folder_name,'best_acc_result1.pth'))
        model.eval()


        # 对整个测试集进行处理
        TestLabel = test_loader.dataset.targets
        # TestImg = test_loader.dataset.data
        Test_class_to_idx = test_loader.dataset.class_to_idx
        Test_classes = test_loader.dataset.classes

        pred_y = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.cuda()  # 将输入数据移至 GPU（如果可用）
                outputs, _ = model(inputs)  # 获取模型的预测输出
                _, predicted = torch.max(outputs.data, 1)  # 获取每个样本的预测值
                pred_y.extend(predicted.cpu().tolist())  # 将预测值添加到列表

        pred_y = np.array(pred_y)
        # 评价指标
        pred_y = torch.from_numpy(pred_y).long()
        Classes = np.unique(TestLabel)
        EachAcc = np.empty(len(Classes))
        AA = 0.0
        for i in range(len(Classes)):
            cla = Classes[i]   # 获取该类对应的值
            right = 0   # 该类预测的正确数
            sums = 0   # 该类的总数
            for j in range(len(TestLabel)):
                if TestLabel[j] == cla:
                    sums += 1
                if TestLabel[j] == cla and pred_y[j] == cla:
                    right += 1
            EachAcc[i] = right.__float__() / sums.__float__()
            AA += EachAcc[i]

        logger.info('-------------------')
        for i in range(len(EachAcc)):
            logger.info('|第{}类精度：{:.2f}'.format(get_key_from_value(Test_class_to_idx, i),EachAcc[i] * 100))
            logger.info('-------------------')
        AA *= 100 / len(Classes)   # 总的正确率

        results = metric.metrics(pred_y, TestLabel, n_classes=len(Classes))
        logger.info('test accuracy（OA）: {:.2f}, AA:{:.2f}, Kappa:{:.2f} '.format(results["Accuracy"],AA,results["Kappa"]))
        logger.info('confusion matrix :')
        logger.info(results["Confusion matrix"])

        # 将 test accuracy（OA）、AA 和 Kappa 写入 TensorBoard
        writer.add_scalar('Test Accuracy', results["Accuracy"], 0)
        writer.add_scalar('AA', AA, 0)
        writer.add_scalar('Kappa', results["Kappa"], 0)

        # 将混淆矩阵写入 TensorBoard
        # writer.add_image('Confusion Matrix', np.array(results["Confusion matrix"]), 0)


        # 混淆矩阵
        save_path = createConfusionMatrix(pred_y,TestLabel,Test_class_to_idx,folder_name)
        image = plt.imread(save_path)
        writer.add_image('Confusion Matrix', image, dataformats='HWC')

    # 关闭 SummaryWriter
    writer.close()
    # 关闭文件处理器
    logger.removeHandler(file_handler)
    file_handler.close()

def get_key_from_value(dictionary, target_value):
    values = list(dictionary.values())
    if target_value in values:
        index = values.index(target_value)
        keys = list(dictionary.keys())
        return keys[index]
    # 如果未找到匹配的键，可以根据需求返回特定值或引发异常
    # return None
    raise ValueError("Value not found")

# 创建混淆矩阵的图片
def createConfusionMatrix(pred_y, y_true, label_dict, folder_name):
    # 将真实标签索引转换为真实标签名称
    y_true_labels = [label for label in label_dict.keys() if label_dict[label] in y_true]

    # 计算混淆矩阵
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, pred_y, labels=labels)

    # 绘制混淆矩阵图像
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, [label for label in label_dict.keys() if label_dict[label] in labels], rotation=45)
    plt.yticks(tick_marks, [label for label in label_dict.keys() if label_dict[label] in labels])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    # 添加真实标签名称到每个格子
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        true_label_idx = labels[i]
        true_label = [label for label in label_dict.keys() if label_dict[label] == true_label_idx][0]
        plt.text(j, i, true_label if i == j else cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    # 保存混淆矩阵图像
    save_path = os.path.join(folder_name, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    return save_path


if __name__ == '__main__':
    main()



