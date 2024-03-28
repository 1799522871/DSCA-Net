import argparse
import logging
import math

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

train_dir = 'D:/codestudy/MyPythonProject/remote-sesing/DSCA-Net/data/UCM_split/train/'
test_dir = 'D:/codestudy/MyPythonProject/remote-sesing/DSCA-Net/data/UCM_split/test/'

normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)




from pathlib import Path
from typing import Callable, Optional, Any
import torchvision
from torchvision.datasets import VisionDataset


def get_UCM(args, root, test_dir):
    transform_labeled = transforms.Compose([
            transforms.Resize(256),  # 将输入图像缩放到256x256
            transforms.CenterCrop(224),  # 从图像中心裁剪出224x224的区域
            transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
            transforms.ToTensor(),  # 将PIL图像或NumPy ndarray转换为FloatTensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
        ])
        # transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(size=224,
        #                       padding=int(224 * 0.125),
        #                       padding_mode='reflect'),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=normal_mean, std=normal_std)])

    # transform_val = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=normal_mean, std=normal_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    base_dataset = datasets.ImageFolder(root=root, transform=None)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets) # base_dataset.targets 获取标签

    train_labeled_dataset = UCMSSL(
        root=root, indexs=train_labeled_idxs,
        transform=transform_labeled)

    # train_unlabeled_dataset = Unlabel_UCMSSL(
    #     root=root, indexs=train_unlabeled_idxs,
    #     transform=TransformFixMatch_RS(mean=normal_mean, std=normal_std))
    train_unlabeled_dataset = Unlabel_UCMSSL(
        root=root, indexs=train_unlabeled_idxs)

    test_dataset = UCMSSL_test(root=test_dir,transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

class TransformFixMatch_RS(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224 * 0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224 * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class UCMSSL(Dataset):
    def __init__(self, root, indexs,
                 transform=None, target_transform=None):
        basedata = datasets.ImageFolder(root,transform=None)
        list = []
        if indexs is not None:
            # for i in indexs:
            #     list.append(np.array(Image.open(basedata.imgs[i][0])))
            for i in indexs:
                image = Image.open(basedata.imgs[i][0])
                resized_image = image.resize((224, 224))
                array = np.array(resized_image)
                list.append(array)

            self.targets = np.array(basedata.targets)[indexs]
            self.classes = basedata.classes  # Add this line
            self.class_to_idx = basedata.class_to_idx
            # self.imgs = np.array(basedata.imgs)[indexs]

            # 看的Dataset.cifar10
            # self.data = np.vstack(list).reshape(-1, 3, 256, 256)
            self.data = np.stack(list, axis = 0)
            # self.data = self.data.transpose((0, 2, 3, 1))
            print('train:{}'.format(self.data.shape))

            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    def __len__(self):
        return len(self.data)

# 无标签数据集获取
class Unlabel_UCMSSL(Dataset):
    def __init__(self, root, indexs,
                 transform=None, target_transform=None):
        basedata = datasets.ImageFolder(root,transform=None)
        list = []
        if indexs is not None:
            # for i in indexs:
            #     list.append(np.array(Image.open(basedata.imgs[i][0])))
            for i in indexs:
                image = Image.open(basedata.imgs[i][0])
                resized_image = image.resize((224, 224))
                array = np.array(resized_image)
                list.append(array)

            self.targets = np.array(basedata.targets)[indexs]
            self.classes = basedata.classes  # Add this line
            self.class_to_idx = basedata.class_to_idx
            # self.imgs = np.array(basedata.imgs)[indexs]

            # 看的Dataset.cifar10
            # self.data = np.vstack(list).reshape(-1, 3, 256, 256)
            self.data = np.stack(list, axis=0)
            # self.data = self.data.transpose((0, 2, 3, 1))
            print('train:{}'.format(self.data.shape))

            # 定义弱增强的变换
            self.weak_augmentation = transforms.Compose([
                transforms.Resize(256),  # 将输入图像缩放到256x256
                transforms.CenterCrop(224),  # 从图像中心裁剪出224x224的区域
                transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
                transforms.ToTensor(),  # 将PIL图像或NumPy ndarray转换为FloatTensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
            ])
            # transforms.Compose([
            #     transforms.RandomHorizontalFlip(),
            #     transforms.RandomCrop(size=224,
            #                           padding=int(224 * 0.125),
            #                           padding_mode='reflect'),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # ])

            # 定义强增强的变换
            self.strong_augmentation = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  # 随机裁剪，并缩放到224x224
                transforms.RandomApply([  # 以下变换随机应用其中之一
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                    torchvision.transforms.autoaugment.RandAugment()
                ], p=0.5),
                transforms.RandomGrayscale(p=0.2),  # 20%的概率将图像转换为灰度图
                transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
                transforms.RandomVerticalFlip(),  # 随机垂直翻转图像
                transforms.RandomRotation(degrees=15),  # 随机旋转图像
                transforms.ToTensor(),  # 将PIL图像或NumPy ndarray转换为FloatTensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
            ])
            # transforms.Compose([
            #     transforms.RandomHorizontalFlip(),
            #     transforms.RandomCrop(size=224,
            #                           padding=int(224 * 0.125),
            #                           padding_mode='reflect'),
            #     RandAugmentMC(n=2, m=10),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # ])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        img_weak = self.weak_augmentation(img)
        img_strong = self.strong_augmentation(img)

        return img_weak, img_strong, target
    def __len__(self):
        return len(self.data)

class UCMSSL_test(Dataset):
    def __init__(self, root,
                 transform=None, target_transform = None):
        basedata = datasets.ImageFolder(root,transform=None)
        list = []
        for i in range(len(basedata.imgs)):
            image = Image.open(basedata.imgs[i][0])
            resized_image = image.resize((224, 224))
            array = np.array(resized_image)
            list.append(array)


        self.targets = basedata.targets
        self.classes = basedata.classes  # Add this line
        self.class_to_idx = basedata.class_to_idx
        self.imgs = basedata.imgs

        # 看的Dataset.cifar100
        # self.data = np.vstack(list).reshape(-1, 3, 256, 256)
        self.data =  np.stack(list,axis=0)
        # self.data = self.data.transpose((0, 2, 3, 1))
        print('test:{}'.format(self.data.shape))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--num-labeled', type=int, default=504,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--total-steps', default=2**10, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=2**5, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--batch-size', default=4, type=int,
                        help='train batchsize')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='number of workers')


    args = parser.parse_known_args()[0]

    args.num_classes = 21

    train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_UCM(args,train_dir,test_dir)
    print("==============================================")

    labeled_trainloader = DataLoader(
        train_labeled_dataset,
        sampler=RandomSampler(train_labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        train_unlabeled_dataset,
        sampler=RandomSampler(train_unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    print('====================================')

    TestLabel = test_dataset.targets
    TestImg = test_dataset.data
    Test_class_to_idx = test_dataset.class_to_idx
    Test_classes = test_dataset.classes


    img1, img2, label = train_unlabeled_dataset[0]
    print(img1.shape)
    img1 = img1.permute(1, 2, 0) # CHW -> HWC
    plt.imshow(img1)




