import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from utils import *
import numpy as np
import torchvision
class loader_CIFAR10():

    def __init__(self, args):
        super(loader_CIFAR10, self).__init__()

        transform_train = transforms.Compose([
            RandomPadandCrop(32),
            RandomFlip(),
            ToTensor(),
        ])

        transform_val = transforms.Compose([
            ToTensor(),
        ])

        download_root = 'D:/2020-2/비즈니스애널리틱스/논문리뷰/Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction/CIFAR10_DATASET'
        dataset = CIFAR10(download_root,train=True, download=True)
        train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(labels=dataset.targets , n_labeled_per_class=args.n_labeled)
        train_labeled_dataset = CIFAR10_labeled(download_root, train_labeled_idxs, train=True, transform=transform_train)
        train_unlabeled_dataset = CIFAR10_unlabeled(download_root, train_unlabeled_idxs, train=True,
                                                    transform=TransformTwice(transform_train))
        val_dataset = CIFAR10_labeled(download_root, val_idxs, train=True, transform=transform_val, download=True)
        test_dataset = CIFAR10_labeled(download_root, train=False, transform=transform_val, download=True)

        self.train_iter_labeld = DataLoader(dataset=train_labeled_dataset , batch_size=args.batch_size , shuffle=True, drop_last=True)
        self.train_iter_unlabeld = DataLoader(dataset=train_unlabeled_dataset , batch_size=args.batch_size , shuffle=True, drop_last=True)
        self.valid_iter = DataLoader(dataset=val_dataset , batch_size=args.batch_size , shuffle=True, drop_last=True)
        self.test_iter = DataLoader(dataset=test_dataset , batch_size=args.batch_size , shuffle=True, drop_last=True)

def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                                              transform=transform, target_transform=target_transform,
                                              download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalise(self.data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                                                transform=transform, target_transform=target_transform,
                                                download=download)
        self.targets = np.array([-1 for _ in range(len(self.targets))])