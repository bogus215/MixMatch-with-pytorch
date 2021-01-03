# %% library
from loader import *
import argparse
from model import *
import numpy as np
import torch
import wandb
import torch.optim as optim
from rich.console import Console
from pytorchtools import EarlyStopping
from tqdm import tqdm
import gc
import random

# %% Train
def training(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(args.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=10, verbose=False, path=f'./parameter/{args.experiment}.pth')

    console = Console()

    for epoch in range(args.epoch+1):
        print("\n===> epoch %d" % epoch)

        args.model.train()
        with tqdm(total=len(args.loader.train_iter_labeld), leave=False, dynamic_ncols=True) as pbar:

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            for batch_idx, (inputs, targets) in enumerate(args.loader.train_iter_labeld):
                inputs = inputs.cuda(args.gpu_device)
                targets = targets.cuda(args.gpu_device)

                outputs = args.model(inputs)
                loss = criterion(outputs , targets.long())
                outputs = F.softmax(outputs, dim =1)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(1)

        train_loss, train_acc = losses.avg, top1.avg

        with tqdm(total=len(args.loader.valid_iter), leave=False, dynamic_ncols=True) as pbar:

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            args.model.eval()

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(args.loader.valid_iter):
                    inputs = inputs.cuda(args.gpu_device)
                    targets = targets.cuda(args.gpu_device)

                    outputs = args.model(inputs)
                    loss = criterion(outputs,targets.long())
                    outputs = F.softmax(outputs, dim =1)

                    # measure accuracy and record loss
                    prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                    losses.update(loss.item(), inputs.size(0))
                    top1.update(prec1.item(), inputs.size(0))
                    top5.update(prec5.item(), inputs.size(0))

                    pbar.update(1)

        val_loss, val_acc = losses.avg, top1.avg

        with tqdm(total=len(args.loader.test_iter), leave=False, dynamic_ncols=True) as pbar:

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            args.model.eval()

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(args.loader.test_iter):
                    inputs = inputs.cuda(args.gpu_device)
                    targets = targets.cuda(args.gpu_device)

                    outputs = args.model(inputs)
                    loss = criterion(outputs, targets.long())
                    outputs = F.softmax(outputs, dim =1)

                    # measure accuracy and record loss
                    prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                    losses.update(loss.item(), inputs.size(0))
                    top1.update(prec1.item(), inputs.size(0))
                    top5.update(prec5.item(), inputs.size(0))

                    pbar.update(1)

            test_loss, test_acc = losses.avg, top1.avg

            console.print(f"Train [{epoch:>04}]/[{args.epoch:>03}]",f"Train acc:{train_acc:.4f}",f"valid acc:{val_acc:.4f}",f"Test acc:{test_acc}",end=' | ', style="Bold Cyan")
            console.print(f"Train loss:{train_loss:.4f}",f"valid loss_x:{val_loss:.4f}",f"Test loss:{test_loss}",end=' | ', style="Bold Cyan")

            wandb.log({'Train acc': train_acc,
                       'Valid acc': val_acc,
                       'Test acc': test_acc,
                       'Train loss':train_loss
                       })
            early_stopping(-val_acc, args.model)
            if early_stopping.early_stop:
                print('Early stopping')
                break


# %% main
def main():
    wandb.init(project='PyTorch MixMatch Training', reinit=True)
    parser = argparse.ArgumentParser(description="'PyTorch MixMatch Training'")
    parser.add_argument("--learning_rate", default=0.002, type=float, help="learning rate")
    parser.add_argument("--epoch", default=1024, type=int, help="number of max epoch")
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument("--gpu_device", default=0, type=int, help="the number of gpu to be used")
    parser.add_argument('--experiment', type=str, default='0102_normal', help='experiment name')
    parser.add_argument('--n_labeled', type=int, default=250, help='Number of labeled data')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--train-iteration', type=int, default=1024,help='Number of iteration per epoch')
    parser.add_argument('--weight_decay', type=float, default=4e-4,help='model weight decay')

    args = parser.parse_args()

    wandb.config.update(args)
    wandb.run.name = args.experiment
    wandb.run.save()
    args.loader = loader_CIFAR10(args)
    args.model = WideResNet(num_classes=10).cuda(args.gpu_device)
    wandb.watch(args.model)
    gc.collect()
    training(args)
    wandb.finish()

# %% run
if __name__ == "__main__":
    main()