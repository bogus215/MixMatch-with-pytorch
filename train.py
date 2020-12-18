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

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(args.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    ema_optimizer= WeightEMA(model = args.model, ema_model= args.model_ema,lr=args.learning_rate, alpha=args.ema_decay)
    early_stopping = EarlyStopping(patience=10, verbose=False, path=f'./parameter/{args.experiment}.pth')

    console = Console()

    for epoch in range(args.epoch+1):
        print("\n===> epoch %d" % epoch)

        labeled_train_iter = iter(args.loader.train_iter_labeld)
        unlabeled_train_iter = iter(args.loader.train_iter_unlabeld)

        args.model.train()
        with tqdm(total=args.train_iteration, leave=False, dynamic_ncols=True) as pbar:
            for batch_idx in range(args.train_iteration):

                losses = AverageMeter()
                losses_x = AverageMeter()
                losses_u = AverageMeter()
                ws = AverageMeter()

                try:
                    inputs_x, targets_x = labeled_train_iter.next()
                    inputs_x = inputs_x.cuda(args.gpu_device)
                    targets_x = targets_x.cuda(args.gpu_device)
                except:
                    labeled_train_iter = iter(args.loader.train_iter_labeld)
                    inputs_x, targets_x = labeled_train_iter.next()
                    inputs_x = inputs_x.cuda(args.gpu_device)
                    targets_x = targets_x.cuda(args.gpu_device)

                try:
                    (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
                    inputs_u = inputs_u.cuda(args.gpu_device)
                    inputs_u2 = inputs_u2.cuda(args.gpu_device)
                except:
                    unlabeled_train_iter = iter(args.loader.train_iter_unlabeld)
                    (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
                    inputs_u = inputs_u.cuda(args.gpu_device)
                    inputs_u2 = inputs_u2.cuda(args.gpu_device)

                batch_size = inputs_x.size(0)
                targets_x = torch.zeros(batch_size, 10).cuda(args.gpu_device).scatter_(1, targets_x.view(-1, 1).long(), 1)

                '''We do not propagate gradients through computing the guessed labels, as is standard'''
                with torch.no_grad():
                    outputs_u = args.model(inputs_u)
                    outputs_u2 = args.model(inputs_u2)
                    p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
                    pt = p ** (1 / args.T) # sharpen
                    targets_u = pt / pt.sum(dim=1, keepdim=True)
                    targets_u = targets_u.detach()

                # mixup
                all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
                all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

                l = np.random.beta(args.alpha, args.alpha)
                l = max(l, 1 - l)

                idx = torch.randperm(all_inputs.size(0))

                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]

                mixed_input = l * input_a + (1 - l) * input_b
                mixed_target = l * target_a + (1 - l) * target_b

                # interleave labeled and unlabeled samples between batches to get correct batchnorm calculation
                mixed_input = list(torch.split(mixed_input, batch_size))
                mixed_input = interleave(mixed_input, batch_size)

                logits = [args.model(mixed_input[0])]
                for input in mixed_input[1:]:
                    logits.append(args.model(input))

                # put interleaved samples back
                logits = interleave(logits, batch_size)
                logits_x = logits[0]
                logits_u = torch.cat(logits[1:], dim=0)

                Lx, Lu, w = train_criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:],
                                      epoch + batch_idx / args.train_iteration, args.lambda_u, epochs = args.epoch)

                loss = Lx + w * Lu

                # record loss
                losses.update(loss.item(), inputs_x.size(0))
                losses_x.update(Lx.item(), inputs_x.size(0))
                losses_u.update(Lu.item(), inputs_x.size(0))
                ws.update(w, inputs_x.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema_optimizer.step()

                pbar.update(1)

        train_loss, train_loss_x, train_loss_u = losses.avg, losses_x.avg, losses_u.avg

        with tqdm(total=len(args.loader.train_iter_labeld), leave=False, dynamic_ncols=True) as pbar:

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            args.model_ema.eval()

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(args.loader.train_iter_labeld):
                    inputs = inputs.cuda(args.gpu_device)
                    targets = targets.cuda(args.gpu_device)

                    outputs = args.model_ema(inputs)
                    outputs = F.softmax(outputs, dim =1)
                    loss = criterion(outputs , targets.long())

                    # measure accuracy and record loss
                    prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                    losses.update(loss.item(), inputs.size(0))
                    top1.update(prec1.item(), inputs.size(0))
                    top5.update(prec5.item(), inputs.size(0))

                    pbar.update(1)

        _, train_acc = losses.avg, top1.avg

        with tqdm(total=len(args.loader.valid_iter), leave=False, dynamic_ncols=True) as pbar:

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            args.model_ema.eval()

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(args.loader.valid_iter):
                    inputs = inputs.cuda(args.gpu_device)
                    targets = targets.cuda(args.gpu_device)

                    outputs = args.model_ema(inputs)
                    outputs = F.softmax(outputs, dim =1)
                    loss = criterion(outputs,targets.long())

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

            args.model_ema.eval()

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(args.loader.test_iter):
                    inputs = inputs.cuda(args.gpu_device)
                    targets = targets.cuda(args.gpu_device)

                    outputs = args.model_ema(inputs)
                    outputs = F.softmax(outputs, dim =1)
                    loss = criterion(outputs, targets.long())

                    # measure accuracy and record loss
                    prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                    losses.update(loss.item(), inputs.size(0))
                    top1.update(prec1.item(), inputs.size(0))
                    top5.update(prec5.item(), inputs.size(0))

                    pbar.update(1)

            test_loss, test_acc = losses.avg, top1.avg

            console.print(f"Train [{epoch:>04}]/[{args.epoch:>03}]",f"Train acc:{train_acc:.4f}",f"valid acc:{val_acc:.4f}",f"Test acc:{test_acc}",sep='  |  ', style="Bold Cyan")
            console.print(f"Train loss:{train_loss:.4f}",f"train loss_x:{train_loss_x:.4f}",f"Train loss_u:{train_loss_u}",sep='  |  ', style="Bold Cyan")

            wandb.log({'Train acc': train_acc,
                       'Valid acc': val_acc,
                       'Test acc': test_acc,
                       'Train loss':train_loss,
                       'Train loss_x':train_loss_x,
                       'Train loss_u':train_loss_u
                       })
            early_stopping(-val_acc, args.model_ema)
            if early_stopping.early_stop:
                print('Early stopping')
                break

class WeightEMA(object):
    def __init__(self, model, ema_model,lr,alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

# %% main
def main():
    wandb.init(project='PyTorch MixMatch Training', reinit=True)
    parser = argparse.ArgumentParser(description="'PyTorch MixMatch Training'")
    parser.add_argument("--learning_rate", default=0.002, type=float, help="learning rate")
    parser.add_argument("--epoch", default=1024, type=int, help="number of max epoch")
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument("--gpu_device", default=0, type=int, help="the number of gpu to be used")
    parser.add_argument('--experiment', type=str, default='1218_Mixmatch', help='experiment name')
    parser.add_argument('--n_labeled', type=int, default=250, help='Number of labeled data')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--alpha', default=0.75, type=float, help ='beta distribution parameter')
    parser.add_argument('--lambda-u', default=75, type=float, help = 'unsupervised loss decay')
    parser.add_argument('--T', default=0.5, type=float ,help = 'pseudo label sharpen')
    parser.add_argument('--ema-decay', default=0.999, type=float, help ='Exponential moving average training')
    parser.add_argument('--train-iteration', type=int, default=1024,help='Number of iteration per epoch')
    parser.add_argument('--weight_decay', type=float, default=4e-4,help='model weight decay')

    args = parser.parse_args()

    wandb.config.update(args)
    wandb.run.name = args.experiment
    wandb.run.save()
    args.loader = loader_CIFAR10(args)
    args.model = WideResNet(num_classes=10).cuda(args.gpu_device)
    args.model_ema = WideResNet(num_classes=10).cuda(args.gpu_device)
    for param in args.model_ema.parameters():
        param.detach_()
    wandb.watch(args.model)
    gc.collect()
    training(args)
    wandb.finish()

# %% run
if __name__ == "__main__":
    main()