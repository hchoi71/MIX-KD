from __future__ import print_function, division

import sys
import time
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F

from .util import AverageMeter, accuracy

## Partial Mixup
def PMU(inputs, targets, percent, beta_a, mixup=True):
    batch_size = inputs.shape[0]

    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]
    targets1_1 = targets1.unsqueeze(1)

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2]
    targets2 = targets[rp2]
    targets2_1 = targets2.unsqueeze(1)

    y_onehot = torch.FloatTensor(batch_size, 100)
    y_onehot.zero_()
    targets1_oh = y_onehot.scatter_(1, targets1_1, 1)

    y_onehot2 = torch.FloatTensor(batch_size, 100)
    y_onehot2.zero_()
    targets2_oh = y_onehot2.scatter_(1, targets2_1, 1)

    if mixup is True:
        a = numpy.random.beta(beta_a, beta_a, [batch_size, 1])
    else:
        a = numpy.ones((batch_size, 1))
    
    ridx = torch.randint(0,batch_size,(int(a.shape[0] * percent),))
    a[ridx] = 1.

    b = numpy.tile(a[..., None, None], [1, 3, 32, 32])

    inputs1 = inputs1 * torch.from_numpy(b).float()
    inputs2 = inputs2 * torch.from_numpy(1 - b).float()

    c = numpy.tile(a, [1, 100])

    targets1_oh = targets1_oh.float() * torch.from_numpy(c).float()
    targets2_oh = targets2_oh.float() * torch.from_numpy(1 - c).float()

    inputs_shuffle = inputs1 + inputs2
    targets_shuffle = targets1_oh + targets2_oh

    return inputs_shuffle, targets_shuffle


## Full Mixup
def FMU(inputs, targets, beta_a, mixup=True):
    batch_size = inputs.shape[0]

    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]
    targets1_1 = targets1.unsqueeze(1)

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2]
    targets2 = targets[rp2]
    targets2_1 = targets2.unsqueeze(1)

    y_onehot = torch.FloatTensor(batch_size, 100)
    y_onehot.zero_()
    targets1_oh = y_onehot.scatter_(1, targets1_1, 1)

    y_onehot2 = torch.FloatTensor(batch_size, 100)
    y_onehot2.zero_()
    targets2_oh = y_onehot2.scatter_(1, targets2_1, 1)

    if mixup is True:
        a = numpy.random.beta(beta_a, beta_a, [batch_size, 1])
    else:
        a = numpy.ones((batch_size, 1))

    b = numpy.tile(a[..., None, None], [1, 3, 32, 32])

    inputs1 = inputs1 * torch.from_numpy(b).float()
    inputs2 = inputs2 * torch.from_numpy(1 - b).float()

    c = numpy.tile(a, [1, 100])

    targets1_oh = targets1_oh.float() * torch.from_numpy(c).float()
    targets2_oh = targets2_oh.float() * torch.from_numpy(1 - c).float()

    inputs_shuffle = inputs1 + inputs2
    targets_shuffle = targets1_oh + targets2_oh

    return inputs_shuffle, targets_shuffle



def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        input, target, index = data
        
        data_time.update(time.time() - end)
        
        if opt.pmixup:
            inputs_shuffle, targets_shuffle = PMU(input, target, opt.partmixup, opt.beta_a)
        else:            
            inputs_shuffle, targets_shuffle = FMU(input, target, opt.beta_a)
            
        inputs_shuffle = inputs_shuffle.float()
        if torch.cuda.is_available():
            inputs_shuffle = inputs_shuffle.cuda()
            targets_shuffle = targets_shuffle.cuda()
            index = index.cuda()

        # ===================forward=====================
        preact = False
        feat_s, logit_s = model_s(inputs_shuffle, is_feat=True, preact=preact)

        with torch.no_grad():
            feat_t, logit_t = model_t(inputs_shuffle, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            # cls + kl div
            loss_cls = criterion_cls(logit_s, target)
            kd_loss = criterion_div(logit_s, logit_t)
            loss = opt.gamma * loss_cls + opt.alpha * kd_loss
        elif opt.distill == 'mixstd':
            loss_cls, kd_loss = criterion_kd(logit_s, logit_t, targets_shuffle)
            loss = loss_cls + kd_loss
        else:
            raise NotImplementedError(opt.distill)

        
        _, targets = torch.max(targets_shuffle.data, 1)
        acc1, acc5 = accuracy(logit_s, targets, topk=(1, 5))
        losses.update(loss.item(), inputs_shuffle.size(0))
        kd_losses.update(kd_loss.item(), inputs_shuffle.size(0))
        top1.update(acc1[0], inputs_shuffle.size(0))
        top5.update(acc5[0], inputs_shuffle.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'KD_Loss {kd_loss.val:.4f} ({kd_loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, kd_loss=kd_losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    #vis_feator = FeatureVisualizer()
    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            # output std scaling 
            std = torch.std(output, dim=-1, keepdim=True)
            output = output/std
        
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

