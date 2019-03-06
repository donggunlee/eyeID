import argparse
import errno
import json
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from loader_featset import FeatLoader, FeatDataset, BucketingSampler

from model import StackedBRNN, supported_rnns
from utils import AverageMeter, _get_variable_nograd, _get_variable_volatile, _get_variable

import pdb
import scipy
import scipy.io as sio

def str2bool(v):
    return v.lower() in ('true', '1')


parser = argparse.ArgumentParser(description='Eye movement')
parser.add_argument('--expnum', type=int, default=0)


parser.add_argument('--train_manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_shuffle.csv')
parser.add_argument('--val_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/valid_shuffle.csv')
parser.add_argument('--test_manifest', metavar='DIR',
                    help='path to test manifest csv', default='data/test_shuffle.csv')

parser.add_argument('--batch_size', default=40, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in data-loading')
parser.add_argument('--max_value', default=400.0, type=float, help='maximum absolute value for feature normalization')


parser.add_argument('--rnn_hidden', default=128, type=int, help='Hidden size of RNNs')
parser.add_argument('--rnn_layers', default=2, type=int, help='Number of RNN layers')
parser.add_argument('--rnn_type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--dropout', default=0)

parser.add_argument('--fc_hidden', type=int, default=128)
parser.add_argument('--fc_layers', type=int, default=2)
parser.add_argument('--nClass', default=57)

parser.add_argument('--print_every', type = int, default=100)

parser.add_argument('--epochs', default=150, type=int, help='Number of training epochs')
parser.add_argument('--gpu', default=0, type=int, help='-1 : cpu')

parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--optim', default='adam', help='adam|sgd')

parser.add_argument('--log_params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model_path', default='', help='Location to save best validation model')  # set up this in code

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

if __name__ == '__main__':
    args, unparsed = parser.parse_known_args()
    if(len(unparsed) > 0):
        print(unparsed)
        assert(len(unparsed) == 0), 'length of unparsed option should be 0'

    save_folder = args.save_folder

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    loss_train, acc_valid = torch.Tensor(args.epochs), torch.Tensor(args.epochs)

    try:
        os.makedirs(save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Model Save directory already exists.')
        else:
            raise

    rnn_type = args.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"

    model = StackedBRNN(input_size=4, rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers, fc_hidden=args.fc_hidden,
                        fc_layers=args.fc_layers, rnn_type=supported_rnns[args.rnn_type], dropout=args.dropout,
                        nClass=args.nClass)
    # model.apply(weights_init) # use default initialization
    print(model)
    if args.gpu >=0:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss(size_average=False)

    avg_loss, start_epoch, start_iter = 0, 0, 0

    parameters = model.parameters()

    # Optimizer
    if(args.optim == 'adam'):
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
    elif(args.optim == 'sgd'):
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, nesterov=True)

    # Data loader
    train_dataset = FeatDataset(manifest_filepath=args.train_manifest, maxval=args.max_value)
    valid_dataset = FeatDataset(manifest_filepath=args.val_manifest, maxval=args.max_value)
    test_dataset = FeatDataset(manifest_filepath=args.test_manifest, maxval=args.max_value)

    train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    train_sampler.shuffle()
    train_loader = FeatLoader(train_dataset, num_workers=args.num_workers, batch_sampler=train_sampler)
    valid_loader = FeatLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = FeatLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Save model file for error check
    file_path = '%s/%d.pth.tar' % (save_folder, args.expnum)  # always overwrite recent epoch's model
    torch.save(model.state_dict(), file_path)

    train_acc_iter = []
    train_acc_set_epoch = torch.empty(args.epochs, 1, dtype=torch.float)
    train_loss_iter = []
    train_loss_set_epoch = torch.empty(args.epochs, 1, dtype=torch.float)

    valid_acc = []
    valid_acc_set_epoch = torch.empty(args.epochs, 1, dtype=torch.float)
    valid_loss = []
    valid_loss_set_epoch = torch.empty(args.epochs, 1, dtype=torch.float)

    test_acc = []
    test_acc_set_epoch = torch.empty(args.epochs, 1, dtype=torch.float)
    test_loss = []
    test_loss_set_epoch = torch.empty(args.epochs, 1, dtype=torch.float)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        end = time.time()
        losses.reset()

        # training
        correct_train = 0
        total_train = 0

        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break
            # measure data loading time
            data_time.update(time.time() - end)

            # load data
            input, target, trial = data
            input = _get_variable_nograd(input, cuda=True)
            target = _get_variable_nograd(target, cuda=True)
            trial = _get_variable_nograd(trial, cuda=True)
            output = model(input, trial)

            # Forward
            # print(target)
            loss = criterion(output, target)
            loss = loss / input.size(0)  # average the loss by minibatch
            losses.update(loss.data[0], input.size(0))

            train_loss_iter.append(losses.val.item())  # save training loss

            # measure accuracy
            pred = output.data.max(1)[1]
            pred = pred.type(torch.cuda.FloatTensor)
            target = target.type(torch.cuda.FloatTensor)
            correct_train += (pred == target).sum().item()
            total_train += input.size(0)

            # Backprop
            model.zero_grad()
            loss.backward()

            # Update
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_every == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {minibatch_acc:.3f}\t'.format((epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time, data_time=data_time, loss=losses, minibatch_acc=correct_train / total_train * 100))

            del loss
            del output

        train_acc_epoch = correct_train / total_train * 100
        print('Training Summary Epoch: [{0}]\t'
              'Average loss {loss:.3f}\t'
              'Average acc {acc:.3f} ({1} / {2})\t'.format(epoch + 1, correct_train, total_train, loss=losses.avg, acc=train_acc_epoch))

        train_acc_set_epoch[epoch] = train_acc_epoch
        train_loss_set_epoch[epoch] = losses.avg.item()

        # validation
        start_iter = 0  # Reset start iteration for next epoch
        model.eval()
        losses.reset()

        correct_valid = 0
        total_valid = 0
        # for i, (data) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        for i, (data) in enumerate(valid_loader, start=start_iter):
            # load data
            input, target, trial = data
            input = _get_variable_volatile(input, cuda=True)
            target = _get_variable_volatile(target, cuda=True)
            trial = _get_variable_volatile(trial, cuda=True)

            # Forward
            output = model(input, trial)
            loss = criterion(output, target)
            loss = loss / input.size(0)  # average the loss by minibatch
            losses.update(loss.data[0], input.size(0))

            valid_loss.append(losses.val.item())  # save validation loss

            # measure accuracy
            pred = output.data.max(1)[1]
            pred = pred.type(torch.cuda.FloatTensor)
            target = target.type(torch.cuda.FloatTensor)
            correct_valid += (pred == target).sum().item()
            total_valid += input.size(0)

        valid_acc_epoch = correct_valid / total_valid * 100

        print('Validation Summary Epoch: [{0}]\t'
              'Average loss {loss:.3f}\t'
              'Average acc {acc:.3f} ({1} / {2})\t'.format(epoch + 1, correct_valid, total_valid, loss=losses.avg, acc=valid_acc_epoch))

        valid_acc_set_epoch[epoch] = valid_acc_epoch
        valid_loss_set_epoch[epoch] = losses.avg.item()

        best_valid_acc = max(valid_acc_set_epoch[0:epoch+1])

        if valid_acc_set_epoch[epoch] >= best_valid_acc:
            file_path = '%s/%d.pth.tar' % (save_folder, args.expnum)  # overwrite the best epoch's model
            torch.save(model.state_dict(), file_path)

        #file_path = '%s/%d.pth.tar' % (save_folder, args.expnum)  # always overwrite recent epoch's model
        #torch.save(model.state_dict(), file_path)

        # test
        start_iter = 0  # Reset start iteration for next epoch
        model.eval()
        losses.reset()

        correct_test = 0
        total_test = 0
        # for i, (data) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        for i, (data) in enumerate(test_loader, start=start_iter):
            # load data
            input, target, trial = data
            input = _get_variable_volatile(input, cuda=True)
            target = _get_variable_volatile(target, cuda=True)
            trial = _get_variable_volatile(trial, cuda=True)

            # Forward
            output = model(input, trial)
            loss = criterion(output, target)
            loss = loss / input.size(0)  # average the loss by minibatch
            losses.update(loss.data[0], input.size(0))

            test_loss.append(losses.val.item())  # save validation loss

            # measure accuracy
            pred = output.data.max(1)[1]
            pred = pred.type(torch.cuda.FloatTensor)
            target = target.type(torch.cuda.FloatTensor)
            correct_test += (pred == target).sum().item()
            total_test += input.size(0)

        test_acc_epoch = correct_test / total_test * 100

        print('Test Summary Epoch: [{0}]\t'
              'Average loss {loss:.3f}\t'
              'Average acc {acc:.3f} ({1} / {2})\t'.format(epoch + 1, correct_test, total_test, loss=losses.avg, acc=test_acc_epoch))

        test_acc_set_epoch[epoch] = test_acc_epoch
        test_loss_set_epoch[epoch] = losses.avg.item()

        print("Shuffling batches...")
        train_sampler.shuffle()

        train_loss_set_epoch_npy = train_loss_set_epoch.numpy()
        train_acc_set_epoch_npy = train_acc_set_epoch.numpy()

        valid_loss_set_epoch_npy = valid_loss_set_epoch.numpy()
        valid_acc_set_epoch_npy = valid_acc_set_epoch.numpy()

        test_loss_set_epoch_npy = test_loss_set_epoch.numpy()
        test_acc_set_epoch_npy = test_acc_set_epoch.numpy()

        matfilename = 'exp_RNN' + str(args.rnn_layers) + '_' + str(args.rnn_hidden) + '_FC' + str(args.fc_layers) + '_' + str(args.fc_hidden)
        sio.savemat(matfilename, dict(train_loss=train_loss_set_epoch_npy, train_acc=train_acc_set_epoch_npy, valid_loss=valid_loss_set_epoch_npy, valid_acc=valid_acc_set_epoch_npy, test_loss=test_loss_set_epoch_npy, test_acc=test_acc_set_epoch_npy))
