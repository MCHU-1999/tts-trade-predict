import os
from collections import OrderedDict
import argparse
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
import model.resnet_cbam as resnet_cbam
from trainer.trainer import Trainer
from utils.logger import Logger
from PIL import Image
from torchnet.meter import ClassErrorMeter, MSEMeter
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from dataset import KlineDataset, split_and_load

# Constants -------------------------------
BATCH_SIZE = 64
MODEL_TYPE = 'resnet18'


# -----------------------------------------

# def load_state_dict(model_dir, is_multi_gpu):
#     state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)['state_dict']
#     if is_multi_gpu:
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             name = k[7:]       # remove `module.`
#             new_state_dict[name] = v
#         return new_state_dict
#     else:
#         return state_dict

def main(args):
    if 0 == len(args.resume):
        logger = Logger('./logs/'+ MODEL_TYPE +'.log')
    else:
        logger = Logger('./logs/'+ MODEL_TYPE +'.log', True)

    logger.append(vars(args))

    if args.display:
        writer = SummaryWriter()
    else:
        writer = None

    # prepare data
    x_train, x_test, y_train, y_test = split_and_load(test_size=0.2, random_state=33)
    train_dataset = KlineDataset(x_train, y_train)
    val_dataset = KlineDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)

    if args.debug:
        x, y =next(iter(train_loader))
        logger.append([x, y])

    cudnn.benchmark = True
    my_model = resnet_cbam.resnet18_cbam()
    is_use_cuda = torch.cuda.is_available()

    #my_model.apply(fc_init)
    if is_use_cuda:
        my_model = my_model.cuda()

    loss_fn = [nn.CrossEntropyLoss()]
    optimizer = optim.SGD(my_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

    # metric = [ClassErrorMeter([1,5], True)]
    metric = [MSEMeter(root=False)]
    start_epoch = 0
    num_epochs = 50

    my_trainer = Trainer(my_model, MODEL_TYPE, loss_fn, optimizer, lr_schedule, 500, is_use_cuda, train_loader, \
                        val_loader, metric, start_epoch, num_epochs, args.debug, logger, writer)
    my_trainer.fit()
    logger.append('Optimize Done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--resume', default='', type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--debug', action='store_true', dest='debug', default=False,
                        help='trainer debug flag')
    # parser.add_argument('-g', '--gpu', default='0', type=str,
    #                     help='GPU ID Select')                    
    # parser.add_argument('-d', '--data_root', default='./datasets',
    #                      type=str, help='data root')
    # parser.add_argument('-t', '--train_file', default='./datasets/train.txt',
    #                      type=str, help='train file')
    # parser.add_argument('-v', '--val_file', default='./datasets/val.txt',
    #                      type=str, help='validation file')
    # parser.add_argument('-m', '--model', default='resnet101',
    #                      type=str, help='model type')
    # parser.add_argument('--batch_size', default=12,
    #                      type=int, help='model train batch size')
    parser.add_argument('--display', action='store_true', dest='display', default=True,
                        help='Use TensorboardX to Display')
    args = parser.parse_args()

    main(args)
