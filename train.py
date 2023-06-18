import os
from collections import OrderedDict
import argparse
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
import model.resnet_cbam as resnet_cbam
import model.transformer1d as transformer1d
from trainer.trainer import Trainer
from utils.logger import Logger
from PIL import Image
from torchnet.meter import ClassErrorMeter, MSEMeter
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from dataset import KlineDataset, KlineDataset_256, split_and_load

# Constants -------------------------------
BATCH_SIZE = 128
MODEL_TYPE = 'transformer1d'
DEVICE = 'mps'

START_EPOCH = 0
NUM_EPOCHS = 50

# -----------------------------------------

def load_state_dict(model_dir, is_multi_gpu):
    state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)['state_dict']
    if is_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]       # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

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
    x_train, x_test, y_train, y_test = split_and_load(test_size=0.1, random_state=33)
    train_dataset = KlineDataset_256(x_train, y_train)
    val_dataset = KlineDataset_256(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)

    # load model
    # state_dict = load_state_dict('./checkpoint/resnet18/Model_best.ckpt', False)

    if args.debug:
        x, y =next(iter(train_loader))
        logger.append([x, y])

    cudnn.benchmark = True
    # my_model = resnet_cbam.resnet34_cbam()
    my_model = transformer1d.Transformer1d()
    # my_model.load_state_dict(state_dict)
    gpu_exist = torch.cuda.is_available() or torch.backends.mps.is_available()
    device = torch.device(DEVICE)

    #my_model.apply(fc_init)
    # if gpu_exist:
    my_model = my_model.to(device)

    # loss_function
    loss_function = [nn.L1Loss()]
    metric = [MSEMeter(root=True)]

    # optimizer 
    optimizer = optim.SGD(my_model.parameters(), lr=0.001, momentum=0.8, weight_decay=1e-4)
    # optimizer = optim.Adam(my_model.parameters(), lr=0.0002)

    # scheduler
    lr_schedule = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # trainer
    my_trainer = Trainer(my_model, MODEL_TYPE, loss_function, optimizer, lr_schedule, 500, gpu_exist, device, train_loader, \
                        val_loader, metric, START_EPOCH, NUM_EPOCHS, args.debug, logger, writer)
    my_trainer.fit()
    logger.append('Optimize Done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--resume', default='', type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--debug', action='store_true', dest='debug', default=False,
                        help='trainer debug flag')
    parser.add_argument('--display', action='store_true', dest='display', default=True,
                        help='Use TensorboardX to Display')
    args = parser.parse_args()

    main(args)
