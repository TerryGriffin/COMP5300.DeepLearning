# COMP.5300 Deep Learning
# Terry Griffin
#
# Program for testing reducing the input of a ResNet model from three
# channels to one channel, using image classification on the Fashion MNIST dataset
# This script is a modified version of the example training script from
# the Pytorch tutorials.

import time
import os
import copy

import torch
import torchvision
from torchvision import datasets, models, transforms

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.utils.tensorboard import SummaryWriter

from to_grayscale_model import resnet_grayscale_model

def train_model(model, criterion, optimizer, scheduler,
                dataloaders, dataset_sizes, device,
                train_writer, val_writer,
                num_epochs=25):
    since = time.time()


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_start = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_elapsed = time.time() - epoch_start
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f} Elapsed: {:.2f}'.format(
                phase, epoch_loss, epoch_acc, epoch_elapsed))

            if phase == 'train':
                train_writer.add_scalar('loss', epoch_loss, epoch)
                train_writer.add_scalar('acc', epoch_acc, epoch)
            else:
                val_writer.add_scalar('loss', epoch_loss, epoch)
                val_writer.add_scalar('acc', epoch_acc, epoch)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def add_graph_to_tensorboard(model, data, writer):
    dataiter = iter(data)
    images, labels = dataiter.next()
    writer.add_graph(model, images)


def main(log_dir, log_name, log_suffix, batch_size, num_epochs):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.449], [0.226]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.449], [0.226]),
    ]),
    }

    # constant for classes
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # datasets
    trainset = torchvision.datasets.FashionMNIST('./data',
                                                 download=True,
                                                 train=True,
                                                 transform=data_transforms['train'])
    testset = torchvision.datasets.FashionMNIST('./data',
                                                download=True,
                                                train=False,
                                                transform=data_transforms['val'])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    dataloaders = {'train' : trainloader,
                   'val' : testloader }
    dataset_sizes = {x: len(dataloaders) for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device ', device)
    train_writer = SummaryWriter(os.path.join(log_dir, log_name + '_train_' + log_suffix))
    val_writer = SummaryWriter(os.path.join(log_dir, log_name + '_val_' + log_suffix))

    model_ft = models.resnet101(pretrained=True)
    resnet_grayscale_model(model_ft)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to match our dataset.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(classes))

    #add_graph_to_tensorboard(model_ft, dataloaders['train'], train_writer)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           dataloaders, dataset_sizes, device, train_writer, val_writer,
                           num_epochs)

    print('Done')

if __name__ == '__main__':
    main('runs', 'resnet_reduce_fashion_mnist', '1', 4, 2)
