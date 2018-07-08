import time
import torch
from tensorboardX import SummaryWriter
from dataset import Meta, AudioSamplesDataset, FullClipDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from model import vgg19_bn, vgg16_bn
from tqdm import tqdm
import pandas as pd
from glob import glob
from transforms_wav import *
from torchvision.transforms import Compose
import os
import re
import csv
import torch.nn.functional as F

def train(path_to_files):
    batch_size = 128
    learning_rate = 1e-4
    weight_decay = 1e-2

    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vgg16_bn(pretrained=False, in_channels=1, num_classes=len(Meta.classes)).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    start_epoch = 0

    lr_scheduler_patience = 10
    lr_scheduler_gamma = 0.1

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lr_scheduler_patience,
                                                              factor=lr_scheduler_gamma)

    def get_lr():
        return optimizer.param_groups[0]['lr']

    writer = SummaryWriter()

    df = pd.read_csv(
        path_to_files + '/meta/meta.txt',
        header=None,
        sep='\t',
        names=['id', 'c1', 'c2', 'c3', 'label'],
        usecols=['id', 'label']
    )

    train_dataset = AudioSamplesDataset(glob(path_to_files + '/audio/*.wav'), df=df, transforms=ToMelSpectrogram())

    weights = train_dataset.make_weights_for_balanced_classes()
    sampler = WeightedRandomSampler(weights, len(weights))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=2)

    test_path = path_to_files + '/test/'
    files = [test_path + f for f in os.listdir(test_path) if not re.search(r'unknown.*\.wav$', f)]

    test_dataset = AudioSamplesDataset(files, df=df, transforms=ToMelSpectrogram())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

    def run_train_epoch(epoch):
        global global_step

        print("epoch %3d with lr=%.02e" % (epoch, get_lr()))
        phase = 'train'
        writer.add_scalar('%s/learning_rate' % phase, get_lr(), epoch)

        model.train()  # Set model to training mode

        running_loss = 0.0
        it = 0
        correct = 0
        total = 0

        pbar = tqdm(train_dataloader, unit="audios", unit_scale=train_dataloader.batch_size)
        for batch in pbar:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            it += 1
            global_step += 1
            running_loss += loss.data[0]
            pred = outputs.data.max(1, keepdim=True)[1]

            correct += pred.eq(targets.data.view_as(pred)).sum()
            total += targets.size(0)

            writer.add_scalar('%s/loss' % phase, loss.data[0], global_step)

            # update the progress bar
            pbar.set_postfix({
                'loss': "%.05f" % (running_loss / it),
                'acc': "%.02f%%" % (100 * correct / total)
            })

        accuracy = correct / total
        epoch_loss = running_loss / it
        writer.add_scalar('%s/accuracy' % phase, 100 * accuracy, epoch)
        writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

    def validate_on_test_set(epoch):
        global best_loss, global_step

        phase = 'valid'
        model.eval()

        running_loss = 0.0
        it = 0
        correct = 0
        total = 0

        pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)
        for batch in pbar:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # statistics
            it += 1
            global_step += 1
            running_loss += loss.data[0]
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).sum()
            total += targets.size(0)

            writer.add_scalar('%s/loss' % phase, loss.data[0], global_step)

            # update the progress bar
            pbar.set_postfix({
                'loss': "%.05f" % (running_loss / it),
                'acc': "%.02f%%" % (100 * correct / total)
            })

        accuracy = correct / total
        epoch_loss = running_loss / it
        writer.add_scalar('%s/accuracy' % phase, 100 * accuracy, epoch)
        writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

        checkpoint = {
            'epoch': epoch,
            'step': global_step,
            'state_dict': model.state_dict(),
            'loss': epoch_loss,
            'accuracy': accuracy,
            'optimizer': optimizer.state_dict(),
        }

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, 'checkpoints/ckpt.pth')

        del checkpoint

        return epoch_loss

    for epoch in range(start_epoch, 70):
        run_train_epoch(epoch)
        epoch_loss = validate_on_test_set(epoch)
        lr_scheduler.step(metrics=epoch_loss)


def create_submission(path_to_files):
    df = pd.read_csv(
        path_to_files + '/meta/meta.txt',
        header=None,
        sep='\t',
        names=['id', 'c1', 'c2', 'c3', 'label'],
        usecols=['id', 'label']
    )

    test_path = path_to_files + '/test/'
    files = glob(test_path + '*.wav')
    test_dataset = FullClipDataset(files, df=df, skip_unknonw=False, transforms=DataToMelSpectrogram())

    checkpoint = torch.load('checkpoints/ckpt.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vgg16_bn(pretrained=False, in_channels=1, num_classes=len(Meta.classes)).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.float()
    model.eval()
    with open('result.txt', "w+") as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        for sample in test_dataset:
            splits = sample['input'].to(device)
            outputs = model(splits)
            outputs = torch.mean(outputs, dim=0)
            outputs = F.softmax(outputs)
            score = outputs.data.max()
            prediction = outputs.data.argmax()
            writer.writerow([sample['name'], '%.3f' % round(score.item(), 3), Meta.classes[prediction]])


if __name__ == '__main__':
    best_loss = 10000
    global_step = 0
    train(path_to_files='.data')
    print("Training finished.")
    create_submission(path_to_files='.data')
    print("Submission created.")
