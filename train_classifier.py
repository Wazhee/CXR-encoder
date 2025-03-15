import os
from torchvision import transforms as torch_transforms
from PIL import Image
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from torchvision import models
from torch.utils.data import Dataset
import torch
import torchvision.transforms as torch_transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import h5py
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import tensorflow as tf

train_config = {
               'batch_size': 64,
               'input_size': (256, 256),
               'n_epochs': 15,
               'nof_workers' : 8,
               'optim': torch.optim.Adam,
               'weighted_bce': False,
               'optim_kwargs': {'lr': 0.001, 'weight_decay': 0.0},
               'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
               'scheduler_kwargs': {'factor': 0.1, 'patience': 3, 'mode': 'max'},
               'early_stopping': 3,
               'experiment_name': "Cardiomegaly",
               'dataset_path': "/workspace/jiezy/CXR/datasets/cxpt_mdb/",
               'output_path': "/workspace/jiezy/CXR/CXR-encoder/results/"
                }

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256, labels=False, filter_label=None):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        self.PRED_LABEL = [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices"]

        if filter_label:
          if filter_label not in self.PRED_LABEL:
            raise Exception("Unrecognized label")
          self.filter_label = self.PRED_LABEL.index(filter_label)
        else:
          self.filter_label = None

        self.resolution = resolution
        self.transform = transform
        self.labels = labels

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
    
    def pos_weight(self):
        labels = np.ndarray(shape=(self.length, 2), dtype=np.uint8)
        
        with self.env.begin(write=False) as txn:
          for idx in range(self.length):
            label_key = f"{self.resolution}-{str(idx).zfill(5)}-label".encode("utf-8")
            label_bytes = txn.get(label_key)
            label = np.frombuffer(label_bytes, dtype=np.uint8).copy().astype(np.float32)        
            labels[idx, :] = label

        num_positives = torch.sum(torch.tensor(labels), dim=0)
        num_negatives = self.length - num_positives
        pos_weight  = num_negatives / num_positives
        
        return pos_weight
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
            if self.labels:
                label_key = f"{self.resolution}-{str(index).zfill(5)}-label".encode("utf-8")
                label_bytes = txn.get(label_key)
                label = np.frombuffer(label_bytes, dtype=np.uint8).copy().astype(np.float32)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)

        if self.transform is not None:
          img = self.transform(img)

        if self.filter_label is not None:
          label = label[self.filter_label]
          label = np.array([0, 1]) if label == 1.0 else np.array([1, 0])

        if self.labels:
            return img, label.astype(np.float32)
        return img
    
    
def get_datalaoders(path_to_data, **train_config):
    """ Returns data loaders of given dataset

    Arguments:
        - dataset (string): 'chexpert', 'brixia', 'combined'
        - path_to_data (string): path to the dataset
        - train_config (dict): dictionary containing parameters
    Returns:
        - train_loader (torch.utils.data.DataLoader)
        - val_loader (torch.utils.data.DataLoader)
        - test_loader (torch.utils.data.DataLoader)
    """

    input_size = train_config['input_size']
    batch_size = train_config['batch_size']

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = {
        'train': torch_transforms.Compose([
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean, std)
        ]),
        'val': torch_transforms.Compose([
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean, std)
        ]),
    }
    
    image_data = {
        'train': MultiResolutionDataset(os.path.join(train_config["dataset_path"]),
                           transform=data_transforms['train'], labels=True, filter_label=train_config['experiment_name']),
        'val': MultiResolutionDataset(os.path.join(train_config["dataset_path"], "valid/"),
                         transform=data_transforms['val'], labels=True, filter_label=train_config['experiment_name']),
        'test': MultiResolutionDataset(os.path.join(train_config["dataset_path"], "valid/"),
                         transform=data_transforms['val'], labels=True, filter_label=train_config['experiment_name']
                         )
    }
    
    num_workers = train_config['nof_workers']
    train_loader = torch.utils.data.DataLoader(
        image_data['train'],
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(
        image_data['val'],
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers)
    
    test_loader = torch.utils.data.DataLoader(
        image_data['test'],
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers)

    return train_loader, val_loader, test_loader

def train_chexpert(model, train_loader, val_loader, **train_config):
    """
    Arguments:
        - model (torch.nn.Module): Pytorch model
        - train_loader (torch.utils.data.DataLoader): Data loader with training set
        - val_loader (torch.utils.data.DataLoader): Data loader with validation set
        - train_config (dict): Dictionary of train parameters
    Returns:
        - (string): Path of the trained model
    """

#     device = ("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f"Process running on: {device}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    early_stopping = train_config.get('early_stopping', None)
    experiment_name = train_config['experiment_name']
    n_epochs = train_config['n_epochs']
    criterion = train_config['criterion']
    optim = train_config['optim'](model.parameters(), **train_config['optim_kwargs'])
    scheduler = train_config['scheduler'](optim, **train_config['scheduler_kwargs'])
    log_path = train_config['output_path'] + f"{experiment_name}"

    os.makedirs(log_path, exist_ok=True)

    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)

    best_auc = None
    early_stopping_val_loss = None

    model.to(device)
    criterion.to(device)
    for i_epoch in range(n_epochs):

        epoch_train_loss = 0
        epoch_val_loss = 0

        model = model.train()
        for data in tqdm(train_loader):
            x, y_target = data

            optim.zero_grad()

            x, y_target = x.to(device), y_target.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y_target)

            loss.backward()
            optim.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= num_train_batches

        model = model.eval()
        num_correct = 0
        num_examples = 0
        y_target_list = []
        y_raw_pred_list = []
        y_pred_list = []

        with torch.no_grad():
            for data in tqdm(val_loader):
                x, y_target = data

                x, y_target = x.to(device), y_target.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y_target)
                epoch_val_loss += loss.item()

                # Apply sigmoid to pred for metrics
                y_pred = torch.sigmoid(y_pred)
                y_raw_pred_list.extend(y_pred.cpu().tolist())

                y_pred[torch.where(y_pred > 0.5)] = 1.0
                y_pred[torch.where(y_pred <= 0.5)] = 0

                y_pred_list.extend(y_pred.cpu().tolist())
                y_target_list.extend(y_target.cpu().tolist())
                num_correct += torch.sum(y_pred == y_target).item()
                num_examples += y_target.shape[0] * y_target.shape[1]

        epoch_val_loss /= num_val_batches
        epoch_val_accuracy = num_correct / num_examples

        # Compute precision, recall F1-score and support for validations set
        epoch_prec, epoch_recall, epoch_f1, epoch_support = precision_recall_fscore_support(y_target_list, y_pred_list,
                                                                                            average="macro")

        # Calculate average auc
        y_target_list = torch.tensor(y_target_list)
        y_raw_pred_list = torch.tensor(y_raw_pred_list)
        epoch_auc = roc_auc_score(y_target_list, y_raw_pred_list)
        
        print("epoch %d, loss %.2f, acc %.2f, prec %.2f, f1 %.2f, " % (i_epoch, epoch_val_loss, epoch_val_accuracy, epoch_prec, epoch_f1))

        if best_auc is None or best_auc < epoch_auc:
            torch.save(model.state_dict(), f"{log_path}/model.pth")
            best_auc = epoch_auc

        if early_stopping is not None and i_epoch % early_stopping == 0:
            if early_stopping_val_loss != None and early_stopping_val_loss < epoch_val_loss:
                break
            else:
                # torch.save(model.state_dict(), f"{log_path}/checkpoint_{i_epoch}.pth")
                early_stopping_val_loss = epoch_val_loss

        scheduler.step(epoch_auc)

    return f"{log_path}/model.pth"


def test_chexpert(model, test_loader, **train_config):
    """ Tests a given model and saves results to tensorbaord

    Arguments:
        - model (torch.nn.Module): Pytorch model
        - test_loader (torch.utils.data.DataLoader): Data loader with test set
        - train_config (dict): Dictionary of train parameters
    Returns:
    """

    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Process running on: {device}")

    num_correct = 0
    num_examples = 0
    y_targets = []
    y_raw_preds = []
    y_preds = []

    nb_classes = 2
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    model.to(device)
    model = model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            x, y_target = data
            x, y_target = x.to(device), y_target.to(device)
            y_pred = model(x)

            _, preds = torch.max(y_pred, 1)
            for t, p in zip(y_target.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            # Apply sigmoid to pred for metrics
            y_pred = torch.sigmoid(y_pred)
            y_raw_preds.extend(y_pred.cpu().tolist())

            y_pred[torch.where(y_pred > 0.5)] = 1.0
            y_pred[torch.where(y_pred <= 0.5)] = 0

            y_preds.extend(y_pred.cpu().tolist())
            
            y_targets.extend(y_target.cpu().tolist())
            num_correct += torch.sum(y_pred == y_target).item()
            num_examples += y_target.shape[0] * y_target.shape[1]

    test_accuracy = num_correct / num_examples

    # Compute precision, recall F1-score and support for test set
    test_prec, test_recall, test_f1, test_support = precision_recall_fscore_support(y_targets, y_preds,
                                                                                    average="macro")

    print(confusion_matrix)
    print(confusion_matrix.diag()/confusion_matrix.sum(1))
    print(test_prec, test_recall, test_f1, test_support, test_accuracy)
    
    
def evaluation():
    pretrained_model_path = "<path>/model.pth"
    # Load trained model
    model.load_state_dict(torch.load(pretrained_model_path))
    # Test model
    test_chexpert(model, val_loader, **train_config)


def train():
    # Create data loaders
    train_loader, val_loader, test_loader = get_datalaoders(path_to_data="", **train_config)

    # Use weighted BCE
    if train_config['weighted_bce']:
        train_config['criterion'] = torch.nn.BCEWithLogitsLoss(pos_weight=train_loader.dataset.pos_weight())
    else:
        train_config['criterion'] = torch.nn.BCEWithLogitsLoss()

    # Load pretrained model on ImageNet
    model = models.densenet121(pretrained=True)
    # Change last layer to CheXpert targets
    model.classifier = torch.nn.Linear(1024, 2)

    # Train model
    pretrained_model_path = train_chexpert(model, train_loader, val_loader, **train_config)
    
if __name__ == "__main__":
    train()