import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from nn_modules.convolutional_siamese_network import ConvolutionalSiamesNetwork, ConvolutionalSiamesNetworkSmall
from nn_modules.trainer import Trainer
from loss_functions.triplet_loss import TripletLoss
from torch_utils import get_device
import file_utils
import matplotlib.pyplot as plt
from image_utils import FaceTensorTransform

eps = 1e-8

def get_transforms(input_dim):
    return transforms.Compose([transforms.Resize((input_dim,input_dim)),
                            transforms.ToTensor()
                            ])



def train_siamese_network(train_loader, val_loader, embedding_dim=3, 
                         margin=1.0, lr=0.0001, num_epochs=100, device='cuda'):

    model = ConvolutionalSiamesNetwork(embedding_dim=embedding_dim).to(device)
    criterion = TripletLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    trainer = Trainer(model, criterion, optimizer, device)

    loss_history = []
    metrics_history = []
    last_validation_pairs_distance = []
    last_val_loss = float('inf')
    last_acc = 0

    for epoch in range(num_epochs):
        train_loss, batch_loss_history = trainer.train_epoch(train_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')

        if (epoch + 1) % 10 == 0:
            # Evaluate
            last_val_loss, val, far, last_acc, last_validation_pairs_distance = trainer.evaluate(val_loader)
            # Print progress
            print(f'Validation Loss: {last_val_loss:.4f}')
            print(f'VAL: {val:.4f}')
            print(f'FAR: {far:.4f}')
            print(f'ACC: {last_acc:.4f}')

            metrics_history.append({ 'val_loss': last_val_loss, 'val': val, 'far': far, 'acc': last_acc})  
        
        loss_history.append({'epoch_mean': train_loss, 'batch_history': batch_loss_history })
    
    return model, last_acc, loss_history, metrics_history, last_validation_pairs_distance

def run_cross_validation_training(dataset, embedding_dim=3, 
                         margin=1.0, lr=0.01, num_epochs=10, batch_size=100, device='cuda'):
    kfold = KFold(n_splits=5)
    best_model = {}
    best_acc = 0
    best_metrics_history = []
    best_loss_history = []
    best_pairs_distance = []
    model_file_name = 'trained_siamese_model_2'
    pairs_file_name = model_file_name + '_pairs.txt'
    metrics_file_name = model_file_name + '_metrics.txt'
    loss_history_file_name = model_file_name + '_train_loss.txt'
    for train_index, test_index in kfold.split(dataset):
        train_sampler = SubsetRandomSampler(train_index)
        test_sampler = SubsetRandomSampler(test_index)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=test_index.size, sampler=test_sampler)

        model, acc, loss_history, metrics_history, last_validation_pairs_distance = train_siamese_network(train_loader, test_loader, embedding_dim, margin, lr, num_epochs, device)

        if best_acc == 0 or best_acc < acc:
            best_model = model
            best_acc = acc
            best_metrics_history = metrics_history
            best_loss_history = loss_history
            best_pairs_distance = last_validation_pairs_distance

    model_scripted = torch.jit.script(best_model)
    model_scripted.save(model_file_name + '.pt')
    file_utils.write_metric_files(pairs_file_name, metrics_file_name, loss_history_file_name, best_pairs_distance, best_loss_history, best_metrics_history)

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

folder_dataset = datasets.ImageFolder(root="./data/faces/tratado/", transform=get_transforms(128))
run_cross_validation_training(folder_dataset, embedding_dim=128, lr=0.0001, margin=0.1, num_epochs=100, batch_size=50, device=get_device())