from transformers import BertTokenizer, BertModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from tqdm.auto import trange
import wandb

from sklearn.model_selection import train_test_split

import argparse
import json

import sys
sys.path.insert(1, '../../models')
sys.path.insert(1, '../data')
from sentiment_net import TextClassifier
from dataset import MyDataset, to_tens


def validate(model, val_loader, criterion, device, log=False):
    correct = 0
    total = 0
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels, embedding = data[0], data[1].to(device), data[2].to(device)
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            
            outputs = model((inputs['input_ids'], inputs['attention_mask']), embedding)
            labels = labels.long()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    accuracy = 100 * correct / total

    if log:
        wandb.log({"Val Loss": val_loss, "Val Accuracy": accuracy})
    
    return val_loss, accuracy

def main(config_file):
    # Load the JSON configuration file
    with open(config_file, "r") as f:
        config = json.load(f)

    # Extract values from the configuration file
    texts_data_path = config["data_paths"]["texts"]
    embeddings_data_path = config["data_paths"]["embeddings"]

    model_save_name = config["model"]["save_name"]
    model_save_path = config["model"]["save_path"]
    model_name = config["model"]["name"]
    num_classes = config["model"]["num_classes"]


    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    num_epochs = config["training"]["num_epochs"]
    test_size = config["training"]["test_size"]
    scheduler = config["training"]["scheduler"]
    freeze = config["training"]["freeze"]
    print_every = config["training"]["print_every"]
    val_step = config["training"]["val_step"]
    log = config["training"]["wandb"]

    device = torch.device(config["device"])

    # Print configuration
    print("Parsed Config Parameters:")
    print("-------------------------")
    print(f"Texts Data Path: {texts_data_path}")
    print(f"Embeddings Data Path: {embeddings_data_path}")
    print(f"Model Save Name: {model_save_name}")
    print(f"Model Save Path: {model_save_path}")
    print(f"Model Name: {model_name}")
    print(f"Number of Classes: {num_classes}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Test Size: {test_size}")
    print(f"Scheduler: {scheduler}")
    print(f"Freeze: {freeze}")
    print(f"Print Every: {print_every}")
    print(f"Validation Step: {val_step}")
    print(f"Logging to WandB: {log}")
    print(f"Device: {device}")
    print("-------------------------")
    
    # Init wandb if True
    if log:
        wandb.init(project="Crypto_sentiment_classification", 
           name="exp_1",
           config={
               "learning_rate": learning_rate,
               "weight_decay": None,
               "architecture": "BERT_FastText_NN",
               "dataset": "CoinDesk_CoinTelegraph",
               "epochs": num_epochs,
               "batch_size": batch_size,
               "scheduler": scheduler
               }
            )

    # Split data
    df = pd.read_csv(texts_data_path, index_col=0)
    embeddings = np.load(embeddings_data_path)
    X = df.title.values.flatten()
    y = df.label.values.flatten()
    train_texts, val_texts, train_labels, val_labels, train_embeddings, val_embeddings = train_test_split(X, y, embeddings, test_size=test_size, shuffle=True)

    # Load the training and validation datasets
    train_dataset = MyDataset(train_texts.tolist(), train_labels.tolist(), train_embeddings, model_name, BertTokenizer, to_tens())
    val_dataset = MyDataset(val_texts, val_labels, val_embeddings, model_name, BertTokenizer, to_tens())

    # Create data loaders for the datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextClassifier(BertModel, model_name, num_classes).to(device)

    # Initialize the optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if scheduler == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,25], gamma=0.1)

    if freeze == True:
        # freeze BERT parameters
        total_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)  
        print(f'\nTotal parameters number: {total_params}')
        print('Trainable parameters:')
        for i, (name, param) in enumerate(model.named_parameters()):
            if 'bert' in name:
                param.requires_grad = False
            else:
                print(f'\t{name}')
                param.requires_grad = True
        total_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)  
        print(f'Trainable parameters number: {total_params}\n')

    
    # Train the model
    print('Start Training model ...\n')
    current_step = 0
    for epoch in trange(num_epochs, desc="Epochs"):
        # Train the model for one epoch
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            current_step += 1
            inputs, labels, embedding = data[0], data[1].to(device), data[2].to(device)
            for key in inputs:
                inputs[key] = inputs[key].to(device)
                
            optimizer.zero_grad()
            outputs = model((inputs['input_ids'], inputs['attention_mask']), embedding)
            
            labels = labels.long()
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if log:
                wandb.log({"Train loss": loss.item()})

            if i % print_every == print_every - 1:    # print every 100 mini-batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / print_every:.4f}')
                running_loss = 0.0
            
            if current_step % val_step == 0:
                print(' ')
                val_loss, val_accuracy = validate(model, val_loader, criterion, device)
                print('-'*100)
                print(f'Validation Step [{current_step}], Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
                print('-'*100)
                print(' ')
                model.train()
        
        if scheduler:
            scheduler.step()
    print('\nFinish Training model!\n')
    
    # Save the trained model
    print('Save model ...')
    torch.save(model.state_dict(), f'{model_save_path}/{model_save_name}.pt')
    print(f'Model saved, path: {model_save_path}/{model_save_name}.pt')


if __name__ == "__main__":
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", type=str, help="Path to JSON configuration file")
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.config_path)