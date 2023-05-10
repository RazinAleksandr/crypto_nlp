from transformers import BertTokenizer, BertModel

import torch
from torch.utils.data import DataLoader

import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import classification_report

import sys
sys.path.insert(1, '../../models')
sys.path.insert(1, '../data')
from sentiment_net import TextClassifier
from dataset import MyDataset, to_tens


def load_model(model_path, model_name, num_classes, device):
    # Load the model from the specified path
    model = TextClassifier(BertModel, model_name, num_classes).to(device)
    
    if device == 'cuda':
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    return model

def load_data(input_path, model_name, batch_size):
    # Load the input data from the specified path
    texts = np.load(f'{input_path}/texts.npy', allow_pickle=True)
    labels = np.load(f'{input_path}/labels.npy', allow_pickle=True)
    embeddings = np.load(f'{input_path}/embeddings.npy', allow_pickle=True)

    dataset = MyDataset(texts, labels, embeddings, model_name, BertTokenizer, to_tens())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return dataset, data_loader

def make_predictions(model, data_loader, dataset, device):
    # Make predictions using the model
    df = {
    'text': [],
    'predict': [],
    'target': [],
    }
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            inputs, labels, embedding = data[0], data[1].to(device), data[2].to(device)
            for key in inputs:
                inputs[key] = inputs[key].to(device)
                    
            outputs = model((inputs['input_ids'], inputs['attention_mask']), embedding)
            y_pred_classes = torch.argmax(outputs.cpu().data, dim=1).numpy()

            all_preds.extend(y_pred_classes)
            all_targets.extend(labels.cpu().data.numpy())
            
            # store data
            for inp in inputs['input_ids']:
                decoded = dataset.tokenizer.decode(token_ids=inp, skip_special_tokens=True)
                df['text'].append(decoded)
            df['predict'].extend(y_pred_classes.tolist())
            df['target'].extend(list(map(int, labels.tolist())))
    
    df = pd.DataFrame(df)
    mapping = {
        0: 'negative',
        1: 'positive',
        2: 'neutral'
    }
    df['target'] = df['target'].replace(mapping)
    df['predict'] = df['predict'].replace(mapping)
    
    return all_preds, all_targets, df

def report(all_targets, all_preds, target_names=['negative', 'positive', 'neutral']):
    all_targets = list(map(int, all_targets))
    print(classification_report(all_targets, all_preds, target_names=target_names))

def save_predictions(predictions, output_path):
    # Save the predictions to the specified output path
    predictions.to_csv(f'{output_path}/prediction.csv')


def main(config_file):
    # Load the JSON configuration file
    with open(config_file, "r") as f:
        config = json.load(f)

    # Extract values from the configuration file
    model_name = config["model"]["name"]
    num_classes = config["model"]["num_classes"]

    input_path = config["predict"]["input_path"]
    output_path = config["predict"]["output_path"]
    model_path = config["predict"]["model_path"]
    batch_size = config["predict"]["batch_size"]

    device = torch.device(config["predict"]["device"])

    # Print configuration
    print("Parsed Config Parameters:")
    print("-------------------------")
    print(f"Test Data Path: {input_path}")
    print(f"Output Data Path: {output_path}")
    print(f"Model Path: {model_path}")
    print(f"Model Name: {model_name}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print("-------------------------\n")


    # Load the model from the specified path
    print('Load model ...')
    model = load_model(model_path, model_name, num_classes, device)
    print('Model loaded!')

    # Load the input data from the specified path
    print('\nLoad dataloader ...')
    dataset, data_loader = load_data(input_path, model_name, batch_size)
    print('Dataloader is ready!')
    print(f'Amount of data for the test: {len(data_loader) * batch_size}')

    # Make predictions using the model
    print('\nStart making predictions ...')
    all_preds, all_targets, df = make_predictions(model, data_loader, dataset, device)
    print('Predicted!\n')

    # Classification_report
    print('\nClassification Report:')
    report(all_targets, all_preds)
    
    # Save the predictions to the specified output path
    print('Save prediction ...')
    save_predictions(df, output_path)
    print('Saved!')


if __name__ == "__main__":
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", type=str, help="Path to JSON configuration file")
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.config_path)
