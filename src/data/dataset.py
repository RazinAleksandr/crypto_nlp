import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertModel

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from typing import List, Optional, Callable, Any
import argparse
import json


class MyDataset(Dataset):
    def __init__(
            self,
            texts: List[str],
            labels: List[str],
            embeddings: np.ndarray,
            model_name: str,
            tokenizer: DistilBertTokenizer,
            transform: Optional[Callable] = None
        ) -> None:
        """
        A custom dataset class to handle the text data.

        Args:
            texts (List[str]): List of strings representing the input text data.
            labels (List[str]): List of strings representing the labels for the input data.
            embeddings (np.ndarray): NumPy array of shape (num_samples, embedding_size) representing the 
                precomputed embeddings for each input text.
            model_name (str): Name of the pre-trained model used for tokenization.
            tokenizer (PreTrainedTokenizer): A tokenizer instance to tokenize the input text.
            transform (Optional[Callable], optional): Optional transform function to apply on the labels and 
                embeddings. Defaults to None.
        """

        self.texts = texts
        self.labels = labels
        self.embeddings = embeddings
        self.tokenizer = tokenizer.from_pretrained(model_name)

        self.transform = transform
        self.label2id = {"negative": 0, "positive": 1, "neutral": 2}

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """

        return len(self.texts)

    def __getitem__(self, index):
        """
        Returns the tokenized input text, label and precomputed embedding for a given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            Tuple[Dict[str, Tensor], int, Tensor]: A tuple containing:
                - A dictionary of tensors containing the tokenized input text data.
                - The integer label for the sample.
                - The precomputed embedding for the input text.
        """

        text = self.texts[index]
        label = self.labels[index]
        embedding = self.embeddings[index]
        
        label = self.label2id[label]
        if self.transform:
            label = self.transform(label)
            embedding = self.transform(embedding)

        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')        
        return {
            'input_ids': encoding['input_ids'].squeeze(), 
            'attention_mask': encoding['attention_mask'].squeeze()}, label, embedding
    

class to_tens():
    def __call__(self, data: Any) -> torch.Tensor:
        """
        Converts the input `data` to a PyTorch Tensor with `dtype` set to `torch.float32`.

        Args:
            data: Input data that needs to be converted to a PyTorch Tensor.

        Returns:
            A PyTorch Tensor with `dtype` set to `torch.float32`.
        """

        return torch.tensor(data, dtype=torch.float32)

    def __repr__(self) -> str:
        """
        Returns the class name as string.

        Returns:
            The name of the class as string.
        """

        return self.__class__.__name__ + '()'


class DataPreparation:
    """
    A class for preparing data for BERT-based classification with classic ML pipeline.
    """

    
    def __init__(self, tokenizer, model, model_name):
        """
        Initialize the DataPreparation object.

        Parameters:
            tokenizer (class): Tokenizer class from the transformers library.
            model (class): BERT model class from the transformers library.
            model_name (str): Name of the pre-trained BERT model to use.
        """

        self.tokenizer = tokenizer.from_pretrained(model_name)
        self.model_pretrained = model.from_pretrained(model_name)

    def get_tokenized(
        self, 
        path_texts: str, 
        class_names: list = ['negative', 'positive', 'neutral']
        ) -> tuple:
        """
        Tokenizes the text data using the BERT tokenizer.

        Parameters:
            path_texts (str): Path to the CSV file containing the text data.
            class_names (list, optional): List of class names for the labels.
                Defaults to ['negative', 'positive', 'neutral'].

        Returns:
            tuple: A tuple of numpy arrays containing the tokenized features and labels.
        """

        # Load the text data as a pandas DataFrame
        df = pd.read_csv(path_texts, index_col=0)

        # Replace the class names with integer labels
        df['label'] = df['label'].replace({name: i for i, name in enumerate(class_names)})

        # Tokenize the text data using the BERT tokenizer
        tokenized_texts = df['title'].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True)).values
        max_len = max(len(text) for text in tokenized_texts)
        padded_texts = torch.tensor([text + [0] * (max_len - len(text)) for text in tokenized_texts])
        
        # Create an attention mask for the padded texts
        attention_mask = torch.where(padded_texts > 0, 1, 0)

        # Run the BERT model to get the hidden states
        output = self.run_bert_model(padded_texts, attention_mask)

        # Get the CLS hidden state as the features
        features = output.last_hidden_state[:, 0, :].numpy()

        # Get the labels
        labels = df['label'].values

        return features, labels

    def get_tokenized_with_embs(self, cls_features, path_embs: str) -> np.ndarray:
        """
        Concatenates the CLS features with pre-trained word embeddings.

        Parameters:
            cls_features (numpy.ndarray): Numpy array of features extracted from the CLS hidden states.
            path_embs (str): Path to the pre-trained word embeddings.

        Returns:
            numpy.ndarray: A numpy array of concatenated features.
        """
        
        # Load the pre-trained word embeddings
        embeddings = np.load(path_embs)

        # Concatenate the CLS features with the embeddings
        features = np.concatenate((cls_features, embeddings), axis=1)

        return features

    def run_bert_model(self, padded_texts, attention_mask):
        """
        Runs the BERT model on the padded texts and attention mask.

        Parameters:
            padded_texts (torch.Tensor): Tensor of padded texts.
            attention_mask (torch.Tensor): Tensor of attention masks.

        Returns:
            transformers.modeling_outputs.SequenceClassifierOutput: Output of the BERT model.
        """

        with torch.no_grad():
            output = self.model_pretrained(padded_texts, attention_mask)
        return output

    def save_df(self, features, labels, save_path):
        """
        Save final dataframe.

        Parameters:
            features (numpy.ndarray): Numpy array of features.
            attention_mask (numpy.ndarray): Numpy array of attention masks.

        Returns:
            pd.DataFrame: Final dataframe with features and labels.
        """

        full_data = pd.DataFrame(np.concatenate([features, np.reshape(labels, (labels.shape[0], 1))], axis=1))
        full_data.to_csv(save_path)
        print(full_data.shape)
        print(full_data)
        return full_data

    def plot_attention_mask(self, attention_mask):
        """
        Plot attention mask for texts.

        Parameters:
            attention_mask (numpy.ndarray): Numpy array of attention masks.
        """

        plt.pcolormesh(attention_mask)
        plt.colorbar()


def main(config_file: str) -> None:
    """
    The main function for data preprocessing.

    Args:
        config_file (str): Path to the JSON configuration file.
    """

    # Load the JSON configuration file
    with open(config_file, "r") as f:
        config = json.load(f)

    # Extract values from the configuration file
    texts_data_path = config["data_paths"]["texts"]
    embeddings_data_path = config["data_paths"]["embeddings"]
    full_df_data_path = config["data_paths"]["full_df"]

    # Initialize the data preparation object with DistilBertTokenizer and DistilBertModel
    data_preproc = DataPreparation(DistilBertTokenizer, DistilBertModel, 'distilbert-base-uncased')

    # Get the tokenized features and labels from the texts data file
    features, labels = data_preproc.get_tokenized(path_texts=texts_data_path)
    
    # Concatenate the tokenized features with the word embeddings
    features_with_embs = data_preproc.get_tokenized_with_embs(
        cls_features=features, 
        path_embs=embeddings_data_path
        )

    # Save the concatenated features and labels as a pandas dataframe in a CSV file
    data_preproc.save_df(
        features=features_with_embs, 
        labels=labels, 
        save_path=full_df_data_path
        )


if __name__ == "__main__":
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", type=str, help="Path to JSON configuration file")
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.config_path)