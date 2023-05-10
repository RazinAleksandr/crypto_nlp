from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import argparse

import bokeh.models as bm
import bokeh.plotting as pl
from bokeh.io import output_notebook

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import fasttext.util


class word_embeddings:
    def __init__(self, model, df, text_column, batch_size=5):
        self.model = model
        self.df = df
        self.text_column = text_column
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(len(df) / batch_size))

    def batch_generator(self):
        while True:
            for i in range(self.num_batches):
                batch = self.df[i*self.batch_size : (i+1)*self.batch_size]
                X_batch = batch[self.text_column].values
                yield X_batch
    
    def predict(self, X):
        return [self.model.get_sentence_vector(x) for x in X]


    def draw_vectors(
        self, x, y, radius=10, alpha=0.25, color="blue", width=600, height=400, show=True, **kwargs
        ):
        """draws an interactive plot for data points with auxilirary info on hover"""
        
        output_notebook()
        if isinstance(color, str):
            color = [color] * len(x)
        data_source = bm.ColumnDataSource({"x": x, "y": y, "color": color, **kwargs})

        fig = pl.figure(active_scroll="wheel_zoom", width=width, height=height)
        fig.scatter("x", "y", size=radius, color="color", alpha=alpha, source=data_source)

        fig.add_tools(bm.HoverTool(tooltips=[(key, "@" + key) for key in kwargs.keys()]))
        if show:
            pl.show(fig)
        return fig
    

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

    ft = fasttext.load_model('cc.en.300.bin')
    df = pd.read_csv(texts_data_path, index_col=0)

    to_vec = word_embeddings(ft, df, 'title')
    num_batches = to_vec.num_batches

    data_generator = to_vec.batch_generator()

    X = []
    for i in tqdm(range(num_batches)):
        X_batch = next(data_generator)
        X.extend(to_vec.predict(X_batch))
    X = np.asarray(X)

    np.save(embeddings_data_path, X)

    """
    pca = PCA(2)
    scaler = StandardScaler()

    phrase_vectors_2d = pca.fit_transform(X)
    phrase_vectors_2d = scaler.fit_transform(phrase_vectors_2d)

    to_vec.draw_vectors(
        phrase_vectors_2d[:, 0],
        phrase_vectors_2d[:, 1],
        phrase=[phrase[:50] for phrase in df.title.to_list()],
        radius=20,
    )"""


if __name__ == "__main__":
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", type=str, help="Path to JSON configuration file")
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.config_path)