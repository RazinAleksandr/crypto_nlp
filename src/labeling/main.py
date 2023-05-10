import json
from utils_ import SpeakEasy, batch_generator, extract_classes
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import argparse


def main(config_file: str) -> None:
    ################ parse config ####################
    with open(config_file, "r") as f:
        config = json.load(f)

    model_assistant = config['models']['gpt']
    
    user = config['user_v2']
    assistant = config['assistant']

    key_path = config['paths']['key']
    df_path = config['paths']['data']
    ###################################################

    # models init
    bot = SpeakEasy(key=key_path, message_history=[user, assistant], model_id=model_assistant)

    df = pd.read_csv(df_path)
    #df = df
    batch_size = 10
    num_batches = int(np.ceil(len(df) / batch_size))
    data_generator = batch_generator(df, 'Name', batch_size)

    labeled_df = pd.DataFrame({
        'title': [],
        'label': []
        })
    labeled_df.to_csv(df_path)

    for _ in tqdm(range(num_batches)):
        X_batch = next(data_generator)

        while True:
            try:
                prediction = bot.run_answer(X_batch)['sentiment']
                labels = extract_classes(prediction[0])
                #labels = eval(prediction[0])
                labeled_df_batch = pd.DataFrame({
                    'title': list(X_batch),
                    'label': labels
                    })

                with open(df_path, 'a') as f:
                    labeled_df_batch.to_csv(f, header=False)
                break
            except Exception as e:
                print(e)
                print(labels)
                print(len(labels))
                print(len(X_batch))
                print(prediction)
                time.sleep(25)
                pass

        time.sleep(25)

        
if __name__ == "__main__":
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", type=str, help="Path to JSON configuration file")
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.config_path)