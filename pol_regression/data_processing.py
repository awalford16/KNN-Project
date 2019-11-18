import pandas as pd
import os

def load_data(file):
    print("Loading in CSV file...")
    return pd.read_csv(os.path.join('../data', file))

def data_norm(data):
    return ((data-data.min())/(data.max()-data.min()))

# Split data into train and test
def split_data(data):
    print('Splitting data...')
    # Normalise the data
    data = data_norm(data)

    # Shuffle data
    data = data.sample(frac=1)

    # Split data at 70%
    percent = int((data.shape[0]) * 0.7)
    train = data[:percent]
    test = data[percent:]

    # Return splits of data set
    return train['x'].values, train['y'].values, test['x'].values, test['y'].values

def standard_data(data):
    return (data-data.mean())/data.std()