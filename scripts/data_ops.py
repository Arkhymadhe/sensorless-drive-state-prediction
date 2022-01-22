import os
import pandas as pd
import numpy as np
import zlib
import zipfile

from sklearn.model_selection import train_test_split

def load_dataset(path):
    ''' Load dataset into memory. '''
    
    data = np.loadtxt(path)
    data = pd.DataFrame(data)
    data = data.dropna(axis = 0)
    
    return data

def variables(data):
    ''' Separate regressors from regressand. '''
    
    return data.iloc[:, : -1], data.iloc[:, -1]


def read_compress(path_to_data, path_to_archive = 'compressed_data.zip'):
    """ Read data into memory; compress and delete raw file."""

    df = load_dataset(path_to_data)

    with zipfile.ZipFile(path_to_archive, 'w') as zf:
        zf.write(path_to_data, arcname = 'sensorless_dataset.csv', compress_type = zipfile.ZIP_DEFLATED)
        zf.close()

    os.remove(path_to_data)

    return df, path_to_archive


def decompress(path_to_archive, path_to_data = 'sensorless_dataset.csv'):
    """ Decompress archived dataset. """

    with zipfile.ZipFile(path_to_archive, 'r') as zf:
        try:
            return zf.read(path_to_data)
        except KeyError:
            print(f'File not found! Ensure archive at {path_to_archive} is available.')
    return


def manual_describe(data, path, save = False):
    ''' Return a DataFrame containing a description of missing values, unique values, and data types. '''
    
    desc = {'Name' : pd.Series(np.array(data.columns) + 1).apply(lambda x: f'Feature {x}'),
            'Null' : data.isnull().sum(),
            'Null (%)' : 100 * data.isnull().sum()/data.shape[0],
            'Unique' : data.nunique(),
            'Unique (%)' : 100 * data.nunique()/data.shape[0],
            'Dtypes' : data.dtypes}

    desc = pd.DataFrame(desc)
    
    if save:
        desc.to_csv(os.path.join(path, 'manual_description.csv'), index = False)
    
    return desc


def auto_description(data):
    ''' Return auto-generated decsription of dataset. '''
    
    return data.describe()


def split_data(X, y, split_size = 0.25, random_state = 42):
    ''' Split dataset into folds. '''
    
    X, X_, y, y_ = train_test_split(X, y, test_size = split_size, stratify = y, random_state = random_state)
    
    return X, X_, y, y_


