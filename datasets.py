import h5py
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(datapath = 'dataset.h5'):
    with h5py.File(datapath, 'r') as hf:
        X = hf['X'][:]
        y = hf['y'][:]
    
    train_x, test_, train_y, test_ = train_test_split(X, y, test_size=0.4, random_state=42)
    test_x, val_x, test_y, val_y = train_test_split(test_, test_, test_size=0.5, random_state=42)

    X_train = []
    for i in range(train_x.shape[0]):
        X_train.append(np.expand_dims(train_x[i], axis=0))
    X_train = np.array(X_train)
    X_test = []
    for i in range(test_x.shape[0]):
        X_test.append(np.expand_dims(test_x[i], axis=0))
    X_test = np.array(X_test)
    X_val = []
    for i in range(val_x.shape[0]):
        X_val.append(np.expand_dims(val_x[i], axis=0))
    X_val = np.array(X_val)

    return {
        'train_images': X_train,
        'train_labels': train_y,
        'validation_images': X_val,
        'validation_labels': val_y,
        'test_images': X_test,
        'test_labels': test_y
    }

if __name__ == '__main__':
    dataset = load_dataset()
    print(dataset['train_images'][0])