from model import Network
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datasets import load_dataset
from utils import *
import warnings
warnings.filterwarnings("ignore")

# dataset = load_dataset('dataset.h5')
dataset = load_mnist()
model = Network()
model.build_model(dataset_name='mnist')
model.train(dataset, num_epochs=100, learning_rate=0.0001, validate=True, regularization=0.01, plot_weights=False, verbose=True)
model.evaluate(dataset['test_images'], dataset['test_labels'], regularization=0.01, plot_correct=0, plot_missclassified=0, verbose=True)


