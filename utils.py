import numpy as np
import matplotlib.pyplot as plt
import idx2numpy

def plot_accuracy_curve(accuracy_history, val_accuracy_history):
    plt.plot(accuracy_history, label='train', color='r')
    plt.plot(val_accuracy_history, label='val', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_curve.png')
    plt.show()

def plot_learning_curve(loss_history):
    plt.plot(loss_history, label='Crossentropy', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()

def plot_sample(img, true_label, predict_label):
    plt.imshow(img)
    plt.title('True label: {} | Predict label: {}'.format(true_label, predict_label))
    plt.savefig('sample.png')
    plt.show()

def plot_histogram(layer_name, layer_weights):
    plt.hist(layer_weights)
    plt.title('Histogram of {}'.format(layer_name))
    plt.xlabel('Value')
    plt.ylabel('Number')
    plt.show()

def minmax_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def load_mnist():
    X_train = idx2numpy.convert_from_file('MNIST_data/train-images.idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('MNIST_data/train-labels.idx1-ubyte')
    X_test = idx2numpy.convert_from_file('MNIST_data/t10k-images.idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('MNIST_data/t10k-labels.idx1-ubyte')

    train_images = []                                                   # reshape train images so that the training set
    for i in range(X_train.shape[0]):                                   # is of shape (60000, 1, 28, 28)
        train_images.append(np.expand_dims(X_train[i], axis=0))
    train_images = np.array(train_images)

    test_images = []                                                    # reshape test images so that the test set
    for i in range(X_test.shape[0]):                                    # is of shape (10000, 1, 28, 28)
        test_images.append(np.expand_dims(X_test[i], axis=0))
    test_images = np.array(test_images)

    indices = np.random.permutation(train_images.shape[0])              # permute and split training data in
    training_idx, validation_idx = indices[:55000], indices[55000:]     # training and validation sets
    train_images, validation_images = train_images[training_idx, :], train_images[validation_idx, :]
    train_labels, validation_labels = train_labels[training_idx], train_labels[validation_idx]

    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'validation_images': validation_images,
        'validation_labels': validation_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }

if __name__ == '__main__':
    dataset = load_mnist()
    print(dataset['train_images'].shape)