import argparse
import numpy as np
import matplotlib.pyplot as plt 
from keras.datasets import mnist, boston_housing

def load_data_mnist(num1, num2, num_train, num_test):
    # Get the number of training and test samples for each class
    num_train_class = num_train // 2
    num_test_class = num_test // 2

    # Load the MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Get the x and y training data
    x_train = np.concatenate((x_train[y_train == num1][:num_train_class],
                              x_train[y_train == num2][:num_train_class])).astype(np.int32)
    y_train = np.concatenate((y_train[y_train == num1][:num_train_class],
                              y_train[y_train == num2][:num_train_class])).astype(np.int32)

    # Get the x and y test data
    x_test = np.concatenate((x_test[y_test == num1][:num_test_class],
                             x_test[y_test == num2][:num_test_class])).astype(np.int32)
    y_test = np.concatenate((y_test[y_test == num1][:num_test_class],
                             y_test[y_test == num2][:num_test_class])).astype(np.int32)

    # Create a map for each class to its interpreted value (actual class)
    unique_map = {1: num1, -1: num2}
    # Substitute 1/-1 for the train class values
    y_train[y_train == num1] = 1
    y_train[y_train == num2] = -1
    # Substitute 1/-1 for the test class values
    y_test[y_test == num1] = 1
    y_test[y_test == num2] = -1

    return (x_train, y_train), (x_test, y_test), unique_map

def load_data_housing(num_train, num_test):
    # The number to split the data on
    # The data is split for classification
    # Below 21 is class 1 and above is class -1
    split_number = 21

    # Get the number of training and test samples for each class
    num_train_class = num_train // 2
    num_test_class = num_test // 2

    # Load the housing data
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    # Get the x and y training data
    x_train = np.concatenate((x_train[y_train <  split_number][:num_train_class],
                              x_train[y_train >= split_number][:num_train_class]))
    y_train = np.concatenate((y_train[y_train <  split_number][:num_train_class],
                              y_train[y_train >= split_number][:num_train_class]))

    # Get the x and y test data
    x_test = np.concatenate((x_test[y_test <  split_number][:num_test_class],
                             x_test[y_test >= split_number][:num_test_class]))
    y_test = np.concatenate((y_test[y_test <  split_number][:num_test_class],
                             y_test[y_test >= split_number][:num_test_class]))

    # Create a map for each class to its interpreted value (actual class)
    unique_map = {1: f'<{split_number}', -1: f'>={split_number}'}
    # Substitute 1/-1 for the train class values
    y_train[y_train <  split_number] = 1
    y_train[y_train >= split_number] = -1
    # Substitute 1/-1 for the test class values
    y_test[y_test <  split_number] = 1
    y_test[y_test >= split_number] = -1

    return (x_train, y_train.astype(np.int32)), (x_test, y_test.astype(np.int32)), unique_map

def hr_vr(images, width, height):
    # Get half the width and height
    half_width = width // 2
    half_height = height // 2

    # Get the left, right, top, and bottom quandrants of the image
    left = images[:, :, :half_width]
    right = images[:, :, half_width:]
    top = images[:, :half_height, :]
    bottom = images[:, half_height:, :]

    # Get HR and VR from the above quadrants
    HR = np.sum(left, axis=(1, 2)) / np.sum(right, axis=(1, 2))
    VR = np.sum(top, axis=(1, 2)) / np.sum(bottom, axis=(1, 2))

    # Set the HR and VR values for each datapoint into an array
    hr_vr_matrix = np.zeros((len(images), 2))
    hr_vr_matrix[:, 0] = HR
    hr_vr_matrix[:, 1] = VR

    return hr_vr_matrix

def reduce_features(x, i1, i2):
    # Reduce the features to the given feature indices
    return x[:, [i1, i2]]

def linear_mapping(hrs_vrs_matrix, ms, bs):
    # Linearly map the hr and vr values
    hrs_vrs_matrix[:, 0] = hrs_vrs_matrix[:, 0] * ms[0] + bs[0]
    hrs_vrs_matrix[:, 1] = hrs_vrs_matrix[:, 1] * ms[1] + bs[1]
    return hrs_vrs_matrix

def l2_normalization(lin_map_matrix):
    # Find the L2 normalization term for each row
    normalization = np.sqrt(np.sum(lin_map_matrix ** 2, axis=1))[np.newaxis].T
    # Divide the points by the normalization term
    lin_map_matrix = lin_map_matrix / normalization
    return lin_map_matrix

def generate_angles(normalized_matrix):
    # Divide the x0 values by the x1 values
    x1_over_x2 = normalized_matrix[:, 0] / normalized_matrix[:, 1]

    # { (x0, x1) in first quandrant take arccos(x0/x1)
    # { (x0, x1) in second quandrant take arccos(x0/x1)
    # { (x0, x1) in third quandrant take arctan(x0/x1)
    # { (x0, x1) in fourth quandrant take arctan(x0/x1)
    thetas = np.zeros(x1_over_x2.shape)
    thetas[normalized_matrix[:, 1] >= 0] = np.arctan(1 / x1_over_x2[normalized_matrix[:, 1] >= 0])
    thetas[normalized_matrix[:, 1] < 0] = np.arctan(x1_over_x2[normalized_matrix[:, 1] < 0])

    return thetas

def plot_by_class_1d(train_1d, train_labels, ax, title):
    # Get the unique labels
    labels = np.unique(train_labels)
    # Plot the x (cos) and y (sin) values for each datapoint that maps to each label
    for label in labels:
        label_points = train_1d[train_labels == label]
        ax.scatter(np.cos(label_points), np.sin(label_points))
    # Make the plot square, within a circle of 1, set the axis lines, and add a title
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')

def preprocess_data_mnist(x):
    # Get the hr and vr data for each image
    hr_vr_matrix = hr_vr(x, x.shape[1], x.shape[2])
    # Linearly map the data to better align it for the kernels
    linear_mapped = linear_mapping(hr_vr_matrix, [1.3, .95], [-.55, -.4])
    # Normalize the data
    normalized = l2_normalization(linear_mapped)
    # Get the thetas from the normalized data
    thetas = generate_angles(normalized)

    return thetas, normalized

def preprocess_data_housing(x, index_1, index_2):
    # Reduce the x values down to 2 features
    x_reduced = reduce_features(x, index_1, index_2)
    # Linearly map the data to better align it for the kernels
    linear_mapped = linear_mapping(x_reduced, [.95, .75], [-1.2, -1.2])
    # Normalize the data
    normalized = l2_normalization(linear_mapped)
    # Get the thetas from the normalized data
    thetas = generate_angles(normalized)

    return thetas, normalized

def load_and_process_data_mnist(first_number, second_number, train_samples, test_samples):
    # Get the training and test data
    (x_train, y_train), (x_test, y_test), dictionary = load_data_mnist(first_number, second_number, train_samples, test_samples)

    # Preprocess the train and test data, both thetas and normalized
    thetas_train, normalized_train = preprocess_data_mnist(x_train)
    thetas_test, normalized_test = preprocess_data_mnist(x_test)

    return thetas_train, normalized_train, y_train, thetas_test, normalized_test, y_test, dictionary

def load_and_process_data_housing(index_1, index_2, train_samples, test_samples):
    # Get the training and test data
    (x_train, y_train), (x_test, y_test), dictionary = load_data_housing(train_samples, test_samples)

    # Preprocess the train and test data, both thetas and normalized
    thetas_train, normalized_train = preprocess_data_housing(x_train, index_1, index_2)
    thetas_test, normalized_test = preprocess_data_housing(x_test, index_1, index_2)

    return thetas_train, normalized_train, y_train, thetas_test, normalized_test, y_test, dictionary

if __name__ == '__main__':
    # Create an argument for the script to select the dataset
    # By default the MNIST dataset will be utilized
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', '--dataset', help='Choose one of the datasets: mnist or housing', choices=['mnist', 'housing'], default='mnist')
    args = parser.parse_args()

    # Get the data for the given dataset
    if args.dataset == 'mnist':
        thetas_train, _, y_train, thetas_test, _, y_test, dictionary = load_and_process_data_mnist(6, 9, 100, 100)
    else:
        thetas_train, _, y_train, thetas_test, _, y_test, dictionary = load_and_process_data_housing(5, 12, 100, 100)

    # Plot the training and test data based on the theta values
    fig, axs = plt.subplots(2, figsize=(7, 7))
    plot_by_class_1d(thetas_train, y_train, axs[0], 'Train Data (thetas)')
    plot_by_class_1d(thetas_test, y_test, axs[1], 'Test Data (thetas)')
    plt.show()