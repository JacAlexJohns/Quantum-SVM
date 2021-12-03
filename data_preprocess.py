import numpy as np
import matplotlib.pyplot as plt 
from keras.datasets import mnist

def load_data(num1, num2, num_train, num_test):
    num_train_class = num_train // 2
    num_test_class = num_test // 2

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.concatenate((x_train[y_train == num1][:num_train_class],
                              x_train[y_train == num2][:num_train_class]))
    y_train = np.concatenate((y_train[y_train == num1][:num_train_class],
                              y_train[y_train == num2][:num_train_class]))

    x_test = np.concatenate((x_test[y_test == num1][:num_test_class],
                             x_test[y_test == num2][:num_test_class]))
    y_test = np.concatenate((y_test[y_test == num1][:num_test_class],
                             y_test[y_test == num2][:num_test_class]))

    return (x_train, y_train), (x_test, y_test)

def hr_vr(images, width, height):
    half_width = width // 2
    half_height = height // 2

    left = images[:, :, :half_width]
    right = images[:, :, half_width:]
    top = images[:, :half_height, :]
    bottom = images[:, half_height:, :]

    HR = np.sum(left, axis=(1, 2)) / np.sum(right, axis=(1, 2))
    VR = np.sum(top, axis=(1, 2)) / np.sum(bottom, axis=(1, 2))

    hr_vr_matrix = np.zeros((len(images), 2))
    hr_vr_matrix[:, 0] = HR
    hr_vr_matrix[:, 1] = VR

    return hr_vr_matrix

def linear_mapping(hrs_vrs_matrix):
    hrs_vrs_matrix[:, 0] = hrs_vrs_matrix[:, 0] * 1.3 - .55
    hrs_vrs_matrix[:, 1] = hrs_vrs_matrix[:, 1] * .95 - .35
    return hrs_vrs_matrix

def l2_normalization(lin_map_matrix):
    normalization = np.sqrt(np.sum(lin_map_matrix ** 2, axis=1))[np.newaxis].T
    lin_map_matrix = lin_map_matrix / normalization
    return lin_map_matrix

def generate_angles(normalized_matrix):
    x1_over_x2 = normalized_matrix[:, 0] / normalized_matrix[:, 1]

    thetas = np.zeros(x1_over_x2.shape)
    thetas[normalized_matrix[:, 1] >= 0] = np.arctan(1 / x1_over_x2[normalized_matrix[:, 1] >= 0])
    thetas[normalized_matrix[:, 1] < 0] = np.arctan(x1_over_x2[normalized_matrix[:, 1] < 0])

    return thetas

def plot_by_class_2d(train_2d, train_labels, ax):
    labels = np.unique(train_labels)
    for label in labels:
        label_points = train_2d[train_labels == label]
        # diff = label_points[:, 1] / label_points[:, 0]
        diff = label_points[:, 0] / label_points[:, 1]
        angles = np.arctan(diff)
        ax.scatter(np.cos(angles), np.sin(angles))
    ax.set_xlim(-.2, 1.2)
    ax.set_ylim(-.2, 1.2)
    ax.set_aspect('equal', adjustable='box')

def plot_by_class_1d(train_1d, train_labels, ax):
    labels = np.unique(train_labels)
    for label in labels:
        label_points = train_1d[train_labels == label]
        ax.scatter(np.cos(label_points), np.sin(label_points))
    ax.set_xlim(-.2, 1.2)
    ax.set_ylim(-.2, 1.2)
    ax.set_aspect('equal', adjustable='box')

def preprocess_data(x, y):
    _, height, width = x.shape

    hr_vr_matrix = hr_vr(x, height, width)
    linear_mapped = linear_mapping(hr_vr_matrix)
    normalized = l2_normalization(linear_mapped)
    thetas = generate_angles(normalized)

    return thetas

def load_and_process_data(first_number, second_number, train_samples, test_samples):
    (x_train, y_train), (x_test, y_test) = load_data(first_number, second_number, train_samples, test_samples)

    thetas_train = preprocess_data(x_train, y_train)
    thetas_test = preprocess_data(x_test, y_test)

    return thetas_train, y_train, thetas_test, y_test

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data(6, 9, 100, 100)

    thetas_train = preprocess_data(x_train, y_train)
    thetas_test = preprocess_data(x_test, y_test)

    fig, axs = plt.subplots(2)
    plot_by_class_1d(thetas_train, y_train, axs[0])
    plot_by_class_1d(thetas_test, y_test, axs[1])
    plt.show()