import argparse
import numpy as np
import matplotlib.pyplot as plt 
from keras.datasets import mnist, boston_housing

def load_data_mnist(num1, num2, num_train, num_test):
    """Generate the MNIST training and test sets.

    This method loads the full MNIST dataset, selects out two of the numbers for use
    in the training and test sets, creates the training and test sets (both x and y),
    then scales down the train and test sets to num_train training points and
    num_test test points.

    Args:
        num1: The first MNIST number to use in the train/test sets.
        num2: The second MNIST number to use in the train/test sets.
        num_train: The number of training points to return.
        num_test: The number of test points to return.
    Returns:
        Train and test tuples (x, y), and the dictionary that maps 1/-1 to the actual values.
    """
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
    """Generate the Housing training and test sets.

    This method loads the full Housing dataset, selects a point to split the y data
    on for the training and test sets so as to convert from a regression to binary
    classification task, creates the training and test sets (both x and y), then scales 
    down the train and test sets to num_train training points and num_test test points.

    Args:
        num_train: The number of training points to return.
        num_test: The number of test points to return.
    Returns:
        Train and test tuples (x, y), and the dictionary that maps 1/-1 to the actual values.
    """
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
    """Creates a matrix of Horizontal Ratio and Verical Ratio values for each image.

    This method calculates the horizontal ratio as the sum of the pixels in the left
    half of the image divided by the sum of the pixels in the right half of the image,
    and the vertical ratio as the sum of the pixels in the top half of the image divided 
    by the sum of the pixels in the bottom half of the image. This is done for each 
    image in the dataset, and the matrix (Nx2) of the HR and VR values is returned.

    Args:
        images: The list of images in the dataset.
        width: The width of each image in the dataset.
        height: The height of each image in the dataset.
    Returns:
        An Nx2 matrix, where N is the number of images and 2 is the HR/VR pairs.
    """
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
    """Reduces the number of features in X down to 2.

    Args:
        x: The x values of the train or test dataset.
        i1: The index of the first feature to utilize.
        i2: The index of the second feature to utilize.
    Returns:
        An Nx2 matrix, where N is the length of the dataset and 2 is the two input features.
    """
    # Reduce the features to the given feature indices
    return x[:, [i1, i2]]

def linear_mapping(data_matrix, ms, bs):
    """Linearly maps (scales) the dataset.

    This method assumes that the dataset only has two features, and the ms and bs
    arrays also only have two values each. Any additional values will be ignored.
    The first index of each datapoint is multiplied by the first value in ms
    and the first value in bs is added. The second index of each datapoint is
    multiplied by the second value in ms and the second value of bs is added.
    This has the effect of making the theta values, which are calculated later,
    more spread out within the first quadrant and also ensures that there are
    not erronious theta values outside of the first quadrant.

    Args:
        data_matrix: The x values of the train or test dataset with only 2 features.
        ms: The m values for each of the two features. Should be a list of length 2.
        bs: The b values for each of the two features. Should be a list of length 2.
    Returns:
        The linearly mapped datapoints.
    """
    # Linearly map the hr and vr values
    data_matrix[:, 0] = data_matrix[:, 0] * ms[0] + bs[0]
    data_matrix[:, 1] = data_matrix[:, 1] * ms[1] + bs[1]
    return data_matrix

def l2_normalization(lin_map_matrix):
    """Applies L2 Normalization to the datapoints in the dataset.

    L2 normalization is defined as sqrt(sum(x0**2 + x1**2)), where x0 and x1 are the
    two x values in each datapoint. This is applied for each datapoint (row) in the dataset.

    Args:
        lin_map_matrix: The x values of the train or test dataset with only 2 features.
    Returns:
        The L2 normalized datapoints.
    """
    # Find the L2 normalization term for each row
    normalization = np.sqrt(np.sum(lin_map_matrix ** 2, axis=1))[np.newaxis].T
    # Divide the points by the normalization term
    lin_map_matrix = lin_map_matrix / normalization
    return lin_map_matrix

def generate_angles(normalized_matrix):
    """Generates the angles (thetas) for each datapoint in the dataset.

    This reduces each datapoint from two values down to one. The angle, theta, is
    derived from the value (x0 / x1). The value has either the arctan or arccot
    function applied, depending on its quadrant of the graph. For the first and
    second quandrant the arccot is taken. For the third and fourth quadrant the
    arctan is taken. Also, the arccot is not a numpy function, but the value is
    derived by taking the arctan(1/(x0 / x1)), which is equivalent.

    Args:
        normalized_matrix: The normalized x values from either the train or test datasets.
    Returns:
        The theta matrix for the dataset (Nx1); one theta for each datapoint.
    """
    # Divide the x0 values by the x1 values
    x1_over_x2 = normalized_matrix[:, 0] / normalized_matrix[:, 1]

    # { (x0, x1) in first quandrant take arccot(x0/x1)
    # { (x0, x1) in second quandrant take arccot(x0/x1)
    # { (x0, x1) in third quandrant take arctan(x0/x1)
    # { (x0, x1) in fourth quandrant take arctan(x0/x1)
    thetas = np.zeros(x1_over_x2.shape)
    thetas[normalized_matrix[:, 1] >= 0] = np.arctan(1 / x1_over_x2[normalized_matrix[:, 1] >= 0])
    thetas[normalized_matrix[:, 1] < 0] = np.arctan(x1_over_x2[normalized_matrix[:, 1] < 0])

    return thetas

def plot_by_class_1d(train_1d, train_labels, ax, title):
    """Plots the theta datapoints by class value.

    This method takes each datapoint (theta) and plots it as an (x, y) pair,
    where x = cos(theta) and y = sin(theta). The points are plotted by class,
    such that the first class is one color and the second class is another.

    Args:
        train_1d: The theta values for the dataset.
        train_labels: The class labels for each datapoint in the dataset.
        ax: The matplotlib ax plot object to plot to.
        title: The title of the plot.
    """
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
    """Run the full preprocessing pipeline for the MNIST dataset x values.

    This method first gets the HR,VR matrix from the X values, then linearly maps
    the data, then normalizes the values, before finally getting the theta values.
    These theta values along with the normalized values (before generating theta)
    are then returned. The theta values are used during the QSVM methods, while
    the normalized values are used during the classical SVM.

    Args:
        x: The x values matrix.
    Returns:
        The theta matrix for the dataset (Nx1); one theta for each datapoint.
        The L2 normalized datapoints matrix.
    """
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
    """Run the full preprocessing pipeline for the Housing dataset x values.

    This method first gets the reduced matrix from the X values, then linearly maps
    the data, then normalizes the values, before finally getting the theta values.
    These theta values along with the normalized values (before generating theta)
    are then returned. The theta values are used during the QSVM methods, while
    the normalized values are used during the classical SVM.

    Args:
        x: The x values matrix.
        index_1: The index of the first feature to utilize.
        index_2: The index of the second feature to utilize.
    Returns:
        The theta matrix for the dataset (Nx1); one theta for each datapoint.
        The L2 normalized datapoints matrix.
    """
    # Reduce the x values down to 2 features
    x_reduced = reduce_features(x, index_1, index_2)
    # Linearly map the data to better align it for the kernels
    linear_mapped = linear_mapping(x_reduced, [1.2, 1.8], [-.8, -1.2])
    # Normalize the data
    normalized = l2_normalization(linear_mapped)
    # Get the thetas from the normalized data
    thetas = generate_angles(normalized)

    return thetas, normalized

def load_and_process_data_mnist(first_number, second_number, train_samples, test_samples):
    """Load the MNIST data (x and y; train and test) and preprocess both the train and test sets.

    Args:
        first_number: The first MNIST number to use in the train/test sets.
        second_number: The second MNIST number to use in the train/test sets.
        train_samples: The number of training samples to return.
        test_samples: The number of test samples to return.
    Returns:
        The training and test thetas, training and test normalized X values,
        training and test Y values, and the dictionary mapping the (1/-1) values
        to the actual values they represent.
    """
    # Get the training and test data
    (x_train, y_train), (x_test, y_test), dictionary = load_data_mnist(first_number, second_number, train_samples, test_samples)

    # Preprocess the train and test data, both thetas and normalized
    thetas_train, normalized_train = preprocess_data_mnist(x_train)
    thetas_test, normalized_test = preprocess_data_mnist(x_test)

    return thetas_train, normalized_train, y_train, thetas_test, normalized_test, y_test, dictionary

def load_and_process_data_housing(index_1, index_2, train_samples, test_samples):
    """Load the Housing data (x and y; train and test) and preprocess both the train and test sets.

    Args:
        index_1: The index of the first feature to utilize.
        index_2: The index of the second feature to utilize.
        train_samples: The number of training samples to return.
        test_samples: The number of test samples to return.
    Returns:
        The training and test thetas, training and test normalized X values,
        training and test Y values, and the dictionary mapping the (1/-1) values
        to the actual values they represent.
    """
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
        thetas_train, _, y_train, thetas_test, _, y_test, dictionary = load_and_process_data_housing(10, 12, 100, 100)

    # Plot the training and test data based on the theta values
    fig, axs = plt.subplots(2, figsize=(7, 7))
    plot_by_class_1d(thetas_train, y_train, axs[0], 'Train Data (thetas)')
    plot_by_class_1d(thetas_test, y_test, axs[1], 'Test Data (thetas)')
    plt.show()