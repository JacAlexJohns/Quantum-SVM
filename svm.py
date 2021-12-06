import argparse
from sklearn.svm import SVC
from data_preprocess import load_and_process_data_mnist, load_and_process_data_housing

if __name__ == '__main__':
    # Create an argument for the script to select the dataset
    # By default the MNIST dataset will be utilized
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', '--dataset', help='Choose one of the datasets: mnist or housing', choices=['mnist', 'housing'], default='mnist')
    args = parser.parse_args()

    if args.dataset == 'mnist':
        # MNIST numbers (2 for binary)
        first, second = 6, 9

        # Number of (train, test) points for svc classifier and load the data
        M, N = 128, 100
        _, x_train, y_train, _, x_test, y_test, unique_map = load_and_process_data_mnist(first, second, M, N)

    else:
        # Feature indices for housing data
        first, second = 5, 12

        # Number of (train, test) points for qsvm circuit and load data
        M, N = 128, 100
        _, x_train, y_train, _, x_test, y_test, unique_map = load_and_process_data_housing(first, second, M, N)

    # Build the classifier object
    classifier = SVC()

    # Fit (train) the classifier on the training data
    classifier.fit(x_train, y_train)

    # Get the predictions from the test data
    predictions = classifier.predict(x_test)

    # Count the number of correct predictions
    num_correct = 0
    for i in range(len(y_test)):
        if predictions[i] == y_test[i]:
            num_correct += 1

    # Print the Test Accuracy
    print(f'Data Map: {unique_map}')
    print(f'Accuracy: {100 * num_correct / len(y_test):.2f}%')