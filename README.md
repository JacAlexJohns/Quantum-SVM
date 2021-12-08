# Quantum-SVM

This repository contains the files to implemenet a Quantum Support Vector Machine utilizing the [Python Cirq library](https://quantumai.google/cirq), from Google Qauntum AI.
The repository is divided into 5 files, which handle different aspects of the QSVM mechanism. The files and their specific purposes are detailed below. 
This repo provides two examples of the use of the QSVM, one utilizing the [Keras MNIST](https://keras.io/api/datasets/mnist/) dataset and the other utilizing the [Keras Boston Housing](https://keras.io/api/datasets/boston_housing/) dataset. 
There is also an implementation of the classical SVM (specifically SVC) from Scikit-Learn which is used for comparison on the given datasets.
This implementation of the QSVM is purely for binary classification. Multiclass classification could be accomplished by modifying the execution for one-vs-all classification, but that has not been setup at this time.
The circuits for the QSVM and for the Quantum Kernels come from the following paper: [Support Vector Machines on Noisy Intermediate-Scale Quantum Computers](http://www.diva-portal.org/smash/get/diva2:1381355/FULLTEXT01.pdf)

# Files

**- gates.py**: This file contains all of the additional gates necessary for the QSVM that were not already included in the Cirq library. Each gate is extended from the Cirq Gate object.

**- data_preprocess.py**: This file contains all of the methods for loading and preprocessing the two datasets, MNIST and Boston Housing. This file will load the data, split it into X and Y / training and test sets, then preprocess the X values so that they are converted to single theta values per datapoint for the QSVM algorithm or 2-pair x0,x1 normalized datapoints for the SVM algorithm. The methods within are utilized by the other files. Running the file itself will plot the train and test thetas generated for the selected dataset.

**- quantum_kernel.py**: This file contains two quantum circuits for calculating the kernel matrix for use by the QSVM circuit. The first implementation makes use of more gates, but fewer qubits than the second making it potentially less practical on a real system. However, the resulting kernel matrix from the first implementation is far better than that of the second, and therefore the second is not utilized in any of the other files. The methods within are utilized by the other files. Running the file itself will calculate the two different kernels on the selected dataset and print them.

**- qsvm.py**: This file contains the circuit for the Quantum Support Vector Machine and the code to run the QSVM on the two datasets to calculate a test accuracy score.

**- svm.py**: This file contains the classical implementation of the Support Vector Classifier from the [scikit-learn library](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html). It also runs the SVC on the two datasets to calculate a test accuracy score. 

To run any of the files, minus the gates.py file which solely contains helper classes, you can simply run the file normally (`python X.py`) which will run the file on the MNIST dataset. If you wish to run on the Boston Housing dataset you can simply add the following flag to the command: `-d housing`. The `-d` flag allows you to select the datasets with the following two options: [mnist, housing].
