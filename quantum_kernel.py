import argparse
import itertools
import numpy as np
from cirq import S, X, H, SWAP, ry
from cirq import Moment, Circuit, LineQubit, Simulator, measure
from data_preprocess import load_and_process_data_mnist, load_and_process_data_housing

def build_kernel_original(M, thetas_train, verbose=0):
    # Circuit
    c = Circuit()

    # Set Log2(M)+1 Qubits
    qubits = []
    init_qubits = int(np.math.log2(M))
    num_qubits = init_qubits + 1
    for i in range(1, num_qubits+1):
        qubits.append(LineQubit(i))

    # Setup Initial Hadamards
    c.append(Moment([H(q) for q in qubits[:-1]]))

    # Setup Controlled Rotations
    control_map = list(map(list, itertools.product([X, 'I'], repeat=init_qubits)))
    for i in range(M):
        control = [control_map[i][j](qubits[j]) for j in range(init_qubits) if control_map[i][j] != 'I']
        c.append(Moment(control))
        c.append(ry(2*thetas_train[i]).on(qubits[-1]).controlled_by(*qubits[:-1]))
        c.append(Moment(control))

    # Print the circuit if desired
    if verbose >= 1:
        print()
        print(c)
        print()

    # Create the simulator
    s = Simulator()

    # Get the simulated results and final state vector
    results = s.simulate(c)
    state = results.final_state_vector.real

    # Get the outer product of the state vector with itself
    state_outer_norm = np.outer(state, state)

    # Set a value to half the length/width of the outer product and use it to find the partial trace TrB
    half = state_outer_norm.shape[0] // 2
    Ktop = np.array([[np.trace(state_outer_norm[:half, :half]), np.trace(state_outer_norm[:half, half:])],
                    [np.trace(state_outer_norm[half:, :half]), np.trace(state_outer_norm[half:, half:])]])

    # If desired print the intermediary matrix values
    if verbose >= 2:
        print()
        print('Final State Vector:', state)
        print('Normalized Outer Product:', state_outer_norm)
        print('K-top [K/tr(K)]:', Ktop)
        print()
    
    # Generate K from the Ktop value, which is the partial trace of the outer product of the state vector
    K = Ktop * 2
    if verbose >= 1:
        print()
        print('K:', K)
        print()

    return K

def build_kernel_simplified(M, thetas_train, verbose=0):
    # Circuit
    c = Circuit()
    
    # Set M Qubits
    qubits = []
    for i in range(1, M+1):
        qubits.append(LineQubit(i))

    # Set M Rotations w/ Each Training Theta
    for i in range(M):
        c.append(H(qubits[i]))
        c.append(ry(2*thetas_train[i]).on(qubits[i]))

    # Print the circuit if desired
    if verbose >= 1:
        print()
        print(c)
        print()

    # Create the simulator
    s = Simulator()

    # Get the simulated results and final state vector
    results = s.simulate(c)
    state = results.final_state_vector.real

    # Get the outer product of the state vector with itself normalized by M
    state_outer_norm = (1. / M) * np.outer(state, state)

    # Set a value to half the length/width of the outer product and use it to find the partial trace TrB
    half = state_outer_norm.shape[0] // 2
    Ktop = np.array([[np.trace(state_outer_norm[:half, :half]), np.trace(state_outer_norm[:half, half:])],
                    [np.trace(state_outer_norm[half:, :half]), np.trace(state_outer_norm[half:, half:])]])

    # If desired print the intermediary matrix values
    if verbose >= 2:
        print()
        print('Final State Vector:', state)
        print('Normalized Outer Product:', state_outer_norm)
        print('K-top [K/tr(K)]:', Ktop)
        print()
    
    # Generate K from the Ktop value, which is the partial trace of the outer product of the state vector
    K = Ktop * 2
    if verbose >= 1:
        print()
        print('K:', K)
        print()

    return K

if __name__ == '__main__':
    # Create an argument for the script to select the dataset
    # By default the MNIST dataset will be utilized
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', '--dataset', help='Choose one of the datasets: mnist or housing', choices=['mnist', 'housing'], default='mnist')
    args = parser.parse_args()

    print('Loading Data')
    if args.dataset == 'mnist':
        # MNIST numbers (2 for binary)
        first, second = 6, 9

        # Number of (training,test) points
        M, N = 8, 0

        # Get the training data
        thetas_train, _, _, _, _, _, _ = load_and_process_data_mnist(first, second, M, N)
    else:
        # Feature indices for housing data
        first, second = 5, 12

        # Number of (train, test) points
        M, N = 8, 0

        # Get the training data
        thetas_train, _, _, _, _, _, _ = load_and_process_data_housing(first, second, M, N)

    # Get the original and the simplified kernels
    print('K Original')
    K_original = build_kernel_original(M, thetas_train)
    print('K Simplified')
    K_simplified = build_kernel_simplified(M, thetas_train)

    # Show the original and the simplified kernels
    print('K Original:', K_original)
    print('K Simplified:', K_simplified)