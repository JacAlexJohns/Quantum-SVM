import itertools
import numpy as np
from cirq import S, X, H, SWAP, ry
from cirq import Moment, Circuit, LineQubit, Simulator, measure
from data_preprocess import load_and_process_data

# Number of (training,test) points
M, N = 128, 100

# MNIST numbers (2 for binary)
first, second = 6, 9

# Get Train and Test Data
thetas_train, y_train, thetas_test, y_test = load_and_process_data(first, second, M, N)

# thetas_train = [np.arctan(1 / (0.987 / 0.195)), np.arctan(1 / (0.345 / 0.935))]

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

    # # Setup Measurements
    # measurements = [measure(qubits[i]) for i in range(init_qubits)]
    # c.append(Moment(measurements))

    if verbose >= 1:
        print()
        print(c)

    s = Simulator()
    results = s.simulate(c)
    state = results.final_state_vector.real
    state_outer_norm = np.outer(state, state)
    half = state_outer_norm.shape[0] // 2
    Ktop = np.array([[np.trace(state_outer_norm[:half, :half]), np.trace(state_outer_norm[:half, half:])],
                    [np.trace(state_outer_norm[half:, :half]), np.trace(state_outer_norm[half:, half:])]])
    if verbose >= 2:
        print('Final State Vector:', state)
        print('Normalized Outer Product:', state_outer_norm)
        print('K-top [K/tr(K)]:', Ktop)
    
    K = Ktop * 2
    if verbose >= 1:
        print('K:', K)

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

    # # Set M Measurements
    # for i in range(M):
    #     c.append(measure(qubits[i]))