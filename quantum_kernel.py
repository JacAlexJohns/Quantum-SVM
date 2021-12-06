import itertools
import numpy as np
from cirq import S, X, H, SWAP, ry
from cirq import Moment, Circuit, LineQubit, Simulator, measure
from data_preprocess import load_and_process_data

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
        print()

    s = Simulator()
    results = s.simulate(c)
    state = results.final_state_vector.real
    state_outer_norm = np.outer(state, state)
    half = state_outer_norm.shape[0] // 2
    Ktop = np.array([[np.trace(state_outer_norm[:half, :half]), np.trace(state_outer_norm[:half, half:])],
                    [np.trace(state_outer_norm[half:, :half]), np.trace(state_outer_norm[half:, half:])]])
    if verbose >= 2:
        print()
        print('Final State Vector:', state)
        print('Normalized Outer Product:', state_outer_norm)
        print('K-top [K/tr(K)]:', Ktop)
        print()
    
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

    # # Set M Measurements
    # for i in range(M):
    #     c.append(measure(qubits[i]))

    if verbose >= 1:
        print()
        print(c)
        print()

    s = Simulator()
    results = s.simulate(c)
    state = results.final_state_vector.real
    state_outer_norm = (1. / M) * np.outer(state, state)
    half = state_outer_norm.shape[0] // 2
    Ktop = np.array([[np.trace(state_outer_norm[:half, :half]), np.trace(state_outer_norm[:half, half:])],
                    [np.trace(state_outer_norm[half:, :half]), np.trace(state_outer_norm[half:, half:])]])
    if verbose >= 2:
        print()
        print('Final State Vector:', state)
        print('Normalized Outer Product:', state_outer_norm)
        print('K-top [K/tr(K)]:', Ktop)
        print()
    
    K = Ktop * 2
    if verbose >= 1:
        print()
        print('K:', K)
        print()

    return K

if __name__ == '__main__':
    # Number of (training,test) points
    M, N = 8, 0

    # MNIST numbers (2 for binary)
    first, second = 6, 9

    # Get Train and Test Data
    print('Loading Data')
    thetas_train, _, _, _ = load_and_process_data(first, second, M, N)

    print('K Original')
    K_original = build_kernel_original(M, thetas_train)
    print('K Simplified')
    K_simplified = build_kernel_simplified(M, thetas_train)

    print('K Original:', K_original)
    print('K Simplified:', K_simplified)