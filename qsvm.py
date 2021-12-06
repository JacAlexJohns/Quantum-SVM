import numpy as np
from cirq import S, X, H, SWAP, ry
from cirq import Moment, Circuit, LineQubit, Simulator, measure

from gates import eF, SDagger
from quantum_kernel import build_kernel_original
from data_preprocess import load_and_process_data

def qsvm_circuit(F, theta0, theta1, theta2, r=4, verbose=0):
    # Circuit
    c = Circuit()

    # Set 4 Qubits
    q1 = LineQubit(1)
    q2 = LineQubit(2)
    q3 = LineQubit(3)
    q4 = LineQubit(4)

    # Setup Non-Zero States
    c.append(Moment([X(q3)]))
    c.append(Moment([H(q3)]))

    # Phase Estimation with Matrix F Derived from K
    c.append(Moment([H(q1), H(q2)]))
    c.append(Moment([eF(F, 2).on(q3).controlled_by(q1)]))
    c.append(Moment([eF(F, 1).on(q3).controlled_by(q1)]))
    c.append(Moment([SWAP(q1, q2)]))
    c.append(Moment([H(q2)]))
    c.append(Moment([SDagger().on(q1).controlled_by(q2)]))
    c.append(Moment([H(q1)]))

    # Controlled Rotation
    c.append(Moment([X(q1)]))
    c.append(Moment([ry(2*np.pi/(2 ** r)).on(q4).controlled_by(q1)]))
    c.append(Moment([ry(np.pi/(2 ** r)).on(q4).controlled_by(q2)]))
    c.append(Moment([X(q1)]))

    # Inverse Phase Estimation
    c.append(Moment([H(q1)]))
    c.append(Moment([S(q1).controlled_by(q2)]))
    c.append(Moment([H(q2)]))
    c.append(Moment([SWAP(q1, q2)]))
    c.append(Moment([eF(F, -1).on(q3).controlled_by(q1)]))
    c.append(Moment([eF(F, -2).on(q3).controlled_by(q1)]))
    c.append(Moment([H(q1), H(q2)]))

    # Training Oracle
    c.append(Moment([X(q3)]))
    c.append(Moment([ry(2*theta1).on(q2).controlled_by(q3, q4)]))
    c.append(Moment([X(q3)]))
    c.append(Moment([ry(2*theta2).on(q2).controlled_by(q3, q4)]))

    # Test Oracle
    c.append(Moment([ry(-2*theta0).on(q2).controlled_by(q4)]))
    c.append(Moment([H(q3).controlled_by(q4)]))

    # # End Measurement
    # c.append(measure(q4))

    if verbose == 1:
        print()
        print(c)
        print()

    s = Simulator()
    results = s.simulate(c)
    state = results.final_state_vector.real
    O = np.zeros((16, 16)); O[1][0] = 1
    E = np.inner(np.matmul(state.conj().T, O), state)
    prediction = int(np.sign(E).real)
    if prediction == 0: prediction = 1

    return prediction

if __name__ == '__main__':
    # MNIST numbers (2 for binary)
    first, second = 6, 9

    # Number of training points for kernel generation, load data, and build kernel
    M = 128
    thetas_train, _, _, _ = load_and_process_data(first, second, M, 0)
    kernel = build_kernel_original(M, thetas_train)

    # Number of (train, test) points for qsvm circuit and load data
    M, N = 2, 100
    thetas_train, y_train, thetas_test, y_test = load_and_process_data(first, second, M, N)

    # Map Y values to +1, -1
    uniques = np.unique(y_train)
    unique_map = {-((i*2)-1):uniques[i] for i in range(len(uniques))}

    # Run QSVM Circuit for each Test Point
    num_correct = 0
    for i in range(N):
        prediction = qsvm_circuit(kernel, thetas_test[i], thetas_train[0], thetas_train[1])
        prediction_number = unique_map[prediction]
        if prediction_number == y_test[i]:
            num_correct += 1

    # Print the Test Accuracy
    print(f'Accuracy: {100 * num_correct / N:.2f}%')