import cirq
import numpy as np
from scipy.linalg import expm

class eF(cirq.Gate):
    def __init__(self, F, coefficient):
        super(eF, self)
        self.F = F
        self.coefficient = coefficient

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return expm(self.coefficient * 1j * self.F * 2 * np.pi)

    def _circuit_diagram_info_(self, args):
        return f'e^({self.coefficient}iFt0)'

class SDagger(cirq.Gate):
    def __init__(self):
        super(SDagger, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return np.array([[1,   0],
                         [0, -1j]])

    def _circuit_diagram_info_(self, args):
        return 'S✝'

class I(cirq.Gate):
    def __init__(self):
        super(I, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return np.array([[1, 0],
                         [0, 1]])

    def _circuit_diagram_info_(self, args):
        return 'I'
