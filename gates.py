import cirq
import numpy as np
from scipy.linalg import expm

class eF(cirq.Gate):
    """The eF gate to be used in the HHL portion of the QSVM algorithm.

    Args:
        F (2x2 Matrix): The F matrix generated from the kernel.
        coefficient (number): The coefficient to apply in the exponent.

    Attributes:
        F (2x2 Matrix): The F matrix generated from the kernel.
        coefficient (number): The coefficient to apply in the exponent.
    """
    def __init__(self, F, coefficient):
        """Invoke the super function to initialize the object based on the parent (Gate) and set the attributes."""
        super(eF, self)
        self.F = F
        self.coefficient = coefficient

    def _num_qubits_(self):
        """Single qubit gate."""
        return 1

    def _unitary_(self):
        """Unitary matrix for e^(CiFt0) (2x2)."""
        return expm(self.coefficient * 1j * self.F * 2 * np.pi)

    def _circuit_diagram_info_(self, args):
        """Diagram symbol: e^(CiFt0)"""
        return f'e^({self.coefficient}iFt0)'

class SDagger(cirq.Gate):
    """The S conjugate gate."""
    def __init__(self):
        """Invoke the super function to initialize the object based on the parent (Gate)."""
        super(SDagger, self)

    def _num_qubits_(self):
        """Single qubit gate."""
        return 1

    def _unitary_(self):
        """Unitary matrix for S conjugate (2x2)."""
        return np.array([[1,   0],
                         [0, -1j]])

    def _circuit_diagram_info_(self, args):
        """Diagram symbol: S✝"""
        return 'S✝'

class I(cirq.Gate):
    """The identity (I) gate."""
    def __init__(self):
        """Invoke the super function to initialize the object based on the parent (Gate)."""
        super(I, self)

    def _num_qubits_(self):
        """Single qubit gate."""
        return 1

    def _unitary_(self):
        """Unitary identity matrix (2x2)."""
        return np.array([[1, 0],
                         [0, 1]])

    def _circuit_diagram_info_(self, args):
        """Diagram symbol: I"""
        return 'I'
