import numpy as np
from cirq import S, X, H, SWAP, ry
from cirq import Moment, Circuit, LineQubit, Simulator, measure
from gates import eF, SDagger

# TEMPORARY HARD-CODING
r = 4
theta0 =  np.pi/8
theta1 =  np.pi/4
theta2 = -np.pi/4
F = np.array([[1, .5], [.5, 1]])

# Circuit
c = Circuit()

# Set 4 Qubits
q1 = LineQubit(1)
q2 = LineQubit(2)
q3 = LineQubit(3)
q4 = LineQubit(4)

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

# End Measurement
c.append(measure(q4))
print(c)

s = Simulator()
results = s.simulate(c)
print(results)