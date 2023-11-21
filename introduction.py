import matplotlib.pyplot as plt
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import SLSQP
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms import VQE

num_qubits = 2
ansatz = TwoLocal(num_qubits, "ry", "cz")
ansatz.decompose().draw("mpl", style="iqx")
plt.show()
estimator = Estimator()
optimizer = SLSQP(maxiter=1000)
vqe = VQE(estimator, ansatz, optimizer)
H2_op = SparsePauliOp.from_list(
    [
        ("II", -1.052373245772859),
        ("IZ", 0.39793742484318045),
        ("ZI", -0.39793742484318045),
        ("ZZ", -0.01128010425623538),
        ("XX", 0.18093119978423156),
    ]
)
result = vqe.compute_minimum_eigenvalue(H2_op)
print(result)

# replace the estimator and optimizer
estimator = Estimator(options={"shots": 1000})
vqe.estimator = estimator
vqe.optimizer = SPSA(maxiter=100)
result = vqe.compute_minimum_eigenvalue(operator=H2_op)
print(result)
