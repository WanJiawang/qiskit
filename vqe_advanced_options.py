import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP
from qiskit_algorithms import VQE
# from qiskit_algorithms.utils import algorithm_globals
import pylab
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms.gradients import FiniteDiffEstimatorGradient

# a qubit operator for VQE
H2_op = SparsePauliOp.from_list(
    [
        ("II", -1.052373245772859),
        ("IZ", 0.39793742484318045),
        ("ZI", -0.39793742484318045),
        ("ZZ", -0.01128010425623538),
        ("XX", 0.18093119978423156),
    ]
)
# instantiate the Estimator of choice for the evaluation of expectation values
estimator = Estimator()

# we will iterate over these different optimizers to compare a set of optimizers through the VQE callback
optimizers = [COBYLA(maxiter=80), L_BFGS_B(maxiter=60), SLSQP(maxiter=60)]
converge_counts = np.empty([len(optimizers)], dtype=object)
converge_vals = np.empty([len(optimizers)], dtype=object)

for i, optimizer in enumerate(optimizers):
    print("\rOptimizer: {} ".format(type(optimizers).__name__), end="")
    # algorithm_globals.random_seed = 50
    ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")

    counts = []
    values = []


    def store_intermediate_result(eval_count, parameter, mean, std):
        counts.append(eval_count)
        values.append(mean)


    vqe = VQE(estimator, ansatz, optimizer, callback=store_intermediate_result)
    result = vqe.compute_minimum_eigenvalue(operator=H2_op)
    converge_counts[i] = np.asarray(counts)
    converge_vals[i] = np.asarray(values)

print("\rOptimization complete ")

# plot the energy value at each objective function call each optimizer makes from the callback data you stored
pylab.rcParams["figure.figsize"] = (12, 8)
for i, optimizer in enumerate(optimizers):
    pylab.plot(converge_counts[i], converge_vals[i], label=type(optimizer).__name__)
pylab.xlabel("Eval count")
pylab.ylabel("Energy")
pylab.title("Energy convergence for various optimizers")
pylab.legend(loc="upper right")
plt.show()

# compute a reference value for the solution
numpy_solver = NumPyMinimumEigensolver()
result = numpy_solver.compute_minimum_eigenvalue(operator=H2_op)
ref_value = result.eigenvalue.real
print(f"Reference value: {ref_value:.5f}")

# plot the difference between the VQE solution and this exact reference value as the algorithm converges towards the
# minimum energy
pylab.rcParams["figure.figsize"] = (12, 8)
for i, optimizer in enumerate(optimizers):
    pylab.plot(
        converge_counts[i],
        abs(ref_value - converge_vals[i]),
        label=type(optimizer).__name__,
    )
pylab.xlabel("Eval count")
pylab.ylabel("Energy difference from solution reference value")
pylab.title("Energy convergence for various optimizers")
pylab.yscale("log")
pylab.legend(loc="upper right")
plt.show()

# initialize both the corresponding primitive and primitive gradient
estimator = Estimator()
gradient = FiniteDiffEstimatorGradient(estimator, epsilon=0.01)

# inspect an SLSQP run using the FiniteDiffEstimatorGradient from above
ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")

optimizer = SLSQP(maxiter=100)

counts = []
values = []


def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)


vqe = VQE(estimator, ansatz, optimizer, callback=store_intermediate_result, gradient=gradient)

result = vqe.compute_minimum_eigenvalue(operator=H2_op)
print(f"Value using Gradient: {result.eigenvalue.real:.5f}")

pylab.rcParams["figure.figsize"] = (12, 8)
pylab.plot(counts, values, label=type(optimizer).__name__)
pylab.xlabel("Eval count")
pylab.ylabel("Energy")
pylab.title("Energy convergence using Gradient")
pylab.legend(loc="upper right")
plt.show()

# look at the results from our previous VQE run
print(result)
cost_function_evals = result.cost_function_evals

# take the optimal_point from the above result and use it as the initial_point for a follow-up computation
initial_pt = result.optimal_point

estimator1 = Estimator()
gradient1 = FiniteDiffEstimatorGradient(estimator, epsilon=0.01)
ansatz1 = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
optimizer1 = SLSQP(maxiter=1000)

vqe1 = VQE(estimator1, ansatz1, optimizer1, gradient=gradient1, initial_point=initial_pt)
result1 = vqe1.compute_minimum_eigenvalue(operator=H2_op)
print(result1)

cost_function_evals1 = result1.cost_function_evals
print()
print(
    f"cost_function_evals is {cost_function_evals1} with initial point versus {cost_function_evals} without it."
)
