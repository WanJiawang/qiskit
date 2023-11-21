from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
import numpy as np
from qiskit.algorithms.optimizers import SLSQP
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.primitives import Estimator
from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint
from qiskit_nature.second_q.algorithms.initial_points import MP2InitialPoint
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

# obtain an ElectronicStructureProblem which we want to solve
drive = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basic="sto-3g")
problem = drive.run()

# setup our QubitMapper
mapper = JordanWignerMapper()

# setup our ansatz
ansatz = UCCSD(
    problem.num_spatial_orbitals,
    problem.num_particles,
    mapper,
    initial_state=HartreeFock(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
    ),
)

# setup a VQE
vqe = VQE(Estimator(), ansatz, SLSQP())

# choosing the initial point
vqe.initial_point = np.zeros(ansatz.num_parameters)

# one can also use HFInitialPoint like so
initial_point = HFInitialPoint()
initial_point.ansatz = ansatz
initial_point.problem = problem
vqe.initial_point = initial_point.to_numpy_array()

# one can also use MP2InitialPoint like so
initial_point = MP2InitialPoint()
initial_point.ansatz = ansatz
initial_point.problem = problem
vqe.initial_point = initial_point.to_numpy_array()

# solve our problem
solver = GroundStateEigensolver(mapper, vqe)
result = solver.solve(problem)

print(f"Total ground state energy = {result.total_energies[0]:.4f}")


