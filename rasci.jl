
using Pkg
Pkg.add("PyCall")
Pkg.add("Arpack")
Pkg.add("NPZ")
Pkg.add("JLD2")
using PyCall
using QCBase
using ActiveSpaceSolvers
using InCoreIntegrals
using LinearAlgebra
using Printf
using Test
using Arpack
using NPZ
using JLD2

h0 = npzread("h0.npy")
h1 = npzread("h1.npy")
h2 = npzread("h2.npy")

#40 occ, 40 vir 20,2

norbs = 102
n_elec_a = 60
n_elec_b = 60
norbs_ras1 = 60
norbs_ras2 = 0
norbs_ras3 = 42
n_holes = 1
n_particles = 1

# get h0, h1, h2 from pyscf or elsewhere and create ints
ints = InCoreInts(h0, h1, h2)	

# to use RASCI, we need to define the number of orbitals, electrons, number of orbitals in each RAS subspace (ras1, ras2, ras3), maximum number of holes allowed in ras1, and maximum number of particle excitations allowed in ras3
ansatz = RASCIAnsatz(norbs, n_elec_a, n_elec_b, (norbs_ras1, norbs_ras2, norbs_ras3), max_h=n_holes, max_p=n_particles)

# We define some solver settings - default uses Arpack.jl
solver = SolverSettings(nroots=3, tol=1e-6, maxiter=100)

# we can now solve our Ansatz and get energies and vectors from solution
solution = solve(ints, ansatz, solver)
 
display(solution)

e = solution.energies
v = solution.vectors

# This solution can then be used to compute the 1RDM
rdm1a, rdm1b = compute_1rdm(solution, root=2)
