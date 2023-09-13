import pyscf
from pyscf import gto, scf, tdscf, mcscf
import numpy as np
import scipy
import matplotlib.pyplot as plt

# Define a function to calculate the density matrix
def tda_density_matrix(td, state_id):
    cis_t1 = td.xy[state_id][0]
    dm_oo = -np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
    dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj())
    mf = td._scf
    dm = np.diag(mf.mo_occ)
    nocc = cis_t1.shape[0]
    dm[:nocc, :nocc] += dm_oo * 2
    dm[nocc:, nocc:] += dm_vv * 2
    mo = mf.mo_coeff
    dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
    return dm

# Define a function to get natural orbitals in active space
def get_natural_orbital_active_space(rdm, S, thresh=.01):
    Ssqrt = scipy.linalg.sqrtm((S+S.T)/2.0)
    Sinvsqrt = scipy.linalg.inv(Ssqrt)
    print(" Number of electrons found %12.8f" %np.trace(S@rdm))
    Dtot = Ssqrt.T @ rdm @ Ssqrt
    D_evals, D_evecs = np.linalg.eigh((Dtot+Dtot.T)/2.0)
    sorted_list = np.argsort(D_evals)[::-1]
    D_evals = D_evals[sorted_list]
    D_evecs = D_evecs[:,sorted_list]
    act_list = []
    doc_list = []
    for idx,n in enumerate(D_evals):
        print(" %4i = %12.8f" %(idx,n),end="")
        if n < 2.0 - thresh:
            if n > thresh:
                act_list.append(idx)
                print(" Active")
            else:
                print(" Virt")
        else:
            doc_list.append(idx)
            print(" DOcc")

    print(" Number of active orbitals: ", len(act_list))
    print(" Number of doc    orbitals: ", len(doc_list))
    D_evecs = Sinvsqrt @ D_evecs
    Cdoc = D_evecs[:, doc_list]
    Cact = D_evecs[:, act_list]
    return Cdoc, Cact 

def active_space_energies_td(mf, norbs, nelec, state_id):
    energies = []  # Store the energies for different active space sizes
    xpoints = []  # Store labels for the x-axis

    for i in range(2, norbs):
        for j in range(2, nelec, 2):
            ncas, nelecas = i, j
            if ncas <= nelec <= norbs:
                mycas = mcscf.CASCI(mf, ncas, nelecas)
                mycas.kernel()
                energies.append(mycas.e_tot)
                xpoints.append(f"({ncas}, {nelecas})")

    return energies, xpoints

# Create the molecule
mol = gto.Mole()
mol.atom = '''
O          0.00000        0.00000        0.11779
H          0.00000        0.75545       -0.47116
H          0.00000       -0.75545       -0.47116
'''
mol.basis = '6-31g'
mol.spin = 0
mol.build()

# Perform Restricted Hartree-Fock calculation
mf = scf.RHF(mol)
mf.verbose = 4
mf.get_init_guess(mol, key='minao')
mf.kernel()

norbs = len(mf.mo_coeff)
nelec = int(mol.nelectron / 2)

avg_rdm1 = mf.make_rdm1()

# Compute singlets and triplets
n_singlets = 1
n_triplets = 1

# CIS
# compute singlets
mytd = tdscf.TDA(mf)
mytd.singlet = True
mytd = mytd.run(nstates=n_singlets)
mytd.analyze()
for i in range(mytd.nroots):
    avg_rdm1 += tda_density_matrix(mytd, i)
print('TDA CIS singlet excited state total energy = ', mytd.e_tot)

mf = mytd._scf
e_s, xpoints_singlets = active_space_energies_td(mf, norbs, nelec, n_singlets)

# Calculate the error for singlets
errors_s = [mytd.e_tot[0] - energy for energy in e_s]

# compute triplets 
mytd = tdscf.TDA(mf)
mytd.singlet = False
mytd = mytd.run(nstates=n_triplets)
mytd.analyze()
for i in range(mytd.nroots):
    avg_rdm1 += tda_density_matrix(mytd, i)
print('TDA CIS Triplet excited state total energy = ', mytd.e_tot)

# varying active space size 
mf = mytd._scf
e_t, xpoints_triplets = active_space_energies_td(mf, norbs, nelec, n_triplets)

# Calculate the error for triplets
errors_t = [mytd.e_tot[0] - energy for energy in e_t]

# Normalize
avg_rdm1 = avg_rdm1 / (n_singlets + n_triplets + 1)
# Compute natural orbital
Cdoc, Cact = get_natural_orbital_active_space(avg_rdm1, mf.get_ovlp(), thresh=0.00275)

# Create a plot
plt.figure(figsize=(10, 6))

# Plot the errors for singlets if there are singlet states
if errors_s:
    plt.plot(xpoints_singlets, errors_s, marker='o', linestyle='-', label='Singlets')

# Plot the errors for triplets if there are triplet states
if errors_t:
    plt.plot(xpoints_triplets, errors_t, marker='o', linestyle='-', label='Triplets')

plt.xlabel('Active Space Size (ncas, nelecas)')
plt.ylabel('Energy Error (eV)')
plt.title('Error in CI Energy vs. Active Space Size')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
