import pyscf 
from pyscf import gto, scf, tdscf, ao2mo
from pyscf.tools import molden
import matplotlib.pyplot as plt

import scipy
import numpy as np
from numpy import linalg

def print_matrix_as_integers(mat):
    for row in mat:
        row_str = " ".join(str(int(elem)) for elem in row)
        print(row_str)

# Create the molecule
mol = gto.Mole()
mol.atom = '''
        O   0. 0. 0.
        H   0. 1. 0.
        H   0. 0. 1.
    '''
mol.unit = 'B'
mol.basis = '6-31g'
mol.spin = 0
mol.build()

Vnn = gto.mole.energy_nuc(mol)
Vne = mol.intor_symmetric('int1e_nuc')
T = mol.intor_symmetric('int1e_kin')

# Perform Restricted Hartree-Fock calculation
mf = scf.RHF(mol)
mf.verbose = 4
mf.kernel()

H_core = mf.get_hcore()
E_init = mf.e_tot

F = mf.get_fock()
J, K = mf.get_jk()
Vsys = mf.get_veff()

S = mf.get_ovlp()
C = mf.mo_coeff
P = mf.make_rdm1()

mftda = tdscf.TDA(mf)
mftda.singlet = True
mftda.run(nstates=1)
mftda.analyze()
eciss = mftda.e_tot
print('cis singlet total energy:', eciss)

mftda = tdscf.TDA(mf)
mftda.singlet = False
mftda.run(nstates=1)
mftda.analyze()
ecist = mftda.e_tot
print('cis triplet total energy:', ecist)

molden.from_mo(mol, 'H2O.molden', C)

orbital_energies = mf.mo_energy
nelec = mol.nelectron
num_orbitals = len(orbital_energies)
print(nelec, num_orbitals) 

mo_occ = mf.get_occ()
# Calculate HOMO and LUMO indices from mo_occf
homo_index = np.where(mo_occ == 2)[0][-1]
lumo_index = homo_index + 1

active_sizes = list(range(2, nelec + 1, 2))
active_sizes = [size // 2 for size in active_sizes]
    
e_ciss_list = []
e_cist_list = []

error_s = []  
error_t = []  

for i in active_sizes:
    act_list = list(range(homo_index - i +1, lumo_index + i))  
    env_list = list(range(0, homo_index - i+1))  
    vir_list = list(range(lumo_index + i, num_orbitals))   

    act_array = np.array(act_list)
    print(type(act_array))

    Cact = C[:, act_list]
    Cenv = C[:, env_list]
    Cvir = C[:, vir_list]

    nact = Cact.shape[1]
    nenv = Cenv.shape[1]
    nvir = Cvir.shape[1]

    print('Number of Active AO: ', nact)
    print('Number of Environment MOs: ', nenv)
    print('Number of Virtual MOs: ', nvir)

    D_A = 2.0 * Cact @ Cact.conj().T
    D_B = 2.0 * Cenv @ Cenv.conj().T
    D_C = 2.0 * Cvir @ Cvir.conj().T

    Venv = mf.get_veff(D_B)
    Vact = mf.get_veff(D_A)

    P_B = S @ (D_B +  D_C)@ S
    mu = 1.0e6

    #new fock ao
    Vemb = Vsys - Vact + (mu * P_B)

    mol = gto.Mole()
    mol.atom = '''
        O   0. 0. 0.
        H   0. 1. 0.
        H   0. 0. 1.
    '''
    mol.unit = 'B'
    mol.basis = '6-31g'
    mol.spin = 0

    na_act = 0.5*np.trace(Cact.conj().T @ S @ P @ S @ Cact)
    nb_act = 0.5*np.trace(Cact.conj().T @ S @ P @ S @ Cact)

    na_act = round(na_act)
    nb_act = round(nb_act)
    n_elec = na_act + nb_act

    emb_mf = scf.RHF(mol)
    mol.nelectron = n_elec

    mol.build()

    emb_mf.verbose = 1
    emb_mf.get_hcore = lambda *args: H_core + Vemb
    emb_mf.max_cycle = 200
    emb_mf.kernel(D_A)
    print('e_emb_mf.e_tot', emb_mf.e_tot)  

    # CIS calculations for singlets and triplets
    mf_eff = tdscf.TDA(emb_mf)
    mf_eff.singlet = True
    mf_eff.run(nstates=1)
    mf_eff.analyze()
    e_ciss = mf_eff.e_tot

    mf_eff = tdscf.TDA(emb_mf)
    mf_eff.singlet = False
    mf_eff.run(nstates=1)
    mf_eff.analyze()
    e_cist = mf_eff.e_tot

    print('e_ciss,ecist', e_ciss, e_cist)
    e_ciss_list.append(e_ciss)
    e_cist_list.append(e_cist)

    error_s.append(e_ciss - eciss)
    error_t.append(e_cist - ecist)

plt.figure(figsize=(10, 6))

active_sizes = list(range(2, nelec + 1, 2))

# Plot e_ciss and e_cist with respect to active space size
plt.plot(active_sizes, e_ciss_list, marker='o', linestyle='-', label='Singlet')
plt.plot(active_sizes, e_cist_list, marker='o', linestyle='-', label='Triplet')

plt.xlabel('Active Space Size')
plt.ylabel('Total energy')
plt.title('Total energy vs Active space size')
plt.grid(True)
plt.legend()
plt.tight_layout()

file_name = "h2o_cno_e_tot.png"
plt.savefig(file_name)

# Plot e_ciss and e_cist with respect to active space size
plt.plot(active_sizes, error_s, marker='o', linestyle='-', label='Singlet')
plt.plot(active_sizes, error_t, marker='o', linestyle='-', label='Triplet')

plt.xlabel('Active Space Size')
plt.ylabel('Energy Difference')
plt.title('TDA CIS Excited State Energy Differences - CIS in Active Space')
plt.grid(True)
plt.legend()
plt.tight_layout()

file_name = "H2O.png"
plt.savefig(file_name)

