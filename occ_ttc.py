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
O          0.00000        0.00000        0.11779
H          0.00000        0.75545       -0.47116
H          0.00000       -0.75545       -0.47116'''
mol.spin = 0
mol.build()

Vnn = gto.mole.energy_nuc(mol)
Vne = mol.intor_symmetric('int1e_nuc')
T = mol.intor_symmetric('int1e_kin')

# Perform Restricted Hartree-Fock calculation
mf = scf.RHF(mol).apply(scf.addons.remove_linear_dep_)
mf.verbose = 4
mf.kernel()

H_core = mf.get_hcore()
E_init = mf.e_tot

F = mf.get_fock()
J, K = mf.get_jk()
Vsys = mf.get_veff()

print('fock')
print_matrix_as_integers(F)

mo_occ = mf.get_occ()
S = mf.get_ovlp()
C = mf.mo_coeff
P = mf.make_rdm1()

mftda = tdscf.TDA(mf)
mftda.singlet = True
mftda.run(nstates=1)
mftda.analyze()

eciss = mftda.e_tot
e0 = mftda.e
print('cis singlet total energy:', eciss)

mftda = tdscf.TDA(mf)
mftda.singlet = False
mftda.run(nstates=1)
mftda.analyze()
ecist = mftda.e_tot
e1 = mftda.e
print('cis triplet total energy:', ecist)

molden.from_mo(mol, 'ttc.molden', C)

orbital_energies = mf.mo_energy
nelec = mol.nelectron
num_orbitals = len(orbital_energies)
print(nelec, num_orbitals) 

# Calculate HOMO and LUMO indices from mo_occf
homo_index = np.where(mo_occ == 2)[0][-1]
lumo_index = homo_index + 1

active_sizes = list(range(2, nelec + 1, 2))
active_sizes = [size // 2 for size in active_sizes]

#list of excitation energy 0 for singlet 1 for triplet
e_0_list = []
e_1_list = []

#list of total energy 
e_ciss_list = []
e_cist_list = []

#list of total energy differences between TDA and active space TDA
error_s = []  
error_t = []  

#list of excitation energy differences between TDA and active space TDA
error_0 = []
error_1 = []

for i in active_sizes:
    '''
    act_list = list(range(homo_index - i +1, lumo_index + i))  
    env_list = list(range(0, homo_index - i+1))  
    vir_list = list(range(lumo_index + i, num_orbitals))   
    '''

    act_list = list(range(0, lumo_index ))
    #env_list = list(range())
    vir_list = list(range(lumo_index + i, num_orbitals))

    act_array = np.array(act_list)
    print(type(act_array))

    Cact = C[:, act_list]
    #Cenv = C[:, env_list]
    Cvir = C[:, vir_list]

    # localiz
    pyscf.tools.molden.from_mo(mol, "Cact.molden", Cact)

    nact = Cact.shape[1]
    #nenv = Cenv.shape[1]
    nvir = Cvir.shape[1]

    print('Number of Active AO: ', nact)
    #print('Number of Environment MOs: ', nenv)
    print('Number of Virtual MOs: ', nvir)

    D_A = 2.0 * Cact @ Cact.conj().T
    #D_B = 2.0 * Cenv @ Cenv.conj().T
    D_C = 2.0 * Cvir @ Cvir.conj().T

    #values for subspace, new mf
    h0 = gto.mole.energy_nuc(mol)
    Jenv, Kenv = mf.get_jk(dm = D_C)

    #integrals h0 (E_0)
    h0 += np.trace(D_C @ (H_core + 0.5 * Jenv - 0.25 * Kenv ))

    P_B = S @ (D_C)@ S
    mu = 1.0e6

    Venv = mf.get_veff(D_C)
    Vact = mf.get_veff(D_A)

#    Jenv,Kenv = mf.get_jk(D_B)
#    Jact,Kact = mf.get_jk(D_A)
#   JKenv = J - 0.5*K

    #new fock ao
    Vemb = Vsys - Vact + (mu * P_B)

    na_act = 0.5*np.trace(Cact.conj().T @ S @ P @ S @ Cact)
    nb_act = 0.5*np.trace(Cact.conj().T @ S @ P @ S @ Cact)

    na_act = round(na_act)
    nb_act = round(nb_act)
    n_elec = na_act + nb_act

    emb_mf = scf.RHF(mol)
    mol.nelectron = n_elec

    mol.build()

    print('emb mf.get_hcore,',emb_mf.get_hcore())
    print('emb mf.get_hcore-Vemb', Vemb)

    emb_mf.verbose = 4
    emb_mf.get_hcore = lambda *args: H_core + Vemb
    emb_mf.max_cycle = 200
    emb_mf.kernel(D_A)
    print('emb_mf.e_tot', emb_mf.e_tot)  

    # CIS calculations for singlets and triplets in Active space
    mf_eff = tdscf.TDA(emb_mf)
    mf_eff.singlet = True
    mf_eff.run(nstates=3)
    mf_eff.analyze()
    e_ciss = mf_eff.e_tot
    e00 = mf_eff.e
    e_ciss = min(e_ciss)
    e00 = min(e00)

    mf_eff = tdscf.TDA(emb_mf)
    mf_eff.singlet = False
    mf_eff.run(nstates=3)
    mf_eff.analyze()
    e_cist = mf_eff.e_tot
    e11 = mf_eff.e
    e_cist = min(e_cist)
    e11 = min(e11)

    print('e_ciss,e_cist', e_ciss, e_cist)
    e_ciss_list.append(e_ciss)
    e_cist_list.append(e_cist)

    error_s.append(e_ciss - eciss)
    error_t.append(e_cist - ecist)

    e_0_list.append(e00)
    e_1_list.append(e11)

    error_0.append(e00 - e0)
    error_1.append(e11 - e1)

print(mf.e_tot)
print(emb_mf.e_tot)
print('cis_singlet', eciss )
print('cis_T',ecist )

plt.figure(figsize=(10, 6))

active_sizes = list(range(2, nelec + 1, 2))

# Create the first subplot
plt.subplot(1, 2, 1)
plt.plot(active_sizes, e_0_list, marker='o', linestyle='-', label='Singlet')
plt.plot(active_sizes, e_1_list, marker='o', linestyle='-', label='Triplet')
plt.axhline(y=e0, color='blue', linestyle='--', label='CIS Singlet')
plt.axhline(y=e1, color='red', linestyle='--', label='CIS Triplet')

plt.xlabel('Active Space Size')
plt.ylabel('Excitation energy (in eV)')
plt.title('water excitation energy by active space sizes')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Create the second subplot
plt.subplot(1, 2, 2)
plt.plot(active_sizes, error_s, marker='o', linestyle='-', label='Singlet')
plt.plot(active_sizes, error_t, marker='o', linestyle='-', label='Triplet')

plt.xlabel('Active Space Size')
plt.ylabel('|E_cis - E_cis_act|')
plt.title('|E_cis - E_cis_act| vs Active space size')
plt.grid(True)
plt.legend()
plt.tight_layout()

file_name = "water2.png"
plt.savefig(file_name)
plt.show()
