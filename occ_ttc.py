import pyscf 
from pyscf import gto, scf, tdscf, ao2mo
from pyscf.tools import molden
import matplotlib.pyplot as plt

import numpy as np

def print_matrix_as_integers(mat):
    for row in mat:
        row_str = " ".join(str(int(elem)) for elem in row)
        print(row_str)

# Create the molecule
mol = gto.Mole()
mol.atom = '''
H       -3.4261000000     -2.2404000000      5.4884000000
H       -5.6274000000     -1.0770000000      5.2147000000
C       -3.6535000000     -1.7327000000      4.5516000000
H       -1.7671000000     -2.2370000000      3.6639000000
C       -4.9073000000     -1.0688000000      4.3947000000
H       -6.1631000000      0.0964000000      3.1014000000
C       -2.7258000000     -1.7321000000      3.5406000000
H       -0.3003000000      1.0832000000     -5.2357000000
C       -5.2098000000     -0.4190000000      3.2249000000
C       -2.9961000000     -1.0636000000      2.3073000000
H       -1.1030000000     -1.5329000000      1.3977000000
H       -0.4270000000     -0.8029000000     -0.8566000000
H        0.2361000000     -0.0979000000     -3.1273000000
C       -1.0193000000      1.0730000000     -4.4150000000
H       -2.4988000000      2.2519000000     -5.5034000000
C       -4.2740000000     -0.3924000000      2.1445000000
H       -5.5015000000      0.7944000000      0.8310000000
C       -2.0613000000     -1.0272000000      1.2718000000
C       -1.3820000000     -0.2895000000     -0.9772000000
C       -0.7171000000      0.4180000000     -3.2476000000
C       -2.2720000000      1.7395000000     -4.5690000000
H       -4.1576000000      2.2412000000     -3.6787000000
C       -4.5463000000      0.2817000000      0.9534000000
C       -2.3243000000     -0.3402000000      0.0704000000
C       -1.6528000000      0.3874000000     -2.1670000000
C       -3.1998000000      1.7341000000     -3.5584000000
C       -3.6044000000      0.3309000000     -0.0943000000
C       -2.9302000000      1.0591000000     -2.3292000000
C       -3.8665000000      1.0187000000     -1.2955000000
H       -4.8243000000      1.5256000000     -1.4217000000
'''
mol.basis = '6-31g'
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
print('cis singlet total energy:', eciss)

mftda = tdscf.TDA(mf)
mftda.singlet = False
mftda.run(nstates=1)
mftda.analyze()
ecist = mftda.e_tot
print('cis triplet total energy:', ecist)

molden.from_mo(mol, 'ttc.molden', C)

orbital_energies = mf.mo_energy
nelec = mol.nelectron
num_orbitals = len(orbital_energies)

# Calculate HOMO and LUMO indices from mo_occf
homo_index = np.where(mo_occ == 2)[0][-1]
lumo_index = homo_index + 1

#active_sizes = len(lumo_index,num_orbitals)
#active_sizes = list(range(0,active_sizes))


e_ciss_list = []
e_cist_list = []

error_s = []  
error_t = []  

active_sizes = []

for i in range(lumo_index,num_orbitals):

    active_sizes.append(i)

    act_list = list(range(0, i))
    vir_list = list(range(i, num_orbitals))

    act_array = np.array(act_list)

    Cact = C[:, act_list]
    Cvir = C[:, vir_list]

    pyscf.tools.molden.from_mo(mol, "Cact.molden", Cact)

    nact = Cact.shape[1]
    nvir = Cvir.shape[1]

    print('Number of Active AO: ', nact)

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

    emb_mf.verbose = 5
    emb_mf.get_hcore = lambda *args: H_core + Vemb
    emb_mf.max_cycle = 200
    emb_mf.kernel(D_A)
    print('emb_mf.e_tot', emb_mf.e_tot)  

    # CIS calculations for singlets and triplets
    es_eff = tdscf.TDA(emb_mf)
    es_eff.singlet = True
    es_eff.run(nstates=3)
    es_eff.analyze()
    e_ciss = min(es_eff.e_tot)


    et_eff = tdscf.TDA(emb_mf)
    et_eff.singlet = False
    et_eff.run(nstates=3)
    et_eff.analyze()
    e_cist = min(et_eff.e_tot)


    e_ciss_list.append(e_ciss)
    e_cist_list.append(e_cist)

    error_s.append(e_ciss - eciss)
    error_t.append(e_cist - ecist)


plt.figure(figsize=(10, 6))

# Create the first subplot
plt.subplot(1, 2, 1)
plt.plot(active_sizes, e_ciss_list, marker='o', linestyle='-', label='Singlet')
plt.plot(active_sizes, e_cist_list, marker='o', linestyle='-', label='Triplet')
plt.axhline(y=eciss, color='red', linestyle='--', label='CIS Singlet')
plt.axhline(y=ecist, color='blue', linestyle='--', label='CIS Triplet')

plt.xlabel('Active Space Size')
plt.ylabel('Total energy(hartree)')
plt.title('Total energy vs Active space size')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Create the second subplot
plt.subplot(1, 2, 2)
plt.plot(active_sizes, error_s, marker='o', linestyle='-', label='Singlet')
plt.plot(active_sizes, error_t, marker='o', linestyle='-', label='Triplet')

plt.xlabel('Active Space Size')
plt.ylabel('Energy Difference(hartree)')
plt.title('|E_cis - E_cis_act|')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save and display the plots
plt.show()

file_name = "ttc_cno.png"
plt.savefig(file_name)

'''
    #rotate 1electron terms to active space
    #TDA hsppens in ao basis/ no rotation needed just projector
    Hact = Cact.T @ H_core @ Cact
    Jact = Cact.T @ J @ Cact
    Kact = Cact.T @ K @ Cact
'''
    #integrals h1 (h_{pq}D_{pq})
 #   h1 = Hact + Jact - 0.5 * Kact'''
