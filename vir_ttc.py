import pyscf 
from pyscf import gto, scf, tdscf
from pyscf.tools import molden
import matplotlib.pyplot as plt

import numpy as np
import scipy

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
mol.basis = 'sto-3g'
mol.spin = 0
mol.build()

# Perform Restricted Hartree-Fock calculation
mf = scf.RHF(mol).run(verbose = 4)

H_core = mf.get_hcore()
e_hf = mf.kernel()

F = mf.get_fock()
J, K = mf.get_jk()
Vsys = mf.get_veff()
mo_occ = mf.get_occ()
mo_energy = mf.mo_energy

S = mf.get_ovlp()
C = mf.mo_coeff
Cocc = C[:,mo_occ>0]
D = mf.make_rdm1()

X = scipy.linalg.sqrtm(S)
Xinv = np.linalg.inv(X)

nF = Xinv @ F @ Xinv
nC = Xinv @ C

molden.from_mo(mol, 'water.molden',nC)

num_orbitals = len(C)

tda_h2o = tdscf.TDA(mf)
tda_h2o.nstates = 5
conv_tol = 1e-12
e0 = min(tda_h2o.kernel()[0])
print('tda e (Hartree)', e0)

tda_h2o = tdscf.TDA(mf)
tda_h2o.nstates = 5
tda_h2o.singlet = False
conv_tol = 1e-12
e_t0 = min(tda_h2o.kernel()[0])
print('triplet tda e (Hartree)', e_t0)


act_list = []
doc_list = []

# Calculate HOMO and LUMO indices from mo_occ
homo_index = np.where(mo_occ == 2)[0][-1]
lumo_index = homo_index + 1

print('homoidx',homo_index)
print('lumoidx',lumo_index)

#active_sizes = len(lumo_index,num_orbitals)
#active_sizes = list(range(0,active_sizes))

active_sizes = []
e000 = []
e111 = []
hf = []

for i in range(0,homo_index):

    active_sizes.append(num_orbitals-i)

    act_list = list(range(i, num_orbitals))
    vir_list = list(range(0, i))

    act_array = np.array(act_list)

    Cact = C[:, act_list]
    act_occ = np.array(mo_occ[act_list])
    Cvir = C[:, vir_list]
    vir_occ = np.array(mo_occ[vir_list])

    nact = len(act_list)
    nvir = len(vir_list)

    D_A = np.dot(Cact*act_occ, Cact.conj().T)
    D_C = np.dot(Cvir*vir_occ, Cvir.conj().T)
    D_tot = D_A +D_C

    #projector
    P_B = S @ D_C @ S
    mu = 1.0e6

    Vsys = mf.get_veff(dm=D_tot)
    Vact = mf.get_veff(dm=D_A) 
    Venv = mf.get_veff(dm=D_C)

    #new fock ao
    Vemb = Vsys - Vact + (mu * P_B)
    verror = Vsys - Vact


    n_act = 2*round(0.5 * np.trace(S@D_A))

    print('Number of Active orbitals: ', nact)
    print('Number of Virtual orbitals: ', nvir) 
    print('Number of electrons in active space',n_act) 

    emb_mf = scf.RHF(mol)
    mol.nelectron = n_act
    mol.build()

    emb_mf.verbose = 4
    emb_mf.get_hcore = lambda *args: H_core + Vemb
    emb_mf.max_cycle = 200
    e_hf_act = emb_mf.kernel(dm0=D_A)
    print('ehfact',e_hf_act)

    #molden.from_mo(mol, 'BDocc__cact.molden', )  

    emb_tda = tdscf.TDA(emb_mf)
    emb_tda.nstates = 5
    conv_tol = 1e-12
    e = min(emb_tda.kernel()[0])
    print('tda e(Hartree)', e)

    emb_tda_t = tdscf.TDA(emb_mf)
    emb_tda_t.nstates = 5
    emb_tda_t.singlet = False
    e_t = min(emb_tda_t.kernel()[0])
    print('triplet tda e (Hartree)', e_t)

    hf.append(e_hf_act)
    e000.append(e)
    e111.append(e_t)


plt.figure(figsize=(10, 6))

# Create the first subplot
plt.subplot(1, 2, 1)
plt.plot(active_sizes, e000, marker='o', linestyle='-', label='Singlet')
plt.plot(active_sizes, e111, marker='o', linestyle='-', label='Triplet')
plt.axhline(y=e0, color='blue', linestyle='--', label='CIS Singlet')
plt.axhline(y=e_t0, color='red', linestyle='--', label='CIS Triplet')


plt.xlabel('# orbitals in active space')
plt.ylabel('Excitation energy (in eV)')
plt.title('Excitation energy of active space')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()

exit()

# Create the second subplot
plt.subplot(1, 2, 2)
plt.plot(active_sizes, hf, marker='o', linestyle='-', label='Hartree fock energy')
plt.axhline(y=e_hf, color='red', linestyle='--', label='Hartree fock in active space')

plt.xlabel('# orbitals in active space')
plt.ylabel('Total energy (hartree)')
plt.title('Total energy of active space_OCC')
plt.grid(True)
plt.legend()
plt.tight_layout()

file_name = "ttc_allocc.png"
plt.savefig(file_name)

plt.show()
