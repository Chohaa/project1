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
C         -2.30986        0.50909        0.01592
C         -0.98261        0.43259       -0.05975
C         -0.26676       -0.68753        0.57690
C          1.06323       -0.78274        0.51273
H         -2.84237        1.33139       -0.45081
H         -2.89518       -0.26464        0.20336
H         -0.43345        1.20285       -0.59462
H         -0.82592       -1.45252        1.10939
H          1.65570       -0.03776       -0.01031
H          1.57073       -1.61569        0.98847
'''
mol.basis = 'sto-3g'
mol.spin = 0
mol.build()

# Perform Restricted Hartree-Fock calculation
mf = scf.RHF(mol).run()

H_core = mf.get_hcore()
E_init = mf.e_tot
print(" Hartree-Fock Energy: %12.8f" % mf.e_tot)

F = mf.get_fock()
J, K = mf.get_jk()
Vsys = mf.get_veff()
mo_occ = mf.get_occ()

S = mf.get_ovlp()
C = mf.mo_coeff
Cocc = C[:,mo_occ>0]
D = mf.make_rdm1()
mo_energy = mf.mo_energy

X = scipy.linalg.sqrtm(S)
Xinv = np.linalg.inv(X)

#orthogonalization
C = X @ C
F = C.conj().T @ F @ C

num_orbitals = len(C)

print(" Number of electrons found %12.8f" %np.trace(S@D))
print('Number of Orbitals %12.8f:',num_orbitals)

#CIS
mftda = tdscf.TDA(mf)
mftda.singlet = True
mftda.run(nstates=1)
mftda.analyze()
eciss = mftda.e_tot
e0 = mftda.e
print('cis excitation energy:', e0)
print('cis singlet total energy:', eciss)

mftda.singlet = False
mftda.run(nstates=1)
mftda.analyze()
ecist = mftda.e_tot
e1 = mftda.e
print('cis excitation energy:', e1)
print('cis triplet total energy:', ecist)

molden.from_mo(mol, 'BD_vir.molden', C)

# Calculate HOMO and LUMO indices from mo_occ
homo_index = np.where(mo_occ == 2)[0][-1]
lumo_index = homo_index + 1

print('homoidx',homo_index)
print('lumoidx',lumo_index)

#active_sizes = len(lumo_index,num_orbitals)
#active_sizes = list(range(0,active_sizes))

estot_list = []
ettot_list = []

toterror_s = []  
toterror_t = []  

active_sizes = []

e000 = []
e111 = []

for i in range(0,homo_index):

    active_sizes.append(num_orbitals-i)

    act_list = list(range(i, num_orbitals))
    vir_list = list(range(0, i))

    print('actlist', act_list)
    print('virlist', vir_list)

    act_array = np.array(act_list)

    Cact = C[:, act_list]
    act_occ = np.array(mo_occ[act_list])
    Cvir = C[:, vir_list]
    vir_occ = np.array(mo_occ[vir_list])

    nact = len(act_list)
    nvir = len(vir_list)

    print('Number of Active orbitals: ', nact)
    print('Number of Virtual orbitals: ', nvir)

    
    D_A = np.dot(Cact*act_occ, Cact.conj().T)
    #D_B = 2.0 * Cenv @ Cenv.conj().T
    D_C = np.dot(Cvir*vir_list, Cvir.conj().T)


    D_tot = D_A + D_C
    print_matrix_as_integers(D - D_tot)



    #projector
    P_B = S @ (D_C) @ S
    mu = 1.0e6

    Vact = mf.get_veff(dm = D_A)
    Venv = mf.get_veff(dm = D_C)

    #new fock ao
    Vemb = Vsys - Vact + (mu * P_B)
    verror = Vsys - Vact
    print('verror',verror)

    D_act = D_tot[:,act_list]
    D_act = D_act[act_list,:]

    print('d_act shape', D_act.shape)
    n_act = 2 * round(0.5*np.trace(D_A))

    print('number of active orbitals',nact)
    print('number of virtual orbitals: ',nvir)
    print('number of electrons in active space', n_act)

    emb_mf = scf.RHF(mol)
    mol.nelectron = n_act
    mol.build()
    
    emb_mf.verbose = 4
    emb_mf.get_hcore = lambda *args: H_core + Vemb
    emb_mf.max_cycle = 200
    emb_mf.kernel(dm0=D_A)

    eerror = emb_mf.e_tot - mf.e_tot
    print('eerror', eerror)

    molden.from_mo(mol, 'BD_cact_vir.molden', Cact)

    # CIS calculations for singlets and triplets
    es_eff = tdscf.TDA(emb_mf)
    es_eff.singlet = True
    es_eff.run(nstates=5)
    es_eff.analyze()
    estot = min(es_eff.e_tot)
    e00 = min(es_eff.e)
  
    et_eff = tdscf.TDA(emb_mf)
    et_eff.singlet = False
    et_eff.run(nstates=5)
    et_eff.analyze()
    ettot = min(et_eff.e_tot)
    e11 = min(et_eff.e)

    e000.append(e00)
    e111.append(e11)

    estot_list.append(estot)
    ettot_list.append(ettot)

    toterror_s.append(estot - eciss)
    toterror_t.append(ettot - ecist)

plt.figure(figsize=(10, 6))

# Create the first subplot
plt.subplot(1, 2, 1)
plt.plot(active_sizes, e000, marker='o', linestyle='-', label='Singlet')
plt.plot(active_sizes, e111, marker='o', linestyle='-', label='Triplet')
plt.axhline(y=e0, color='blue', linestyle='--', label='CIS Singlet')
plt.axhline(y=e1, color='red', linestyle='--', label='CIS Triplet')


plt.xlabel('# orbitals in active space')
plt.ylabel('Excitation energy (in eV)')
plt.title('Excitation energy of active space')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Create the second subplot
plt.subplot(1, 2, 2)
plt.plot(active_sizes, estot_list, marker='o', linestyle='-', label='Singlet')
plt.plot(active_sizes, ettot_list, marker='o', linestyle='-', label='Triplet')
plt.axhline(y=eciss, color='red', linestyle='--', label='CIS Singlet')
plt.axhline(y=ecist, color='blue', linestyle='--', label='CIS Triplet')

plt.xlabel('# orbitals in active space')
plt.ylabel('Total energy(hartree)')
plt.title('Total energy of active space_VIR')
plt.grid(True)
plt.legend()
plt.tight_layout()

file_name = "BD_allvir.png"
plt.savefig(file_name)

plt.show()

