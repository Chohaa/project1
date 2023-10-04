import pyscf 
from pyscf import gto, scf, tdscf, ao2mo

import scipy
import numpy as np
from numpy import linalg

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


def get_natural_orbital_active_space(rdm, S, thresh=.01, save_lengths=False):
    Ssqrt = scipy.linalg.sqrtm((S + S.T) / 2.0)
    Sinvsqrt = scipy.linalg.inv(Ssqrt)
    print(" Number of electrons found %12.8f" % np.trace(S @ rdm))
    Dtot = Ssqrt.T @ rdm @ Ssqrt
    D_evals, D_evecs = np.linalg.eigh((Dtot + Dtot.T) / 2.0)
    sorted_list = np.argsort(D_evals)[::-1]
    D_evals = D_evals[sorted_list]
    D_evecs = D_evecs[:, sorted_list]
    act_list = []
    doc_list = []
    for idx, n in enumerate(D_evals):
        print(" %4i = %12.8f" % (idx, n), end="")
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
    
    if save_lengths:
        # Save the lengths of act_list and doc_list to a text file
        with open('list_lengths.txt', 'w') as file:
            file.write(f"Length of act_list: {len(act_list)}")
            file.write(f"Length of doc_list: {len(doc_list)}")

    D_evecs = Sinvsqrt @ D_evecs
    Cdoc = D_evecs[:, doc_list]
    Cact = D_evecs[:, act_list]
    
    return Cdoc, Cact, doc_list, act_list  

def semi_canonicalize(C,F):
    e,v = np.linalg.eigh(C.conj().T @ F @ C)
    C_bar = C @ v
    return C_bar, e

def generate_subsets(nacts):
    subsets = list(range(1, nacts + 1))  # Create a list of integers from 1 to nacts
    half = nacts // 2
    act = []  # Create an empty list to store subsets
    for i in range(half):
        subset = subsets[half - 2 - i : half + i]  # Select elements as specified
        act.append(subset)
    # Return the list of subsets
    return act

def print_matrix_as_integers(mat):
    for row in mat:
        # Cast elements to integers, then join them as strings with proper spacing
        row_str = " ".join(str(int(elem)) for elem in row)
        print(row_str)


# Create the molecule
mol = gto.Mole()
mol.atom = '''
O          0.00000        0.00000        0.11779
H          0.00000        0.75545       -0.47116
H          0.00000       -0.75545       -0.47116'''
mol.basis = '6-31g'
mol.spin = 0
mol.build()

# Perform Restricted Hartree-Fock calculation
mf = scf.RHF(mol)
mf.verbose = 4
mf.kernel()


# mean-field
F = mf.get_fock()
print(F)
Vnn = gto.mole.energy_nuc(mol)
J, K = mf.get_jk()
V = mf.get_veff()
P = mf.make_rdm1()
S = mf.get_ovlp() # overlab matrix in AO basis
C = mf.mo_coeff # mo coefficient 

H_core = mf.get_hcore()
E_init = mf.e_tot

Vne = mol.intor_symmetric('int1e_nuc')
T = mol.intor_symmetric('int1e_kin')

atom_index = 0

ao_list = []
for ao_idx,ao in enumerate(mf.mol.ao_labels(fmt=False)):
    for i in range(nact):
        if ao[0] == i:
            ao_list.append(ao_idx)

print('AOs:', ao_list)
n_aos = len(ao_list)
print(n_aos)


#values for subspace
n_occ = mf.tdscf.nocc
act = generate_subsets(nacts)

virt = len(n_occ) + len(act)
mu = 10^6
r_c = mu * S @ c[:virt,:] @ c[:virt,:].T @ S

# Embeded mean-field

Cenv, e_orb_env = semi_canonicalize(Cenv, F)
Cact, e_orb_act = semi_canonicalize(Cact, F)

D_B = 2.0 * Cenv @ Cenv.conj().T
D_A = 2.0 * Cact @ Cact.conj().T





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

# varying active space size singlet


# compute triplets 
mytd = tdscf.TDA(mf)  
mytd.singlet = False
mytd = mytd.run(nstates=n_triplets)
mytd.analyze()
for i in range(mytd.nroots):
    avg_rdm1 += tda_density_matrix(mytd, i)
print('TDA CIS Triplet excited state total energy = ', mytd.e_tot)

# varying active space size 

# New density matrix
avg_rdm1 = avg_rdm1 / (n_singlets + n_triplets + 1)
Cdoc, Cact, doc_list, act_list = get_natural_orbital_active_space(avg_rdm1, S, thresh=0.01, save_lengths=True)



print('number of orbitals, electrons =', norbs, nelec)
print('TDA CIS singlet excited state total energy = ', e_ciss)
print('TDA CIS triplet excited state total energy = ', e_cist)

# Create a plot
plt.figure(figsize=(10, 6))

# Plot the errors for singlets if there are singlet states
if errors_s:
    plt.plot(xpoints_singlets, errors_s, marker='o', linestyle='-', label='Singlets')

# Plot the errors for triplets if there are triplet states
if errors_t:
    plt.plot(xpoints_triplets, errors_t, marker='o', linestyle='-', label='Triplets')

plt.xlabel('Active Space Size (ncas, nelecas)')
plt.ylabel('Energy difference; CIS - active space (eV)')
plt.title('CI Energy vs. Active Space Size')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('ttc_cis_activespace.png')
