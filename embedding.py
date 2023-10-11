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
    #Dtot = 
    #D_evecs = 
    Dtot = Ssqrt.T @ rdm @ Ssqrt
    D_evals, D_evecs = np.linalg.eigh((Dtot + Dtot.T) / 2.0)
    sorted_list = np.argsort(D_evals)[::-1]
    D_evals = D_evals[sorted_list]
    D_evecs = D_evecs[:, sorted_list]
    act_list = []
    doc_list = []
    vir_list = []
    for idx, n in enumerate(D_evals):
        print(" %4i = %12.8f" % (idx, n), end="")
        if n < 2.0 - thresh:
            if n > thresh:
                act_list.append(idx)
                print(" Active")
            else:
                vir_list.append(idx)
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
    Cvir = D_evecs[:, vir_list]
    
    return Cdoc, Cact, doc_list, act_list, vir_list  

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

F = mf.get_fock()
Vnn = gto.mole.energy_nuc(mol)
J, K = mf.get_jk()
#V = mf.get_veff()
P = mf.make_rdm1()

S = mf.get_ovlp() # overlab matrix in AO basis
C = mf.mo_coeff # mo coefficient 

H_core = mf.get_hcore()
E_init = mf.e_tot

Vne = mol.intor_symmetric('int1e_nuc')
T = mol.intor_symmetric('int1e_kin')

Cdoc, Cact, doc_list, act_list, vir_list = get_natural_orbital_active_space(P, S, thresh=0.01, save_lengths=True)

Cenv = C[:, doc_list]
Cact = C[:, act_list]
Cvir = C[:, vir_list]

nact = Cact.shape[1]
nenv = Cenv.shape[1]
nvir = Cvir.shape[1]

print('Number of Active MOs: ', nact)
print('Number of Environment MOs: ', nenv)
print('Number of Virtual MOs: ', nvir)

D_C = 2.0 * Cvir @ Cvir.conj().T
D_B = 2.0 * Cenv @ Cenv.conj().T
D_A = 2.0 * Cact @ Cact.conj().T

#values for subspace, new mf
Jenv, Kenv = mf.get_jk(dm = D_B)
Jact, Kact = mf.get_jk(dm = D_A)

P_B = S @ D_B @ S + S @ D_C @ S
mu = 1.0e6

Vact = Jact + Kact
Vemb = J + K - Vact + (mu * P_B)

'''
#DFT
Venv = mf.get_veff(dm=D_B)
Vact = mf.get_veff(dm=D_A)
V = mf.get_veff()
'''
mol = gto.Mole()
mol.atom = '''
O          0.00000        0.00000        0.11779
H          0.00000        0.75545       -0.47116
H          0.00000       -0.75545       -0.47116'''
mol.basis = '6-31g'
mol.spin = 0
emb_mf = scf.RHF(mol)

mol.nelectron = int(np.trace(S@P))
emb_mf.verbose = 4
emb_mf.get_hcore =  lambda *args: H_core + Vemb
emb_mf.max_cycle = 200
emb_mf.kernel(dm0=D_A)

S_emb = emb_mf.get_ovlp(mol)
F_emb = emb_mf.get_fock()
C_emb = emb_mf.mo_coeff
P_emb = emb_mf.make_rdm1


# Embeded mean-field

Cenv, e_orb_env = semi_canonicalize(Cenv, F)
Cact, e_orb_act = semi_canonicalize(Cact, F)

# Compute singlets and triplets
n_singlets = 1
n_triplets = 1

# CIS
# compute singlets
mf_eff = tdscf.TDA(emb_mf)
mf_eff.singlet = True
mf_eff = mf_eff.run(nstates=n_singlets)
mf_eff.analyze()
for i in range(mf_eff.nroots):
    avg_rdm1 += tda_density_matrix(mf_eff, i)
print('TDA CIS singlet excited state total energy = ', mf_eff.e_tot)

# varying active space size singlet


# compute triplets 
mf_eff = tdscf.TDA(emb_mf)  
mf_eff.singlet = False
mf_eff = mf_eff.run(nstates=n_triplets)
mf_eff.analyze()
for i in range(mf_eff.nroots):
    avg_rdm1 += tda_density_matrix(mf_eff, i)
print('TDA CIS Triplet excited state total energy = ', mf_eff.e_tot)

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
