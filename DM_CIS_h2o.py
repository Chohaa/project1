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

    for i in range(2, norbs, 2):
        for j in range(2, nelec, 2):
            ncas, nelecas = i, j
            if ncas <= norbs and nelecas <= ncas:  # Ensure valid CAS setup
                mycas = mcscf.CASCI(mf, ncas, nelecas)
                mycas.kernel()
                energies.append(mycas.e_tot)
                xpoints.append(f"({ncas}, {nelecas})")

    return energies, xpoints


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
e_s, xpoints_singlets = active_space_energies_td(mf, 14, nelec, n_singlets)
e_ciss = mytd.e_tot

# Calculate the error for singlets
errors_s = [e_ciss[0] - energy for energy in e_s]

# compute triplets 
mytd = tdscf.TDA(mf)
mytd.singlet = False
mytd = mytd.run(nstates=n_triplets)
mytd.analyze()
for i in range(mytd.nroots):
    avg_rdm1 += tda_density_matrix(mytd, i)
print('TDA CIS Triplet excited state total energy = ', mytd.e_tot)

#EOM-CC
e_s, c_s = mycc.eomee_ccsd_singlet(nroots=1)
e_t, c_t = mycc.eomee_ccsd_triplet(nroots=1)
# Calculate the error
errors_eom = (e_ciss - e_s)
errort_eom = (e_cist - e_t)

# varying active space size 
mf = mytd._scf
e_t, xpoints_triplets = active_space_energies_td(mf, 14, nelec, n_triplets)
e_cist = mytd.e_tot

# Calculate the error for triplets
errors_t = [e_ciss - energy for energy in e_t]

# Normalize
avg_rdm1 = avg_rdm1 / (n_singlets + n_triplets + 1)
# Compute natural orbital
Cdoc, Cact = get_natural_orbital_active_space(avg_rdm1, mf.get_ovlp(), thresh=0.00275)

#TDDFT
mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.kernel()

mytd = tddft.TDDFT(mf)
mytd.nstates = 10
mytd.singlet = True
mytd.kernel()
mytd.analyze()

e_dft_s = mytd.e_tot

# Calculate the error
errors_tddft = (e_ciss -  e_dft_s)

mytd = tddft.TDDFT(mf)
mytd.nstates = 10
mytd.singlet = False
mytd.kernel()
mytd.analyze()

e_dft_t = mytd.e_tot

# Calculate the error
errort_tddft = (e_cist - e_eft_t)

print('TDA CIS singlet excited state total energy = ', e_ciss)
print('TDA CIS Triplet excited state total energy = ', e_cist)
print('EOM-CC singlet excited state total energy = ', e_s, errors_eom)
print('EOM-CC triplet excited state total energy = ', e_t, errort_eom)
print('TDDFT singlet excited state total energy = ', e_dft_s, errors_tddft)
print('TDDFT singlet excited state total energy = ', e_dft_t, errort_tddft)


# Create a plot
plt.figure(figsize=(10, 6))

# Plot the errors for singlets if there are singlet states
if errors_s:
    plt.plot(xpoints_singlets, errors_s, marker='o', linestyle='-', label='Singlets')

# Plot the errors for triplets if there are triplet states
if errors_t:
    plt.plot(xpoints_triplets, errors_t, marker='o', linestyle='-', label='Triplets')
    

plt.xlabel('Active Space Size (ncas, nelecas)')
plt.ylabel('Energy Error; CIS - active space (eV)')
plt.title('Error in CI Energy vs. Active Space Size')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
