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

def get_natural_orbital_active_space(rdm, S, thresh=0.01):
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
    return act_list, doc_list


def active_space_energies_td(mf, norbs, nelec, n_singlets):
    energies = []  # Store the energies for different active space sizes
    xpoints = []  # Store labels for the x-axis

    for i in range(2, norbs, 2):
        for j in range(2, nelec, 2):
            ncas, nelecas = i, j
            if ncas <= norbs and nelecas <= ncas:  # Ensure valid CAS setup
                # Check if the number of virtual orbitals is non-negative
                nvir = norbs - nelecas
                if nvir >= 0:
                    mycas = mcscf.CASCI(mf, ncas, nelecas)
                    try:
                        mycas.kernel()
                        energies.append(mycas.e_tot)
                        xpoints.append(f"({ncas}, {nelecas})")
                    except Exception as e:
                        print(f"Error for CAS({ncas}, {nelecas}): {str(e)}")

    return energies, xpoints


# Create the molecule
mol = gto.Mole()
mol.atom = '''
O          0.00000        0.00000        0.11779
H          0.00000        0.75545       -0.47116
H          0.00000       -0.75545       -0.47116'''
mol.basis = '6-31g'
mol.basis = '6-31g'
mol.spin = 0
mol.build()

# Perform Restricted Hartree-Fock calculation
mf = scf.RHF(mol)
mf.verbose = 4
mf.get_init_guess(mol, key='minao')
mf.kernel() 

norbs = len(mf.mo_coeff)
nelec = int(mol.nelectron)

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

mfs = mytd._scf
e_s, xpoints_singlets = active_space_energies_td(mfs,norbs, nelec, n_singlets)
e_ciss = mytd.e_tot

# Calculate the error for singlets
errors_s = [e_ciss[0] - energy for energy in e_s]

# compute triplets
mytd = tdscf.TDA(mfs)  # Fixed variable name from mftd to mfs
mytd.singlet = False
mytd = mytd.run(nstates=n_triplets)
mytd.analyze()
for i in range(mytd.nroots):
    avg_rdm1 += tda_density_matrix(mytd, i)
print('TDA CIS Triplet excited state total energy = ', mytd.e_tot)

# varying active space size
mfs = mytd._scf
e_t, xpoints_triplets = active_space_energies_td(mfs, norbs, nelec, n_triplets)
e_cist = mytd.e_tot

# Calculate the error for tripletnit_guess(mol, key='minao')
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

mfs = mytd._scf
e_s, xpoints_singlets = active_space_energies_td(mfs,norbs, nelec, n_singlets)
e_ciss = mytd.e_tot

# Calculate the error for singlets
errors_s = [e_ciss[0] - energy for energy in e_s]

# compute triplets
mytd = tdscf.TDA(mfs)  # Fixed variable name from mftd to mfs
mytd.singlet = False
mytd = mytd.run(nstates=n_triplets)
mytd.analyze()
for i in range(mytd.nroots):
    avg_rdm1 += tda_density_matrix(mytd, i)
print('TDA CIS Triplet excited state total energy = ', mytd.e_tot)

# varying active space size
mfs = mytd._scf
e_t, xpoints_triplets = active_space_energies_td(mfs, norbs, nelec, n_triplets)
e_cist = mytd.e_tot

# Calculate the error for triplets
errors_t = [e_cist - energy for energy in e_t]

# new density matrix
avg_rdm1 = avg_rdm1 / (n_singlets + n_triplets + 1)
# Compute natural orbital
Cdoc, Cact = get_natural_orbital_active_space(avg_rdm1, mf.get_ovlp(), thresh=0.00275)

print('TDA CIS singlet excited state total energy = ', e_ciss)
print('TDA CIS Triplet excited state total energy = ', e_cist)

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
plt.savefig('ttc_cis_activespace.png')

errors_t = [e_cist - energy for energy in e_t]

# new density matrix
avg_rdm1 = avg_rdm1 / (n_singlets + n_triplets + 1)
# Compute natural orbital
Cdoc, Cact = get_natural_orbital_active_space(avg_rdm1, mf.get_ovlp(), thresh=0.00275)

print('TDA CIS singlet excited state total energy = ', e_ciss)
print('TDA CIS Triplet excited state total energy = ', e_cist)

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
plt.savefig('ttc_cis_activespace.png')
