import matplotlib.pyplot as plt
from pyscf import gto, scf, mcscf, tdscf

def active_space_energies(mf, norbs, nelec):
    # Calculate natural orbitals
    noons, natorbs = mcscf.addons.make_natural_orbitals(mf)

    energies = []  # Store the energies for different active space sizes

    for i in range(2, norbs):
        for j in range(2, nelec, 2):
            ncas, nelecas = i, j
            if ncas <= nelec <= norbs:
                mycas = mcscf.CASCI(mf, ncas, nelecas)
                mycas.kernel(natorbs)

                energies.append(mycas.e_tot)

    return energies

def main():
    # Define the molecular geometry and build the molecule
    mol = gto.Mole()
    mol.atom = '''
    O          0.00000        0.00000        0.11779
    H          0.00000        0.75545       -0.47116
    H          0.00000       -0.75545       -0.47116'''
    mol.basis = '6-31g'
    mol.spin = 0
    mol.build()

    # Initialize and run the RHF calculation
    mf = scf.RHF(mol)
    mf.verbose = 4
    mf.get_init_guess(mol, key='minao')
    mf.conv_tol = 1e-9
    mf.run()

    # Calculate the active space energies
    norbs = len(mf.mo_coeff)
    nelec = int(mol.nelectron / 2)
    energies = active_space_energies(mf, norbs, nelec)

    # Create xpoints corresponding to the active space sizes (string labels)
    xpoints = [f"({i}, {j})" for i in range(2, norbs) for j in range(2, nelec, 2) if i <= nelec <= norbs]

    # Compute TD-HF
    n_states = 1  # Number of states to compute
    mytd = tdscf.TDHF(mf)
    mytd.nstates = n_states
    mytd.kernel()
    tdhf_energies = mytd.e * 27.2114  # Convert to eV

    # Calculate the error
    errors = [tdhf_energies - energy for energy in energies]  # Difference between TD-HF and CI energies

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(xpoints, errors, marker='o', linestyle='-')
    plt.xlabel('Active Space Size (ncas, nelecas)')
    plt.ylabel('Energy Error (eV)')
    plt.title('Error in CI Energy vs. Active Space Size')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
