import pyscf
from pyscf import gto, scf, tdscf, mcscf, cc, dft
import numpy as np
import scipy
import matplotlib.pyplot as plt

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
mf.get_init_guess(mol, key='minao')
mf.kernel() 

norbs = len(mf.mo_coeff)
nelec = int(mol.nelectron)

e_ciss = -75.63765745
e_cist = -75.67286802

# EOM-CC
mycc = cc.CCSD(mf).run()
e_s, c_s = mycc.eomee_ccsd_singlet(nroots=1)
e_t, c_t = mycc.eomee_ccsd_triplet(nroots=1)


# Calculate the error
errors_eom = (e_ciss - e_s)
errort_eom = (e_ciss - e_t)  # Fixed variable name from e_cist to e_ciss

# TDDFT
mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.kernel()

mytd = tdscf.TDA(mf)
mytd.nstates = 10
mytd.singlet = True
mytd.kernel()
mytd.analyze()

e_dft_s = mytd.e_tot

# Calculate the error
errors_tddft = (e_ciss -  e_dft_s)

mytd = tdscf.TDA(mf)  # Fixed variable name from tddft to tdscf
mytd.nstates = 10
mytd.singlet = False
mytd.kernel()
mytd.analyze()

e_dft_t = mytd.e_tot

# Calculate the error
errort_tddft = (e_cist - e_dft_t)

print('EOM-CC singlet excited state total energy = ', e_s, errors_eom)
print('EOM-CC triplet excited state total energy = ', e_t, errort_eom)
print('TDDFT singlet excited state total energy = ', e_dft_s, errors_tddft)
print('TDDFT triplet excited state total energy = ', e_dft_t, errort_tddft)
