import pyscf
from pyscf import gto, scf, tdscf, lib, tddft
import numpy as np
from numpy import linalg as la
import scipy

mol = gto.Mole()
mol.atom = '''
O          0.00000        0.00000        0.11779
H          0.00000        0.75545       -0.47116
H          0.00000       -0.75545       -0.47116'''
mol.basis = '6-31g'
mol.spin = 0
mol.build()

mf = scf.RHF(mol)
mf.verbose = 4
mf.get_init_guess(mol, key='minao')
mf.run()

mo = mf.mo_coeff
mo_energy = mf.mo_energy

nocc = int(mol.nelectron/2)
nvir = mo.shape[1]-nocc

mo_occ = mo_energy[nocc:]
mo_vir = mo_energy[:nocc]

print('nelec =', nocc)
print('norb = ', len(mo))

#number of electrons from density matrix 



#average density matrix


