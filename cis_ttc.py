import pyscf
from pyscf import gto, scf, tdscf, lib, tddft

import numpy as np
from numpy import linalg as la
import scipy

mol = gto.Mole()
mol.atom = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161'''
mol.basis = '6-31g'
mol.spin = 0
mol.build()
mf = scf.RHF(mol)
mf.verbose = 4
mf.get_init_guess(mol, key='minao')
mf.kernel()

mo = mf.mo_coeff
mo_energy = mf.mo_energy

nocc = int(mol.nelectron/2)
nvir = mo.shape[1]-nocc

mo_occ = mo_energy[:nocc]
mo_vir = mo_energy[nocc:]

print('nelec =', nocc)
print('norb = ', mo.shape)

#cis A = a + b, B = b; a = delta_ij*delta_ab(e_a - e_i); b = 2-electron integral; iajb
#a

eai = lib.direct_sum('a,i->ai',mo_vir,-mo_occ)
eai = eai.flatten('C')
A_a = np.diag(eai)
print('a = ',A_a.shape)

#b
ao_int2e = mol.intor('int2e')
print('ao_int2e = ', mol.intor('int2e').shape)

temp1 = np.einsum('ip,pqrs->iqrs',mo[:nocc],ao_int2e)
temp2 = np.einsum('aq,iqrs->iars',mo[nocc:],temp1)
temp1 = np.einsum('jr,iars->iajs',mo[:nocc],temp2)
iajb = np.einsum('bs,iajs->iajb',mo[nocc:],temp1)

iajb = np.reshape(iajb,(nocc*nvir,nocc*nvir))
print('iajb = ',iajb.shape)

A = A_a + iajb
AT = A.T
# TDA - cis 
cis_1 = np.block([[A,np.zeros((nocc*nvir,nocc*nvir))], [np.zeros((nocc*nvir,nocc*nvir)), AT]])

# pyscf TDA
'''
mytd = tdscf.TDA(mf)
cis_t1 = mytd.xy

print(cis_t1.shape)

compare = cis - cis_t1

print(compare)
'''

evals,evec = la.eig(cis_1)
w = np.diag(evals)
x = evec

print(x.shape)
print(w.shape)

cis = np.block(([[mo,np.zeros(nocc*nvir*nocc*nvir,nocc*nvir)], [np.zeros(nocc*nvir,nocc*nvir*nocc*nvir),cis_1]]))
