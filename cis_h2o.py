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

#cis A = a + b, B = b; a = delta_ij*delta_ab(e_a - e_i); b = 2-electron integral; iajb
#a

eai = lib.direct_sum('a,i->ai',mo_vir,-mo_occ)
eai = eai.flatten('C')
A_a = np.diag(eai)

#b
ao_int2e = mol.intor('int2e')
print('ao_int2e = ', mol.intor('int2e').shape)

temp1 = np.einsum('ip,pqrs->iqrs',mo[:nocc],ao_int2e)
temp2 = np.einsum('aq,iqrs->iars',mo[nocc:],temp1)
temp1 = np.einsum('jr,iars->iajs',mo[:nocc],temp2)
iajb = np.einsum('bs,iajs->iajb',mo[nocc:],temp1)
iajb = np.reshape(iajb,(nocc*nvir,nocc*nvir))


A = A_a + iajb

evals,evec = la.eig(A)

w = np.diag(evals)
x = evec

print('shape mo_energy', mo.shape)
print('shape mo', mo.shape)
print('A_a = ', A_a.shape)
print('iajb',iajb.shape)
print('eigenvalues',w.shape, w)

eigenvalues = scipy.sparse.linalg.eigs(w, k=1, which='SM')
print('excitation energy', eigenvalues)

#total energy of excited states
c_ia = np.einsum('i,a->ia', mo[nocc:], mo[:nocc])
c_iajb = np.einsum('ia,jb->iajb', c_ia,c_ia)

print('shape c_ia,c_iajb', c_ia, c_iajb)
E_cis = mo_energy + c_ia*c_ia*A_a + c_iajb * iajb
print('E_cis =', E_cis )

# pyscf TDA
mytd = tdscf.TDA(mf)
mytd.verbose = 4
mytd.run()
print('TDA CIS total energy = ', mytd.e_tot)
