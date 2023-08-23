import pyscf
from pyscf import gto, scf, tdscf
import numpy as np
from numpy import linalg as la
import scipy

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

print(mo_occ.shape)
print(mo_vir.shape)

S = mf.get_ovlp()
F = mf.get_fock()
#cis A = a + b, B = bl; a = delta_ij*delta_ab(e_a - e_i); 
# b = 2-electron integral; iajb
#a
first = np.zeros((nocc * nvir, nocc * nvir))
for k in range(nvir, nocc):
    for i in range(mo_occ):
        for a in range(mo_vir):
            first[k * mp_occ + i, k * mo_vir + a] = mo_vir[a] - mo_occ[i]
A = first

#b
ao_int2e = mol.intor('int2e')
temp1 = np.einsum('pi,pqrs->iqrs',mo,ao_int2e)
temp2 = np.einsum('qa,iqrs->iars',mo,temp1)
temp1 = np.einsum('rj,iars->iajs',mo,temp2)
iajb = np.einsum('sb,iajs->iajb',mo,temp1)
b = reshape(iajb,(nocc*nvir,nocc*nvir))

print(a.shape)
print(iajb.shape)

#CIS energy: 
A = a[:nocc] + b[nocc:]

w = la.eigvlas(A)

print(w)

#compare with td_cis; pyscf
mf2 = td._scf
tdmo = mf2.mo_energy







