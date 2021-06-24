# # Hsieh model

import numpy as np
import qsymm
import math
import sympy
from IPython.display import display
from IPython.display import Math
from scipy.linalg import block_diag
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import kwant.builder
import scipy.sparse.linalg as sla
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.optimize import least_squares
get_ipython().run_line_magic('matplotlib', 'inline')

# Pauli matrices and the like
sigma_x = np.array([[0, 1],[1, 0]])
sigma_y = np.array([[0, -1j],[1j, 0]])
sigma_z = np.array([[1, 0],[0, -1]])
zero2 = np.zeros((2,2))
zero3 = np.zeros((3,3))

# Spin-orbit interaction matrix
soc_1 = np.hstack((zero2,sigma_z,-sigma_y))
soc_2 = np.hstack((-sigma_z, zero2, sigma_x))
soc_3 = np.hstack((sigma_y, -sigma_x, zero2))
soc = np.vstack((soc_1, soc_2, soc_3))

# Hamiltonian with spin-orbit interaction
def Hsieh(kx, ky, kz, mSn, mTe, tSn, tTe, t, tpi, tSnpi, tTepi, a, lambdaSn, lambdaTe):
    Hos = np.diag([mSn, mSn, mSn, mTe, mTe, mTe])
    
    Hamnn = 2*t*np.array([
        [math.cos(kx*a), 0, 0],
        [0, math.cos(ky*a), 0],
        [0, 0, math.cos(kz*a)]
    ])
    Hnn = np.vstack((
        np.hstack((zero3, Hamnn)),
        np.hstack((Hamnn, zero3))))
    Hamnnpi = -2*tpi*np.array([
        [math.cos(ky*a) + math.cos(kz*a), 0, 0],
        [0, math.cos(kx*a) + math.cos(kz*a), 0],
        [0, 0, math.cos(kx*a) + math.cos(ky*a)]
    ])
    Hnnpi = np.vstack((
        np.hstack((zero3, Hamnnpi)),
        np.hstack((Hamnnpi, zero3))))
    Hamxynnn = 2*np.array([
        [math.cos(kx*a)*math.cos(ky*a), -math.sin(kx*a)*math.sin(ky*a), 0],
        [-math.sin(kx*a)*math.sin(ky*a), math.cos(kx*a)*math.cos(ky*a), 0],
        [0, 0, 0]
    ])
    Hxynnn = np.vstack((
        np.hstack((tSn*Hamxynnn, zero3)),
        np.hstack((zero3, tTe*Hamxynnn))))
    Hamxznnn = 2*np.array([
        [math.cos(kx*a)*math.cos(kz*a), 0, -math.sin(kx*a)*math.sin(kz*a)],
        [0, 0, 0],
        [-math.sin(kx*a)*math.sin(kz*a), 0, math.cos(kx*a)*math.cos(kz*a)]
    ])
    Hxznnn = np.vstack((
        np.hstack((tSn*Hamxznnn, zero3)),
        np.hstack((zero3, tTe*Hamxznnn))))
    Hamyznnn = 2*np.array([
        [0, 0, 0],
        [0, math.cos(ky*a)*math.cos(kz*a), -math.sin(ky*a)*math.sin(kz*a)],
        [0, -math.sin(ky*a)*math.sin(kz*a), math.cos(ky*a)*math.cos(kz*a)]
    ])
    Hyznnn = np.vstack((
        np.hstack((tSn*Hamyznnn, zero3)),
        np.hstack((zero3, tTe*Hamyznnn))))
    Hamnnnpi = -2*np.array([
        [math.cos(ky*a)*math.cos(kz*a), 0, 0],
        [0, math.cos(kx*a)*math.cos(kz*a), 0],
        [0, 0, math.cos(kx*a)*math.cos(ky*a)]        
    ])
    Hnnnpi = np.vstack((
        np.hstack((tSnpi*Hamnnnpi, zero3)),
        np.hstack((zero3, tTepi*Hamnnnpi))))
    H = np.add(np.add(np.add(np.add(np.add(np.add(Hnn, Hxynnn), Hyznnn), Hxznnn), Hos), Hnnpi), Hnnnpi)
    ham = np.kron(H, np.eye(2))
    ham_SOC = np.kron([[lambdaSn, 0], [0, lambdaTe]], soc)*-1j/2 # We believe that it should be -1j, is what we observe in literature
    return np.add(ham, ham_SOC)

matrixsizeHsieh = len(Hsieh(0,0,0,0,0,0,0,0,0,0,0,0,0,0)) # Size of the basis

#########################################################

a = 1 # Normalized lattice spacing
stepsize = 50

def HSP(hsp, stepsize):
    
    kx_array = []
    ky_array = []
    kz_array = []
    
    norm = 0
    hsp_k = 0
    hsp_list = [0]
    
    for i in range(len(hsp) - 1):
        norm = np.linalg.norm(hsp[1][0] - hsp[0][0])
        norm_new = np.linalg.norm(hsp[i + 1][0] - hsp[i][0])
        if i == len(hsp) - 2:
            endpoint = True # Take into account k on last high symmetry point
        else:
            endpoint = False
        kx_new = np.linspace(hsp[i][0][0], hsp[i + 1][0][0], int(stepsize*norm_new/norm), endpoint = endpoint)
        ky_new = np.linspace(hsp[i][0][1], hsp[i + 1][0][1], int(stepsize*norm_new/norm), endpoint = endpoint)
        kz_new = np.linspace(hsp[i][0][2], hsp[i + 1][0][2], int(stepsize*norm_new/norm), endpoint = endpoint)
        kx_array = np.hstack([kx_array, kx_new])
        ky_array = np.hstack([ky_array, ky_new])
        kz_array = np.hstack([kz_array, kz_new])

        hsp_k_new = len(kx_new)
        hsp_k += hsp_k_new
        hsp_list.append(hsp_k)
        
        k_array = np.linspace(0, len(kx_array) - 1, len(kx_array))
    return kx_array, ky_array, kz_array, k_array, hsp_list

#######################################################

G_k = [np.pi/a*np.array([0, 0, 0]), '$\Gamma$']
X_k = [np.pi/a*np.array([1, 0, 0]), '$X$']
W_k = [np.pi/a*np.array([1, 1/2, 0]), '$W$']
L_k = [np.pi/a*np.array([1/2, 1/2, 1/2]), '$L$']
K_k = [np.pi/a*np.array([3/4, 3/4, 0]), '$K$']

hsp = G_k, X_k, W_k, L_k, G_k, K_k # Sequence of high symmetry points

mSn = -1.65
mTe = 1.65
tSn = -0.5
tTe = 0.5
t = 0.9
tpi = 0
tSnpi = 0
tTepi = 0
t3 = 0
a = 1
lambdaSn = 0.6
lambdaTe = 0.6
fit_Hsieh = mSn, mTe, tSn, tTe, t, lambdaSn, lambdaTe, tpi, tSnpi, tTepi

kx_array, ky_array, kz_array, k_array, hsp_list = HSP(hsp, stepsize)

energiesHsieh = np.zeros((matrixsizeHsieh, len(k_array)))
energiesHsieh_SOC = np.zeros((matrixsizeHsieh, len(k_array)))
for i in range(int(len(k_array))):   
    kx = kx_array[i]
    ky = ky_array[i]
    kz = kz_array[i]
    for j in range(matrixsizeHsieh):
        energiesHsieh[j][i] = LA.eigvalsh(Hsieh(kx, ky, kz, mSn, mTe, tSn, tTe, t, tpi, tSnpi, tTepi, a, 0, 0))[j]
        energiesHsieh_SOC[j][i] = LA.eigvalsh(Hsieh(kx, ky, kz, mSn, mTe, tSn, tTe, t, tpi, tSnpi, tTepi, a, lambdaSn, lambdaTe))[j]
        
###########################################

minimum = -5
maximum = 5

# Generate the plot
fig = plt.figure(figsize = (8,7))

for i in range(matrixsizeHsieh):
    plt.plot(k_array, energiesHsieh[i], 'silver')
    plt.plot(k_array, energiesHsieh_SOC[i], 'red')
plt.ylabel(r"Energy (eV)", fontsize = 15)
plt.xlim(0, max(k_array))
plt.ylim(minimum, maximum, 0.2)
plt.xticks([])
plt.yticks(np.arange(minimum, maximum + 0.1, 1))
for i in range(len(hsp)):
    hsp_point = np.ones(2)*hsp_list[i]
    plt.plot(hsp_point, [minimum, maximum], linestyle = 'dashed', color = 'black')
    plt.text(hsp_list[i], 1.1*minimum, hsp[i][1], fontsize = 15)
plt.show()

gap = min(energiesHsieh[6]) - max(energiesHsieh[5])
#print("Band gap:", gap, "eV")