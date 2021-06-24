# Depending on how many parameters we wish to incorporate into the fit, one can comment lines out.
#As of now, these are set to all parameters but the pi-hopping

# HSP
G_k = [np.pi/a*np.array([0, 0, 0]), '$\Gamma$']
X_k = [np.pi/a*np.array([1, 0, 0]), '$X$']
W_k = [np.pi/a*np.array([1, 1/2, 0]), '$W$']
L_k = [np.pi/a*np.array([1/2, 1/2, 1/2]), '$L$']
K_k = [np.pi/a*np.array([3/4, 3/4, 0]), '$K$']

hsp = G_k, X_k, W_k, L_k, G_k, K_k # Sequence of high symmetry points

kx_array, ky_array, kz_array, k_array, hps_list = HSP(hsp, stepsize)

# Initial guess
mSn = 1.562470155
mTe = 1.292499124
tSn = 0.42497536
tTe = 0.3001542240086133
ts = 0.5
t = 0.4541455503407132
tpi = 0 #0.276174899
tSnpi = 0 #-0.554268105
tTepi = 0 #-0.201321961 
t3 = 0
a = 1
lambdaSn = 0.5947135806218169
lambdaTe = 0.36404025057409817
#fit_Pim = mSn, mTe, tSn, tTe, t, lambdaSn, lambdaTe, tpi, tSnpi, tTepi # Initial guess with pi-hopping
fit_Pim = mSn, mTe, tSn, tTe, t, lambdaSn, lambdaTe # Initial guess

def Eig(kx_mat, ky_mat, kz_mat, param):
    energyHsieh = np.zeros((matrixsizeHsieh, kx_mat.shape[0]))      
    for i in range(len(kx_array)):
        kx = kx_mat[0][i][0]
        ky = ky_mat[i][0][0]
        kz = kz_mat[0][0][i]
        for j in range(matrixsizeHsieh):
            #energyHsieh[j][i] = LA.eigvalsh(Hsieh(kx, ky, kz, param[0], param[1], param[2], param[3], param[4], param[7], param[8], param[9], 0, 1, param[5], param[6]))[j]
            energyHsieh[j][i] = LA.eigvalsh(Hsieh(kx, ky, kz, param[0], param[1], param[2], param[3], param[4], 1, param[5], param[6]))[j]
    return energyHsieh # The band from Hsieh to be compared with Lent

# Lower and upper bound fitting. These are determined with the procedure in section 4.3.2
tSn_low, tSn_high = 0.3, 0.7
tTe_low, tTe_high = 0.3, 0.7
lambdaSn_low, lambdaSn_high = 0.1, 0.7
lambdaTe_low, lambdaTe_high = 0.1, 0.7
t_low, t_high = 0.1, 2.5
mSn_low, mSn_high = 1, 2
mTe_low = 2.03*tSn_low + 2.27*tTe_low - np.abs(mSn_low)
mTe_high = 3.467*tSn_high + 3.4725*tTe_high - np.abs(mSn_high)
tpi_low, tpi_high = -1, 1
tSnpi_low, tSnpi_high = -1, 1
tTepi_low, tTepi_high = -1, 1

underbound = (mSn_low, mTe_low, tSn_low, tTe_low, t_low, lambdaSn_low, lambdaTe_low)
upperbound = (mSn_high, mTe_high, tSn_high, tTe_high, t_high, lambdaSn_high, lambdaTe_high)

#underbound = (mSn_low, mTe_low, tSn_low, tTe_low, t_low, lambdaSn_low, lambdaTe_low, tpi_low, tSnpi_low, tTepi_low)
#upperbound = (mSn_high, mTe_high, tSn_high, tTe_high, t_high, lambdaSn_high, lambdaTe_high, tpi_high, tSnpi_high, tTepi_high)

kx_ar = kx_array
ky_ar = ky_array
kz_ar = kz_array

gridx, gridy, gridz = np.meshgrid(kx_ar, ky_ar, kz_ar)

# Fermi level w.r.t Lent model
Fermi = (max(energiesLent[9]) + min(energiesLent[10]))/2 

# Weight w.r.t to distance from the Fermi level. 
w = 10 # scalar weight
weight = np.zeros((int(len(energiesHsieh)/2), len(k_array)))
for i in range(int(len(energiesHsieh)/4)):
    weight[i] = np.add(np.ones(len(k_array))*Fermi,-energiesLent[2*i + 4])
    weight[i + 3] = np.add(-np.zeros(len(k_array))*Fermi, energiesLent[2*i + 10])
    
W = hsp_list[2]
L = hsp_list[3]
G = hsp_list[4]

# Weight around the L-point
weightL = np.ones(len(k_array))
weightstartL = int(W + (L-W)/2)
weightstopL = int(L + (G - L)/2)

for i in range(weightstartL, weightstopL):
    weightL[i] = 1/w

w = 10 # scalar weight

# Weight between -0.5 and +0.5 eV
array = np.ones((int(len(energiesHsieh)/2), len(k_array)))
for i in range(int(len(energiesHsieh)/4)):
    for j in range(len(k_array)):
        if energiesLent[2*i + 4][j] > -0.5:
            array[i][j] = 1/(10*w)
        if energiesLent[2*i + 10][j] < 0.5:
            array[i + 3][j] = 1/(10*w)

def fun(param):
    red = np.zeros(len(k_array))
    for i in range(int(len(energiesHsieh)/2)): 
        red_ = np.divide(np.square(Eig(gridx, gridy, gridz, param)[2*i] - energiesLent[2*i + 4]), w*np.multiply(weight[i],array[i]))
        red = np.add(red_, red)
    return np.divide(red, weightL)

res = least_squares(fun, fit_Pim, bounds = (underbound, upperbound), max_nfev = 200)
fit = res.x
#print("mSn = ", fit[0], "mTe =", fit[1], "tSn =", fit[2], "tTe =", fit[3], "t =", fit[4], "t\u03C0 =", fit[7], "t\u03C0Sn =", fit[8], "t\u03C0Te =", fit[9], "\u03BBSn =", fit[5], "\u03BBTe =", fit[6])
print("mSn = ", fit[0], "mTe =", fit[1], "tSn =", fit[2], "tTe =", fit[3], "t =", fit[4], "\u03BBSn =", fit[5], "\u03BBTe =", fit[6])

# Find the eigenvalues from these fitting parameters

energies = Eig(gridx, gridy, gridz, fit)

# Plotting
minimum = -1
maximum = 1

fig = plt.figure(figsize = (7,7))
for i in range(matrixsizeLent):
    plt.plot(k_array, energiesLent[i], 'black', label = "Lent")
for i in range(matrixsizeHsieh):
    plt.plot(k_array, energies[i], 'red')
plt.ylabel(r"Energy (eV)", fontsize = 15)
plt.xlim(0, max(k_array))
plt.ylim(np.min(energiesLent) - 1,np.max(energiesLent) + 1)
plt.ylim(minimum, maximum, 0.2)
plt.xticks([])
plt.yticks(np.arange(minimum, maximum + 0.1, 0.2))
for i in range(len(hsp)):
    hsp_point = np.ones(2)*hsp_list[i]
    plt.plot(hsp_point, [minimum, maximum], linestyle = 'dashed', color = 'black')
    plt.text(hsp_list[i], minimum - 0.1*maximum, hsp[i][1], fontsize = 15)
plt.show()

gap = min(energiesHsieh[6]) - max(energiesHsieh[5])
print("Band gap:", gap, "eV")