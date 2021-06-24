mSn = -1.65
mTe = 1.65
t = 0.9
tSn = -0.5
tTe = 0.5
lambdaSn = 0.6
lambdaTe = 0.6

a = 1 # Normalized lattice spacing
stepsize = 100

G_k = [np.pi/a*np.array([0, 0, 0]), '$\Gamma_2$']
X_k = [np.pi/a*np.array([1, 0, 0]), '$X_2$']
W_k = [np.pi/a*np.array([1, 1/2, 0]), '$W_2$']
L_k = [np.pi/a*np.array([1/2, 1/2, 1/2]), '$L_2$']
L_k2 = [np.pi/a*np.array([3/2, 1/2, 1/2]), '$L_2\'$']
K_k = [np.pi/a*np.array([3/4, 3/4, 0]), '$K_2$']
X_k2 = [np.pi/a*np.array([1, 1, 0]), '$X_2\'$']

#hsp = G_k, X_k, W_k, L_k, G_k, K_k # Total k-path
hsp = G_k, K_k, X_k2, K_k, G_k # Band_folding_1
#hsp = G_k, X_k, W_k, X_k2, W_k, X_k, G_k # Band_folding_2
#hsp = G_k, L_k, W_k, L_k2, X_k, L_k2, W_k, L_k, G_k # Band_folding_3

kx_array, ky_array, kz_array, k_array, hsp_list = HSP(hsp, stepsize)

energiesHsieh = np.zeros((matrixsizeHsieh, len(k_array)))
for i in range(int(len(k_array))):   
    kx = kx_array[i]
    ky = ky_array[i]
    kz = kz_array[i]
    for j in range(matrixsizeHsieh):
        energiesHsieh[j][i] = LA.eigvalsh(Hsieh(kx, ky, kz, mSn, mTe, tSn, tTe, t, 0, 0, 0, a, lambdaSn, lambdaTe))[j]
        
#########################################

minimum = -5
maximum = 5

# Generate the plot
fig = plt.figure(figsize = (8,6))

for i in range(matrixsizeHsieh):
    plt.plot(k_array, energiesHsieh[i], 'red')
plt.ylabel(r"Energy (eV)", fontsize = 15)
plt.xlim(0, max(k_array))
plt.ylim(minimum, maximum, 0.2)
plt.xticks([])
plt.yticks(np.arange(minimum, maximum + 0.1, 1))
for i in range(len(hsp)):
    hsp_point = np.ones(2)*hsp_list[i]
    plt.plot(hsp_point, [minimum, maximum], linestyle = 'dashed', color = 'black')
    plt.text(hsp_list[i], minimum - 0.5, hsp[i][1], fontsize = 15)
plt.show()

#################################################### Sessi

a = 1
stepsize = 100

# HSP of Sessi BZ

G_k = [np.pi/a*np.array([0, 0, 0]), '$\\Gamma_4$']
G_k2= [np.pi/a*np.array([2, 0, 0]), '$\\Gamma_4\'$']
X_k = [np.pi/a*np.array([1, 0, 0]), '$X_4$']
M_k = [np.pi/a*np.array([1, 1, 0]), '$M_4$']
R_k = [np.pi/a*np.array([1, 0, 1/2]), '$R_4$']
R_k2 = [np.pi/a*np.array([2, -1, 1/2]), '$R_4\'$']
W_k = [np.pi/a*np.array([3/2, -1/2, 0]), '$W_2$']

hsp = G_k, X_k, G_k2, X_k, G_k # Band_folding_1
#hsp = G_k, M_k, G_k2, M_k, G_k # Band_folding_2
#hsp = G_k, R_k, W_k, R_k2, G_k2, R_k2, W_k, R_k, G_k # Band_folding_3

kx_ar, ky_ar, kz_ar, k_ar, hsp_list = HSP(hsp, stepsize)

kx_array = 1/2*(kx_ar - ky_ar)  # Wave vector transformation for basis Sessi
ky_array = 1/2*(kx_ar + ky_ar)
kz_array = kz_ar

#######################################################################

# HSP of Hsieh BZ

#G_k = [np.pi/a*np.array([0, 0, 0]), '$\Gamma_2$']
#X_k = [np.pi/a*np.array([1, 0, 0]), '$X_2$']
#W_k = [np.pi/a*np.array([1, 1/2, 0]), '$W_2$']
#L_k = [np.pi/a*np.array([1/2, 1/2, 1/2]), '$L_2$']
#K_k = [np.pi/a*np.array([3/4, 3/4, 0]), '$K_2$']


#hsp = G_k, X_k, W_k, L_k, G_k, K_k # Sequence of high symmetry points

#kx_ar, ky_ar, kz_ar, k_ar, hsp_list = HSP(hsp, stepsize)

#kx_array = kx_ar # For basis Hsieh
#ky_array = ky_ar 
#kz_array = kz_ar

#######################################################################

k_array = k_ar

energiesSessi = np.zeros((matrixsizeSessi, len(k_ar)))
for i in range(int(len(k_ar))):   
    kx = kx_array[i]
    ky = ky_array[i]
    kz = kz_array[i]
    for j in range(matrixsizeSessi):
        energiesSessi[j][i] = LA.eigvalsh(Sessi(kx, ky, kz, mSn, mTe, tSn, tTe, t, a, lambdaSn, lambdaTe))[j]
        
###############################################

minimum = -5
maximum = 5

# Generate the plot
fig = plt.figure(figsize = (8,6))

for i in range(matrixsizeSessi):
    plt.plot(k_array, energiesSessi[i], 'red')
plt.ylabel(r"Energy (eV)", fontsize = 15)
plt.xlim(0, max(k_array))
plt.ylim(minimum, maximum, 0.2)
plt.xticks([])
plt.yticks(np.arange(minimum, maximum + 0.1, 1))
for i in range(len(hsp)):
    hsp_point = np.ones(2)*hsp_list[i]
    plt.plot(hsp_point, [minimum, maximum], linestyle = 'dashed', color = 'black')
    plt.text(hsp_list[i], minimum - 0.5, hsp[i][1], fontsize = 15)
plt.show()