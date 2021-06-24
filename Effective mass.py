# Physical constants
lat = 0.63E-9
mass_el = 9.10938E-31
hbar = 1.054571E-34
eV = 1.602E-19
minimum = -2.2*eV

hsp = W_k, L_k, G_k # Sequence of high symmetry points

W = int(k_array[0])
L = int(k_array[np.where(energiesHsieh[5] == max(energiesHsieh[5]))])
G = int(k_array[-1])

# Region of fitting
fit_reg = np.array([ # VB_WL, VB_LG, CB_WL, VB_LG
    np.array([0.9, 1.0])*len(k_array[W : L]), 
    np.array([0, 0.3])*len(k_array[L : G]) + np.ones(2)*L, 
    np.array([0.9, 1.0])*len(k_array[W : L]), 
    np.array([0, 0.35])*len(k_array[L : G]) + np.ones(2)*L])
fit_reg = fit_reg.astype(int)

kx_array, ky_array, kz_array, k_array, hsp_list = HSP(hsp, stepsize)

# Effective mass function
def effmass(kx_mat, ky_mat, kz_mat, k0, E0, m):
    E_array = np.zeros(kx_mat.shape[0])
    for i in range(len(k_array)):
        kx = kx_mat[0][i][0]
        ky = ky_mat[i][0][0]
        kz = kz_mat[0][0][i]
        E_array[i] = E0 + 1/(2*m)*((kx - k0[0])**2 + (ky - k0[1])**2 + (kz - k0[2])**2)
    return E_array

# Use energies function and fitting parameters from previous code 
energiesHsieh = Eig(gridx, gridy, gridz, fit)

gridx, gridy, gridz = np.meshgrid(kx_array, ky_array, kz_array) # Meshgrid kx, ky, kz

# Approximate around L-point
E0 = [max(energiesHsieh[5]), max(energiesHsieh[5]), min(energiesHsieh[6]), min(energiesHsieh[6])]
k0 = hsp[1][0]

def fun0(m):
    fit = np.square(energiesHsieh[5][fit_reg[0][0] : fit_reg[0][1]] - effmass(gridx, gridy, gridz, k0, E0[0], m)[fit_reg[0][0] : fit_reg[0][1]])
    return fit
def fun1(m):
    fit = np.square(energiesHsieh[5][fit_reg[1][0] : fit_reg[1][1]] - effmass(gridx, gridy, gridz, k0, E0[1], m)[fit_reg[1][0] : fit_reg[1][1]])
    return fit
def fun2(m):
    fit = np.square(energiesHsieh[6][fit_reg[2][0] : fit_reg[2][1]] - effmass(gridx, gridy, gridz, k0, E0[2], m)[fit_reg[2][0] : fit_reg[2][1]])
    return fit
def fun3(m):
    fit = np.square(energiesHsieh[6][fit_reg[3][0] : fit_reg[3][1]] - effmass(gridx, gridy, gridz, k0, E0[3], m)[fit_reg[3][0] : fit_reg[3][1]])
    return fit

# Initial guess for masses
m_trial = [-1, -1, 1, 1]

# Least square function
res0 = least_squares(fun0, m_trial[0], max_nfev = 100) 
res1 = least_squares(fun1, m_trial[1], max_nfev = 100) 
res2 = least_squares(fun2, m_trial[2], max_nfev = 100) 
res3 = least_squares(fun3, m_trial[3], max_nfev = 100) 

massH = np.array([res0.x[0], res1.x[0], res2.x[0], res3.x[0]])
#print(massH)


# Plotting
_list = [-2, 2]

fig = plt.figure(figsize = (7,7))

band = 0

plt.plot(k_array, energiesHsieh[5], color = 'red')
plt.plot(k_array, energiesHsieh[6], color = 'red')
plt.plot(k_array[W : L], effmass(gridx, gridy, gridz, k0, E0[0], massH[0])[W : L], color = 'deepskyblue', linestyle = 'dashed')
plt.plot(k_array[L : G], effmass(gridx, gridy, gridz, k0, E0[1], massH[1])[L : G], color = 'deepskyblue', linestyle = 'dashed')
plt.plot(k_array[W : L], effmass(gridx, gridy, gridz, k0, E0[2], massH[2])[W : L], color = 'deepskyblue', linestyle = 'dashed')
plt.plot(k_array[L : G], effmass(gridx, gridy, gridz, k0, E0[3], massH[3])[L : G], color = 'deepskyblue', linestyle = 'dashed')
plt.ylim(_list[0], _list[1])
plt.ylabel(r"Energy (eV)", fontsize = 15)
plt.xlim(k_array[0], k_array[-1])
plt.xticks([])
plt.text(W, _list[0] - 0.2, hsp[0][1], fontsize = 15)
plt.text(L, _list[0] - 0.2, hsp[1][1], fontsize = 15)
plt.text(G, _list[0] - 0.2, hsp[2][1], fontsize = 15)
plt.plot(L*np.ones(2), _list, color = 'black', linestyle = 'dashed')

massH_list = massH*(hbar**2)/mass_el/(lat**2)/eV
xy = [[110, -0.8], [350, -1.5], [100, 1.4], [400, 1.3]]
for i in range(len(xy)):
    plt.annotate( '$%.3f \; m_e$' % (massH_list[i]), (xy[i][0], xy[i][1]), fontsize = 12, bbox = dict(boxstyle = 'round', fc = 'yellow', alpha = 0.3))
plt.show()

print("\r Hsieh effective mass conduction band along W-L:", massH_list[2], "m\u2091, Literature says: +",0.03, "m\u2091")
print("\r Hsieh effective mass conduction band along L-\u0393:", massH_list[3], "m\u2091, Literature says: +",0.125, "m\u2091")