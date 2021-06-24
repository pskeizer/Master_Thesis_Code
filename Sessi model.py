## Sessi model

zero6 = np.zeros((6,6))
zero3 = np.zeros((3,3))

# Pauli matrices and the like
sigma_x = np.array([[0, 1],[1, 0]])
sigma_y = np.array([[0, -1j],[1j, 0]])
sigma_z = np.array([[1, 0],[0, -1]])
zeros = np.zeros((2,2))

# Spin-orbit interaction matrix
soc_1 = np.hstack((zeros,sigma_z,-sigma_y))
soc_2 = np.hstack((-sigma_z, zeros, sigma_x))
soc_3 = np.hstack((sigma_y, -sigma_x, zeros))
soc = np.vstack((soc_1, soc_2, soc_3))

# Hamiltonian
def Sessi(kx, ky, kz, mSn, mTe, tSn, tTe, t, a, lambdaSn, lambdaTe):
    Hos = np.diag([mSn, mSn, mSn, mSn, mSn, mSn, mTe, mTe, mTe, mTe, mTe, mTe])
    Hamnn = 2*t*np.array([
        [math.cos(kx*a), 0, 0, 0, 0, 0],
        [0, math.cos(ky*a), 0, 0, 0, 0],
        [0, 0, 0, 0, 0, math.cos(kz*a)],
        [0, 0, 0, math.cos(kx*a), 0, 0],
        [0, 0, 0, 0, math.cos(ky*a), 0],
        [0, 0, math.cos(kz*a), 0, 0, 0]
    ])
    Hnn = np.vstack((
        np.hstack((zero6, Hamnn)),
        np.hstack((Hamnn, zero6))))
    Hamxynnn = 2*np.array([
        [math.cos(kx*a)*math.cos(ky*a), -math.sin(kx*a)*math.sin(ky*a), 0],
        [-math.sin(kx*a)*math.sin(ky*a), math.cos(kx*a)*math.cos(ky*a), 0],
        [0, 0, 0]
    ])
    Hamznnn = 2*np.array([
        [math.cos(kx*a)*math.cos(kz*a), 0, -math.sin(kx*a)*math.sin(kz*a)],
        [0, math.cos(ky*a)*math.cos(kz*a), -math.sin(ky*a)*math.sin(kz*a)],
        [-math.sin(kx*a)*math.sin(kz*a), -math.sin(ky*a)*math.sin(kz*a), math.cos(kz*a)*(math.cos(kx*a) + math.cos(ky*a))]
    ])
    Hamnnn = np.vstack((
    np.hstack((Hamxynnn, Hamznnn)),
    np.hstack((Hamznnn, Hamxynnn))
    ))
    Hnnn = np.vstack((
    np.hstack((tSn*Hamnnn, zero6)),
    np.hstack((zero6, tTe*Hamnnn))
    ))
    Hxynnn = np.vstack((    
    np.hstack((zero3, tSn*Hamxynnn, zero3, zero3)),
    np.hstack((tSn*Hamxynnn, zero3, zero3, zero3)),
    np.hstack((zero3, zero3, zero3, tTe*Hamxynnn)),
    np.hstack((zero3, zero3, tTe*Hamxynnn, zero3)),
    ))
    H = np.add(np.add(Hnn, Hnnn), Hos)
    ham = np.kron(H, np.eye(2))
    ham_SOC = np.kron([[lambdaSn, 0, 0, 0], [0, lambdaSn, 0, 0],[0, 0, lambdaTe, 0],[0, 0, 0, lambdaTe]], soc)*-1j/2
    return np.add(ham, ham_SOC)

matrixsizeSessi = len(Sessi(0,0,0,0,0,0,0,0,0,0,0)) # Size of the basis

######################################################

G_k = [np.pi/a*np.array([0, 0, 0]), '$\\Gamma$']
X_k = [np.pi/a*np.array([0, 1, 0]), '$X$']
M_k = [np.pi/a*np.array([1, 1, 0]), '$M$']
Z_k = [np.pi/a*np.array([0, 0, 1/2]), '$Z$']
R_k = [np.pi/a*np.array([1, 0, 1/2]), '$R$']
A_k = [np.pi/a*np.array([1, 1, 1/2]), '$A$']

hsp = G_k, X_k, M_k, G_k, Z_k, R_k, A_k, Z_k # Sequence of high symmetry points

mSn = -1.65
mTe = 1.65
tSn = -0.5
tTe = 0.5
t = 0.9
lambdaSn = 0.6
lambdaTe = 0.6
a = 1
stepsize = 200

kx_ar, ky_ar, kz_ar, k_ar, hsp_list = HSP(hsp, stepsize)

kx_array = 1/2*(kx_ar - ky_ar) # Wave vector transformation for basis Sessi
ky_array = 1/2*(kx_ar + ky_ar)
kz_array = kz_ar
k_array = k_ar

#kx_array = kx_ar # Basis Hsieh
#ky_array = ky_ar
#kz_array = kz_ar

energiesSessi = np.zeros((matrixsizeSessi, len(k_ar)))
energiesSessi_SOI = np.zeros((matrixsizeSessi, len(k_ar)))
for i in range(int(len(k_ar))):   
    kx = kx_array[i]
    ky = ky_array[i]
    kz = kz_array[i]
    for j in range(matrixsizeSessi):
        energiesSessi[j][i] = LA.eigvalsh(Sessi(kx, ky, kz, mSn, mTe, tSn, tTe, t, a, 0, 0))[j]
        energiesSessi_SOI[j][i] = LA.eigvalsh(Sessi(kx, ky, kz, mSn, mTe, tSn, tTe, t, a, lambdaSn, lambdaTe))[j] 

#####################################################        
        
minimum = -5
maximum = 5

# Generate the plot
fig = plt.figure(figsize = (8,7))

for i in range(matrixsizeSessi):
    plt.plot(k_ar, energiesSessi[i], 'silver')
    plt.plot(k_ar, energiesSessi_SOI[i], 'red')
plt.ylabel(r"Energy (eV)", fontsize = 15)
plt.xlim(0, max(k_ar))
plt.ylim(minimum, maximum, 0.2)
plt.xticks([])
plt.yticks(np.arange(minimum, maximum + 0.1, 1))
for i in range(len(hsp)):
    hsp_point = np.ones(2)*hsp_list[i]
    plt.plot(hsp_point, [minimum, maximum], linestyle = 'dashed', color = 'black')
    plt.text(hsp_list[i], minimum - 0.5, hsp[i][1], fontsize = 15)
plt.text(1240, 4.50, 'w/o SOI', fontsize = 12, color ='grey', bbox = {'facecolor': 'white','alpha': 0.5})
plt.text(1240, 3.8, 'w/ SOI', fontsize = 12, color ='red', bbox = {'facecolor': 'white','alpha': 0.5})
plt.show()

gap = min(energiesSessi_SOI[12]) - max(energiesSessi_SOI[11])
#print("Band gap:", gap, "eV")