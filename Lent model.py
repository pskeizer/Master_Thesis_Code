## Lent model

def H(self): # Easier way to compute Hermitian conjugate
    return self.conj().T

def Lent(esc, esa, epc, epa, edc, eda, lambdac, lambdaa, vss, vsp, vps, vpp, vpppi, vpd, vpdpi, vdp, vdppi, vdd, vdddelta, vddpi, kx, ky, kz, a):
    
    g0 = 2*(math.cos(kx*a) + math.cos(ky*a) + math.cos(kz*a))
    g1 = 1j*math.sin(kx*a)
    g2 = 1j*math.sin(ky*a)
    g3 = 1j*math.sin(kz*a)
    g4 = math.cos(kx*a)
    g5 = math.cos(ky*a)
    g6 = math.cos(kz*a)
    
    vxx = 2*g4*vpp + 2*(g5 + g6)*vpppi
    vyy = 2*g5*vpp + 2*(g4 + g6)*vpppi
    vzz = 2*g6*vpp + 2*(g4 + g5)*vpppi
    
    Hss  = np.array([[esc, 0, g0*vss, 0],
                     [0, esc, 0, g0*vss],
                     [g0*vss, 0, esa, 0],
                     [0, g0*vss, 0, esa]])
    Hpcs = np.array([[0, 0, -2*g1*vps, 0],
                     [0, 0, -2*g2*vps, 0],
                     [0, 0, -2*g3*vps, 0],
                     [0, 0, 0, -2*g1*vps],
                     [0, 0, 0, -2*g2*vps],
                     [0, 0, 0, -2*g3*vps]])
    Hpas = np.array([[-2*g1*vsp, 0, 0, 0],
                     [-2*g2*vsp, 0, 0, 0],
                     [-2*g3*vsp, 0, 0, 0],
                     [0, -2*g1*vsp, 0, 0],
                     [0, -2*g2*vsp, 0, 0],
                     [0, -2*g3*vsp, 0, 0]])
    Hpcpc = np.array([[epc, -1j*lambdac/2, 0, 0, 0, lambdac/2],
                     [1j*lambdac/2, epc, 0, 0, 0, -1j*lambdac/2],
                     [0, 0, epc, -lambdac/2, 1j*lambdac/2, 0],
                     [0, 0, -lambdac/2, epc, 1j*lambdac/2, 0],
                     [0, 0, -1j*lambdac/2, -1j*lambdac/2, epc, 0],
                     [lambdac/2, 1j*lambdac/2, 0, 0, 0, epc]])
    Hpapa = np.array([[epa, -1j*lambdaa/2, 0, 0, 0, lambdaa/2],
                     [1j*lambdaa/2, epa, 0, 0, 0, -1j*lambdaa/2],
                     [0, 0, epa, -lambdaa/2, 1j*lambdaa/2, 0],
                     [0, 0, -lambdaa/2, epa, 1j*lambdaa/2, 0],
                     [0, 0, -1j*lambdaa/2, -1j*lambdaa/2, epa, 0],
                     [lambdaa/2, 1j*lambdaa/2, 0, 0, 0, epa]])
    Hpapc = np.array([[vxx, 0, 0, 0, 0, 0],
                      [0, vyy, 0, 0, 0, 0],
                      [0, 0, vzz, 0, 0, 0],
                      [0, 0, 0, vxx, 0, 0],
                      [0, 0, 0, 0, vyy, 0],
                      [0, 0, 0, 0, 0, vzz]])
    Hdapc = np.array([[-np.sqrt(3)*g1*vpd, np.sqrt(3)*g2*vpd, 0, 0, 0, 0],
                      [g1*vpd, g2*vpd, -2*g3*vpd, 0, 0, 0],
                      [-2*g2*vpdpi, -2*g1*vpdpi, 0, 0, 0, 0],
                      [0, -2*g3*vpdpi, -2*g2*vpdpi, 0, 0, 0],
                      [-2*g3*vpdpi, 0, -2*g1*vpdpi, 0, 0, 0],
                      [0, 0, 0, -np.sqrt(3)*g1*vpd, np.sqrt(3)*g2*vpd, 0],
                      [0, 0, 0, g1*vpd, g2*vpd, -2*g3*vpd],
                      [0, 0, 0, -2*g2*vpdpi, -2*g1*vpdpi, 0],
                      [0, 0, 0, 0, -2*g3*vpdpi, -2*g2*vpdpi],
                      [0, 0, 0, -2*g3*vpdpi, 0, -2*g1*vpdpi]])
    Hdcpa = np.array([[-np.sqrt(3)*g1*vdp, np.sqrt(3)*g2*vdp, 0, 0, 0, 0],
                      [g1*vdp, g2*vdp, -2*g3*vdp, 0, 0, 0],
                      [-2*g2*vdppi, -2*g1*vdppi, 0, 0, 0, 0],
                      [0, -2*g3*vdppi, -2*g2*vdppi, 0, 0, 0],
                      [-2*g3*vdppi, 0, -2*g1*vdppi, 0, 0, 0],
                      [0, 0, 0, -np.sqrt(3)*g1*vdp, np.sqrt(3)*g2*vdp, 0],
                      [0, 0, 0, g1*vdp, g2*vdp, -2*g3*vdp],
                      [0, 0, 0, -2*g2*vdppi, -2*g1*vdppi, 0],
                      [0, 0, 0, 0, -2*g3*vdppi, -2*g2*vdppi],
                      [0, 0, 0, -2*g3*vdppi, 0, -2*g1*vdppi]])
    Hdadc = np.array([[3/2*(g4 + g5)*vdd + (2*g6 + g4/2 + g5/2)*vdddelta, np.sqrt(3)/2*(g5 - g4)*(vdd - vdddelta), 0, 0, 0, 0, 0, 0, 0, 0],
                      [np.sqrt(3)/2*(g5 - g4)*(vdd - vdddelta), 3/2*(g4 + g5)*vdddelta + (2*g6 + g4/2 + g5/2)*vdd, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 2*(g4 + g5)*vddpi + 2*g6*vdddelta, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 2*(g5 + g6)*vddpi + 2*g4*vdddelta, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 2*(g4 + g6)*vddpi + 2*g5*vdddelta, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 3/2*(g4 + g5)*vdd + (2*g6 + g4/2 + g5/2)*vdddelta, np.sqrt(3)/2*(g5 - g4)*(vdd - vdddelta), 0, 0, 0],
                      [0, 0, 0, 0, 0, np.sqrt(3)/2*(g5 - g4)*(vdd - vdddelta), 3/2*(g4 + g5)*vdddelta + (2*g6 + g4/2 + g5/2)*vdd, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 2*(g4 + g5)*vddpi + 2*g6*vdddelta, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 2*(g5 + g6)*vddpi + 2*g4*vdddelta, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 2*(g4 + g6)*vddpi + 2*g5*vdddelta]])
    Hdcdc = np.diag([edc, edc, edc, edc, edc, edc, edc, edc, edc, edc])
    Hdada = np.diag([eda, eda, eda, eda, eda, eda, eda, eda, eda, eda])
    
    ham = np.vstack((
        np.hstack((Hss, H(Hpcs), H(Hpas), np.zeros((4, 10)), np.zeros((4,10)))),
        np.hstack((Hpcs, Hpcpc, H(Hpapc), np.zeros((6, 10)), H(Hdapc))),
        np.hstack((Hpas, Hpapc, Hpapa, H(Hdcpa), np.zeros((6, 10)))),
        np.hstack((np.zeros((10, 4)), np.zeros((10, 6)), Hdcpa, Hdcdc, H(Hdadc))),
        np.hstack((np.zeros((10, 4)), Hdapc, np.zeros((10, 6)), Hdadc, Hdada))
    ))
    return ham

esc = -6.578
esa = -12.067
epc = 1.659
epa = -0.167
edc = 8.38
eda = 7.73
lambdac = 0.592
lambdaa = 0.564
vss = -0.510
vsp = 0.949
vps = -0.198
vpp = 2.218
vpppi = -0.446
vpd = -1.11
vpdpi = 0.624
vdp = -1.67
vdppi = 0.766
vdd = -1.772
vdddelta = 0.618
vddpi = 0
a = 1

matrixsizeLent = len(Lent(esc, -12.067, 1.659, -0.167, 8.38, 7.73, 0.592, 0.564, -0.510, 0.949, -0.198, 2.218, -0.446, -1.11, 0.624, -1.67, -0.766, -1.772, -0.618, 0, np.pi/2, np.pi, np.pi/2, 1))

############################################################

hsp = G_k, X_k, W_k, L_k, G_k, K_k # Sequence of high symmetry points

kx_array, k_array, kz_array, k_array, hsp_list = HSP(hsp, stepsize)
   
energiesLent = np.zeros((matrixsizeLent, len(k_array)))
for i in range(int(len(k_array))):
    kx = kx_array[i]
    ky = ky_array[i]
    kz = kz_array[i]
    for j in range(matrixsizeLent):
        energiesLent[j][i] = LA.eigvalsh(Lent(esc, esa, epc, epa, edc, eda, lambdac, lambdaa, vss, vsp, vps, vpp, vpppi, vpd, vpdpi, vdp, vdppi, vdd, vdddelta, vddpi, kx, ky, kz, a))[j]
        
minimum = -15
maximum = 15

# Generate the plot
fig = plt.figure(figsize = (8,7))

for i in range(matrixsizeLent):
    plt.plot(k_array, energiesLent[i], 'red')
plt.ylabel(r"Energy (eV)", fontsize = 15)
plt.xlim(0, max(k_array))
plt.ylim(np.min(energiesLent) - 1,np.max(energiesLent) + 1)
plt.ylim(minimum, maximum, 0.2)
plt.xticks([])
plt.yticks(np.arange(minimum, maximum + 0.1, 5))
for i in range(len(hsp)):
    hsp_point = np.ones(2)*hsp_list[i]
    plt.plot(hsp_point, [minimum, maximum], linestyle = 'dashed', color = 'black')
    plt.text(hsp_list[i], minimum - 1.5, hsp[i][1], fontsize = 15)
plt.show()

gap = min(energiesLent[10]) - max(energiesLent[9])
print("Band gap:", gap, "eV")