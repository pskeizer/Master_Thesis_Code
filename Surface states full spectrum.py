def Slab_par(H, limit):
    ham_mass = ("kron(Matrix([[m_Sn, 0],[0, m_Te]]), eye(3))")
    ham_nn = ("2*t*kron(sigma_x, Matrix([[cos(k_x), 0, 0],[0, cos(k_y), 0],[0, 0, cos(k_z)]]))")
    ham_nnn = ("2*kron(Matrix([[t_Sn, 0],[0, t_Te]]), Matrix([[cos(k_x)*(cos(k_y)+cos(k_z)), -sin(k_x)*sin(k_y), -sin(k_x)*sin(k_z)],[-sin(k_y)*sin(k_x), cos(k_y)*(cos(k_x)+cos(k_z)), -sin(k_y)*sin(k_z)],[-sin(k_z)*sin(k_x), -sin(k_z)*sin(k_y), cos(k_z)*(cos(k_x) + cos(k_y))]]))")
    ham_nnpi = ("-2*tpi * kron(sigma_x, Matrix([[cos(k_y) + cos(k_z), 0, 0],[0, cos(k_x) + cos(k_z), 0],[0, 0, cos(k_x) + cos(k_y)]]))")
    ham_nnnpi = ("-2*kron(Matrix([[tpi_Sn, 0],[0, tpi_Te]]), Matrix([[cos(k_y)*cos(k_z), 0, 0],[0, cos(k_x)*cos(k_z), 0],[0, 0, cos(k_x)*cos(k_y)]]))")
    ham = "("+ ham_mass +" + " + ham_nn +" + " + ham_nnn +" + " + ham_nnpi +" + " + ham_nnnpi +")"
    ham_ = "kron(" + ham + ", eye(2))"
    soc = ("Matrix([[0, 0, 1, 0, 0, 1j],[0, 0, 0, -1, -1j, 0],[-1, 0, 0, 0, 0, 1],[0, 1, 0, 0, 1, 0],[0, -1j, 0, -1, 0, 0],[1j, 0, -1, 0, 0, 0]])")
    soi = ("-1j/2*kron(Matrix([[lambda_Sn, 0],[0, lambda_Te]]), " + soc + ")")
    Ham = "(" + ham_ +" + " + soi +")"
    hamiltonian = qsymm.Model(Ham)

    template = kwant.qsymm.model_to_builder(hamiltonian, 
                                            (('Sn', 6), ('Te', 6)),
                                           np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]),
                                           (([0, 0, 0], [0, 0, 1]))) # Lattice vectors rock salt SnTe
    
    # For the following section, comment out all directions but one.
    
    ######################################## (001)-direction
    def shape(site):
        (x, y, z) = site.pos
        return  0 <= z < H
    syst = kwant.Builder(symmetry = kwant.lattice.TranslationalSymmetry([1, 1, 0], [1, -1, 0]))

    ####################################### (110)-direction
    def shape(site):
        (x, y, z) = site.pos
        return  0 <= x + y < H  
    syst = kwant.Builder(symmetry = kwant.lattice.TranslationalSymmetry([-1, 1, 0], [0, 0, 2]))
    
    ###################################### (111)-direction
    template = kwant.qsymm.model_to_builder(hamiltonian,
                                            (('Sn', 6), ('Te', 6)),
                                           np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]),
                                           (([0, 0, 0], [0, 0, 1]))) # For different termination, exchange the 0 and 1
    def shape(site):
        (x, y, z) = site.pos
        return  0 <= x + y + z < H  
    syst = kwant.Builder(symmetry = kwant.lattice.TranslationalSymmetry([-1, 0, 1], [0, 1, -1]))
    
    ############################ Finalizing Kwant
    
    syst.fill(template, shape, (0,0,0))
    
    syst = kwant.wraparound.wraparound(syst)
    syst = syst.finalized() 
    
    kx_array, ky_array, k_array, hsp_list = HSP(hsp, stepsize = 100)
        
    matrixsizeslab = H*12 # Size of basis: 2 atoms per unit cell, 6 orbitals per atom, H atoms thick in slab
    energiesslab = np.zeros((matrixsizeslab, len(kx_array)))
    for i in range(len(kx_array)):
        param_Hsieh = dict(
        m_Sn = -1.65, 
        m_Te = 1.65,
        t = 0.9, 
        t_Sn = -0.5, 
        t_Te = 0.5,
        tpi = 0,
        tpi_Sn = 0,
        tpi_Te = 0,
        lambda_Sn = 0.6, 
        lambda_Te = 0.6,
        k_x = kx_array[i],
        k_y = ky_array[i])
        for j in range(matrixsizeslab):
            energiesslab[j][i] = LA.eigvalsh(syst.hamiltonian_submatrix(params = param_Hsieh))[j]
            print('\r' + str(int(j/matrixsizeslab*100))+'% completed of a single k and ' + str((i/len(kx_array)*100))+'% completed of total k-path',end='')
        
        return energiesslab
    
# Running slab-geometry function Comment out all directions but one

# Along (001)-direction
GX_k = [np.pi/a*np.array([1/2, 0]), '$\overline{\\Gamma} \\leftarrow$']
X_k = [np.pi/a*np.array([1, 0]), '$\overline{X}$']
XM_k = [np.pi/a*np.array([1, 1/2]), '$\\rightarrow \overline{M}$']

H = 80 # Number of atoms thick
hsp = (GX_k, X_k, XM_k)
kx_array, ky_array, k_array, hsp_list = HSP(hsp, stepsize = 100)

energiesslab = Slab(H, hsp)
matrixsizeslab = H*12 # Size of basis: 2 atoms per unit cell, 6 orbitals per atom, H atoms thick in slab

# Along (110)-direction

Y_k = [np.pi/a*np.array([0, 1]), '$\overline{Y}$']
M_k = [np.pi/a*np.array([1, 1]), '$\overline{S}$']
GY_k = [np.pi/a*np.array([0, 0.5]), '$\overline{\\Gamma} \\leftarrow$']
MG_k = [np.pi/a*np.array([0.5, 0.5]), '$\\rightarrow \overline{\\Gamma}$']

H = 56 # Number of atoms thick
hsp = (GY_k, Y_k, M_k, MG_k)
kx_array, ky_array, k_array, hsp_list = HSP(hsp, stepsize = 100)

energiesslab = Slab(H, hsp)
matrixsizeslab = H*12 # Size of basis: 2 atoms per unit cell, 6 orbitals per atom, H atoms thick in slab

# Along (111)-direction
G_k = [np.pi/a*np.array([0, 0]), '$\overline{\\Gamma}$']  
M_k = [np.pi/a*np.array([0, 1]), '$\overline{M}$']
K_k = [np.pi/a*np.array([2/3, 2/3]), '$\overline{K}$']

H = 91 # Number of atoms thick
hsp = (K_k, G_k, M_k, K_k)
kx_array, ky_array, k_array, hsp_list = HSP(hsp, stepsize = 100)

energiesslab = Slab(H, hsp)
matrixsizeslab = H*6 # Size of basis: 1 atoms per unit cell, 6 orbitals per atom, H atoms thick in slab

# Plotting

minimum = -.5
maximum = .5

surface = int(energiesslab.shape[0]/2)

begin1, end1 = 44, 170 # Region of surface states
begin2, end2 = 43, 172

fig = plt.figure(figsize = (7,7))
for i in range(matrixsizeslab):
    plt.plot(k_array, energiesslab[i], 'silver')
plt.plot(k_array[begin1 : end1], energiesslab[surface - 3][begin1 : end1], 'red')
plt.plot(k_array[begin1 : end1], energiesslab[surface - 1][begin1 : end1], 'red')
plt.plot(k_array[begin2 : end2], energiesslab[surface][begin2 : end2], 'red')
plt.plot(k_array[begin2 : end2], energiesslab[surface + 2][begin2 : end2], 'red')
plt.ylabel(r"Energy (eV)", fontsize = 15)
plt.xlim(0, max(k_array))
plt.ylim(minimum, maximum, 0.1)
plt.xticks([])
plt.yticks(np.arange(minimum, maximum + 0.1, 0.1))
for i in range(len(hsp)):
    hsp_point = np.ones(2)*hsp_list[i]
    plt.plot(hsp_point, [minimum, maximum], linestyle = 'dashed', color = 'black')
    plt.text(hsp_list[i], minimum - 0.05, hsp[i][1], fontsize = 15)
plt.show()