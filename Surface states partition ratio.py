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
    
    def manual_bands(kx):
        new_params_ = dict(
            m_Sn = -1.65, # This one should be negative
            m_Te = 1.65, # This one should be positive
            t = 0.9, 
            t_Sn = -0.5, # This one should be negative
            t_Te =  0.5, # This one should be positive
            tpi = 0,
            tpi_Sn = 0,
            tpi_Te = 0,
            lambda_Sn = 0.6, 
            lambda_Te = 0.6,
            k_y = 0,
            **dict(k_x = kx))
        H = syst.hamiltonian_submatrix(params = new_params_, sparse = False)
        ev, es = np.linalg.eigh(H)
        return ev, es
    
    momenta = np.linspace(np.pi/2, np.pi, 101)
    ev_es = [manual_bands(kx) for kx in momenta]
    
    momenta = np.linspace(np.pi/2, np.pi, 101)
    ev_es = [manual_bands(kx) for kx in momenta]
    
    momenta = np.linspace(0, np.pi, 101)
    ev_es = [manual_bands(kx) for kx in momenta]
    
    
    # obtain the IPR of each eigenstate
    E_k = [i[0] for i in ev_es] # just the eigenvalues
    # only keep some of the eigenvalues
    small_ek = []
    for i in range(len(E_k)):
        sub_small_ek = []
        for j in range(len(E_k[0])):
            if np.abs(E_k[i][j]) <= limit:
                sub_small_ek.append(E_k[i][j])
        small_ek.append(sub_small_ek)
    IIPR = [] #the inverse inverse participation ratio (high for edge states, low for bulk states)
    for k_point in range(len(momenta)):
        sub_iipr = []
        es = ev_es[k_point][1] #eigenvectors at a given k
        for i, es_point in enumerate(range(len(es))):
            eigenvector = es[:, es_point]
            if np.abs(E_k[k_point][i])> limit:
                continue
            treated_entries = []
            for entry in eigenvector:
                treated_entries.append(np.abs(np.conj(entry) * entry) ** 4) #the (inverse) inverse participation ratio
            sub_iipr.append(np.sum(np.array(treated_entries)))
        IIPR.append(sub_iipr)


    m_ = [[momenta[ind] for j in range(len(ev_es[ind][0]))] for ind in range(len(momenta))]

    max_iipr = 0  # since it's a ragged array we have to go through each row individually (can't just flatten the array)
    for iipr_ in IIPR:
        for iipr in iipr_:
            if iipr>max_iipr:
                max_iipr = iipr

    fig = plt.figure(figsize = (3.5,7))
    for k_point in range(len(momenta)):
        e_k = small_ek[k_point]#select an array of energies at a single point in k-space
        iipr = IIPR[k_point] #select the corresponding IIPR
        k_points = [momenta[k_point] for i in range(len(e_k))]
        rgba_colors = np.zeros((len(iipr), 4))
        rgba_colors[:, 3] = (iipr / max_iipr) ** 0.4 #the third index of the color sets the transparency, and has to be a value between 0 and 1.

        plt.scatter(k_points, e_k, color=rgba_colors, s=0.3, rasterized=True)
    #plt.xlabel(r"k", fontsize = 15)
    plt.xticks([])
    plt.ylabel(r"Energy (eV)", fontsize = 15)
    plt.xlim(0, momenta[-1])
    plt.ylim(-limit, limit, 0.2)
    plt.yticks(np.arange(-limit, limit + 0.1, 0.1))
    
    plt.text(momenta[0], -limit - 0.05, '$\overline{\\Gamma} \leftarrow$', fontsize = 15) # Axes (001)-direction
    plt.text(momenta[-1], -limit - 0.05, '$\overline{X}$', fontsize = 15)
    
    
    plt.text(momenta[0], -limit - 0.05, '$\overline{\\Gamma} \leftarrow$', fontsize = 15) # Axes (110)-direction
    plt.text(momenta[-1], -limit - 0.05, '$\overline{Y}$', fontsize = 15)
    
    plt.text(momenta[0], -limit - 0.08, '$\overline{\\Gamma}$', fontsize = 15) # Axes (111)-direction
    plt.text(momenta[-1], -limit - 0.08, '$\overline{M}$', fontsize = 15)
    return E_k, IIPR

# Run one of three directions

E_k, IIPR = Slab_par(H = 80, limit = 0.5) # (001)-direction
E_k, IIPR = Slab_par(H = 56, limit = 0.5) # (110)-direction
E_k, IIPR = Slab_par(H = 91, limit = 0.8) # (111)-direction