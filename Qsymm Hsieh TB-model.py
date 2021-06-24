# Generating a family of matrices from symmetry constraints

sigma_x = np.array([[0, 1],[0, 1]])
sigma_y = np.array([[0, -1j],[1j, 0]])
sigma_z = np.array([[1, 0],[0, -1]])

hopping_vectors = [
                ('Sn_px', 'Te_px', np.array([1, 0, 0])), # nn px - px
                ('Sn_py', 'Te_py', np.array([0, 1, 0])), # nn py - py
                ('Sn_pz', 'Te_pz', np.array([0, 0, 1])), # nn pz - pz
                ('Sn_px', 'Sn_px', np.array([1, 1, 0])), # Sn nnn px - px  x,y-plane
                ('Sn_px', 'Sn_py', np.array([1, 1, 0])), # Sn nnn px - py  x,y-plane
                ('Sn_py', 'Sn_py', np.array([1, 1, 0])), # Sn nnn py - py  x,y-plane
                ('Sn_px', 'Sn_px', np.array([1, 0, 1])), # Sn nnn px - px  x,z-plane
                ('Sn_px', 'Sn_pz', np.array([1, 0, 1])), # Sn nnn px - pz  x,z-plane
                ('Sn_pz', 'Sn_pz', np.array([1, 0, 1])), # Sn nnn pz - pz  x,z-plane
                ('Sn_py', 'Sn_py', np.array([0, 1, 1])), # Sn nnn py - py  y,z-plane
                ('Sn_py', 'Sn_pz', np.array([0, 1, 1])), # Sn nnn py - pz  y,z-plane
                ('Sn_pz', 'Sn_pz', np.array([0, 1, 1])), # Sn nnn pz - pz  y,z-plane 
                ('Te_px', 'Te_px', np.array([1, 1, 0])), # Te nnn px - px  x,y-plane
                ('Te_px', 'Te_py', np.array([1, 1, 0])), # Te nnn px - py  x,y-plane
                ('Te_py', 'Te_py', np.array([1, 1, 0])), # Te nnn py - py  x,y-plane
                ('Te_px', 'Te_px', np.array([1, 0, 1])), # Te nnn px - px  x,z-plane
                ('Te_px', 'Te_pz', np.array([1, 0, 1])), # Te nnn px - pz  x,z-plane
                ('Te_pz', 'Te_pz', np.array([1, 0, 1])), # Te nnn pz - pz  x,z-plane
                ('Te_py', 'Te_py', np.array([0, 1, 1])), # Te nnn py - py  y,z-plane
                ('Te_py', 'Te_pz', np.array([0, 1, 1])), # Te nnn py - pz  y,z-plane
                ('Te_pz', 'Te_pz', np.array([0, 1, 1])), # Te nnn pz - pz  y,z-plane 
                  ]

sites = ['Sn_px', 'Sn_py', 'Sn_pz', 'Te_px', 'Te_py', 'Te_pz'] # The basis
norbs = [(site, 1) for site in sites]

# Time reversal symmetry
TR = qsymm.time_reversal(3, U = -np.eye(6))

# Four-fold rotational symmetry
sphi = np.pi/2 # Angle of rotation for C_4

# Rotation around the z-axis
UC4z = np.array([
    [0,-1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0,-1, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1]
])

C4z = qsymm.PointGroupElement(
    sympy.Matrix([
        [sympy.cos(sphi),-sympy.sin(sphi),0],
        [sympy.sin(sphi), sympy.cos(sphi),0],
        [0,0,1]
    ]),
    U = UC4z)

# Rotation around the y-axis
UC4y = np.array([
    [0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
   [-1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0,-1, 0, 0]
])

C4y = qsymm.PointGroupElement(
    sympy.Matrix([
        [sympy.cos(sphi),0,sympy.sin(sphi)],
        [0,1,0],
        [-sympy.sin(sphi),0,sympy.cos(sphi)]
    ]),
    U = UC4y)

# Rotation around the x-axis
UC4x = np.array([
   [-1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0,-1, 0, 0, 0, 0],
    [0, 0, 0,-1, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0,-1, 0]
])

C4x = qsymm.PointGroupElement(
    sympy.Matrix([
        [1,0,0],
        [0,sympy.cos(sphi),-sympy.sin(sphi)],
        [0,sympy.sin(sphi),sympy.cos(sphi)]
    ]),
    U = UC4x)

# Mirror symmetries
Mx = qsymm.mirror([1, 0, 0], U = np.diag([-1, 1, 1,-1, 1, 1]))
My = qsymm.mirror([0, 1, 0], U = np.diag([ 1,-1, 1, 1,-1, 1]))
Mz = qsymm.mirror([0, 0, 1], U = np.diag([ 1, 1,-1, 1, 1,-1]))

# Inversion symmetry
I = qsymm.PointGroupElement(-np.eye(3), U = -np.eye(6))

# Chiral symmetry
UC = np.array([
    [0, 0, 0, -1j, 0, 0],
    [0, 0, 0, 0, -1j, 0],
    [0, 0, 0, 0, 0, -1j],
    [1j, 0, 0, 0, 0, 0],
    [0, 1j, 0, 0, 0, 0],
    [0, 0, 1j, 0, 0, 0]
])
C = qsymm.chiral(3, U = UC)

# The group elements
generators = {TR, C4x, C4y, C4z, Mx, My, Mz, I} # without Chiral symmetry
sg = qsymm.groups.generate_group(generators)
generators_C = {TR, C4x, C4y, C4z, Mx, My, Mz, I, C} # with Chiral symmetry
sg_C = qsymm.groups.generate_group(generators_C)

################################### Matrix family without C-symmetry

family = qsymm.bloch_family(hopping_vectors, generators, norbs = norbs)

print("Without C-symmetry, we find", len(family), "matrices")
qsymm.display_family(family)

################################## Matrix family with C-symmetry

family_C = qsymm.bloch_family(hopping_vectors, generators_C, norbs = norbs)
print("With C-symmetry, we find", len(family_C), "matrices")
qsymm.display_family(family_C)

# Extrating all symmetries from the Hsieh model

ham_mass = ("kron(Matrix([[m_Sn, 0],[0, m_Te]]), eye(3))")
ham_nn = ("2*t*kron(sigma_x, Matrix([[cos(k_x), 0, 0],[0, cos(k_y), 0],[0, 0, cos(k_z)]]))")
ham_nnn = ("2*kron(Matrix([[t_Sn, 0],[0, t_Te]]),  Matrix([[cos(k_x)*(cos(k_y)+cos(k_z)), -sin(k_x)*sin(k_y), -sin(k_x)*sin(k_z)],[-sin(k_y)*sin(k_x), cos(k_y)*(cos(k_x)+cos(k_z)), -sin(k_y)*sin(k_z)],[-sin(k_z)*sin(k_x), -sin(k_z)*sin(k_y), cos(k_z)*(cos(k_x) + cos(k_y))]]))")
ham = "("+ ham_mass +" + " + ham_nn +" + " + ham_nnn +")"
ham_ = "kron(" + ham + ", eye(2))"
soc = ("Matrix([[0, 0, 1, 0, 0, 1j],[0, 0, 0, -1, -1j, 0],[-1, 0, 0, 0, 0, 1],[0, 1, 0, 0, 1, 0],[0, -1j, 0, -1, 0, 0],[1j, 0, -1, 0, 0, 0]])")
soi = ("-1j*kron(Matrix([[lambda_Sn, 0],[0, lambda_Te]]), " + soc + ")")
Ham = "(" + ham_ +" + " + soi +")"

H = qsymm.Model(ham) # Spinless model
#H = qsymm.Model(ham_) # Spinfull model without SOI
#H = qsymm.Model(Ham) # Full model with SOI
H.tosympy(nsimplify=True)

cubic_group = qsymm.groups.cubic() # O_h group

ds3D, cs3D = qsymm.symmetries(H, cubic_group) # Find all operations belonging to O_h
print('For a model with SOI, the number of discrete symmetries is:', len(ds3D),'and the number of continuous symmetries is: ', len(cs3D))

############################################ Parameter substitution which yields C-symmetry

H_c = H.subs({'t_Te': '-t_Sn'})
H_c = H_c.subs({'m_Te' : '-m_Sn'}) # Substitute the hopping and mass terms
H_c.tosympy(nsimplify=True)

cubic_group = qsymm.groups.cubic()

ds3D_, cs3D_ = qsymm.symmetries(H_c, cubic_group)
print('If we perform a parameter substitution for C-symmetry, the number of discrete symmetries is:', len(ds3D_),'and the number of continuous symmetries is: ', len(cs3D_))