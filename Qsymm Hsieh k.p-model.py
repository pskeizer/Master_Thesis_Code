# Double spin-1/2 representation
spin = [np.kron(s, np.eye(2)) for s in qsymm.groups.spin_matrices(1/2)]

# Three-fold symmetry around k_3 axis
C3 = qsymm.rotation(1/3, axis = [0, 0, 1], spin = spin)

# Two-fold symmetry around k_1 axis
C2 = qsymm.rotation(1/2, axis = [1, 0, 0], spin = spin)

# Mirror symmetry w.r.t k_1 axis
M = qsymm.mirror(axis = [1, 0, 0], spin = spin)

# Inversion
IU = np.kron(np.eye(2), np.diag([1, -1]))
I = qsymm.inversion(3, U = IU)

# Time reversal
TR = qsymm.time_reversal(3, spin = spin)

symmetries = [TR, I, M, C3]


dim = 3 # Three-dimensional theory
total_power = 1 # Up to linear terms in k
family = qsymm.continuum_hamiltonian(symmetries, dim, total_power, prettify=True)

identity_terms = qsymm.continuum_hamiltonian([qsymm.groups.identity(dim, 1)], dim, total_power)

identity_terms = [
    qsymm.Model({
        key: np.kron(val, np.eye(4))
        for key, val in term.items()
    })
    for term in identity_terms
]

family = qsymm.hamiltonian_generator.subtract_family(family, identity_terms, prettify = True)
print("The Hsieh k.p-theory consists of", len(family)," linear indepent matrices")
qsymm.display_family(family)