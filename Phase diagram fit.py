# Data lower limit
data_low = np.array([
    [-0.7, 0.3, 2.25],
    [-0.8, 0.3, 2.25],
    [-0.5, 0.4, 2.25],
    [-0.6, 0.4, 2.5],
    [-0.7, 0.4, 2.5],
    [-0.8, 0.4, 2.75],
    [-0.4, 0.5, 2.25],
    [-0.5, 0.5, 2.5],
    [-0.6, 0.5, 2.75],
    [-0.7, 0.5, 2.75],
    [-0.8, 0.5, 3],
    [-0.4, 0.6, 2.25],
    [-0.5, 0.6, 2.5],
    [-0.6, 0.6, 2.75],
    [-0.7, 0.6, 3],
    [-0.8, 0.6, 3.25],
    [-0.4, 0.7, 2.5],
    [-0.5, 0.7, 2.75],
    [-0.6, 0.7, 3],
    [-0.7, 0.7, 3.25],
    [-0.8, 0.7, 3],
    [-0.4, 0.8, 2.75],
    [-0.5, 0.8, 3],
    [-0.6, 0.8, 3.25],
    [-0.8, 0.7, 3.5],
    [-0.8, 0.8, 3.75],
    [-0.4, 0.9, 3],
    [-0.5, 0.9, 3.25],
    [-0.6, 0.9, 3.5],
    [-0.7, 0.9, 3.75],
    [-0.8, 0.9, 4]    
])

# Data upper limit
data_high = np.array([
    [-0.4, 0.1, 2.25],
    [-0.5, 0.1, 2.5],
    [-0.3, 0.2, 2],
    [-0.4, 0.2, 2.5],
    [-0.5, 0.2, 2.75],
    [-0.6, 0.3, 3.5],
    [-0.1, 0.4, 2],
    [-0.2, 0.4, 2.75],
    [-0.3, 0.4, 2.75],
    [-0.4, 0.4, 3.25],
    [-0.5, 0.4, 3.5],
    [-0.1, 0.5, 2.5],
    [-0.2, 0.5, 2.75],
    [-0.3, 0.5, 3.25],
    [-0.4, 0.5, 3.5]
])
#print(len(data_low)) # Data size
#print(len(data_high))

fig = plt.figure(figsize = (7,7))
ax = Axes3D(fig)

data = data_low, data_high
colour = 'deepskyblue', 'red'

for i in range(len(data)):
    mn = np.min(data[i], axis = 0)
    mx = np.max(data[i], axis = 0)

    # Regular grid covering the domain

    X, Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
    XX = X.flatten()
    YY = Y.flatten()

    # Best-fit linear plane
    A = np.c_[data[i][:,0], data[i][:,1], np.ones(data[i].shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data[i][:,2])

    Z = C[0]*X + C[1]*Y + C[2]
    
    print(C[0], C[1], C[2])
    
    ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, alpha = 0.2)
    ax.scatter(data[i][:,0], data[i][:,1], data[i][:,2], c = colour[i], s = 50)

ax.set_zlabel('$|m_{Sn}| + m_{Te}$', fontsize = 15)
plt.ylabel('$t_{Te}$', fontsize = 15)
plt.xlabel('$t_{Sn}$', fontsize = 15)
ax.view_init(elev = 10, azim = 25)
ax.axis('tight') 
plt.show()