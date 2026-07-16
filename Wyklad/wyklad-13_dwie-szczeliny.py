import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters (user can adjust at top)
Lx, Ly = 1.0, 1.0      # well dimensions
Nx, Ny = 200, 200       # spatial grid points
T = 0.01                # total time
K = 200                 # time steps
V0 = 1e5                # barrier height
dV = 1e20               # infinite walls
x1, x2 = 0.4, 0.6       # barrier x-interval
y1, y2 = 0.45, 0.55     # first slit y-interval
y3, y4 = 0.7, 0.8       # second slit y-interval

# Initial Gaussian wave packet parameters
x0, y0 = 0.1, Ly/2
sigma_x, sigma_y = 0.05, 0.05
p0 = 200.0

# Discretization
dx = Lx / Nx
dy = Ly / Ny
dt = T / K
x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

# Potential V(x,y)
V = np.zeros((Nx+1, Ny+1))
# infinite walls
V[0,:] = dV; V[-1,:] = dV
V[:,0] = dV; V[:,-1] = dV
# barrier with two slits
ix1, ix2 = int(x1/dx), int(x2/dx)
for i in range(ix1, ix2+1):
    for j in range(Ny+1):
        yj = j*dy
        if not (y1 <= yj <= y2 or y3 <= yj <= y4):
            V[i,j] = V0

# Flatten inner grid points (exclude boundaries)
ix = np.arange(1, Nx)
jy = np.arange(1, Ny)
N = (Nx-1)*(Ny-1)

# Map 2D index (i,j) to 1D m
def idx(i, j):
    return (i-1)*(Ny-1) + (j-1)

# Build sparse matrices G and H
rx = -1j * dt/(2*dx**2)
ry = -1j * dt/(2*dy**2)

data_G = []
row_G = []
col_G = []

data_H = []
row_H = []
col_H = []

for i in range(1, Nx):
    for j in range(1, Ny):
        m = idx(i,j)
        Vij = V[i,j]
        a = (1 - 2*rx - 2*ry + -1j*dt/2*Vij)
        b = (1 + 2*rx + 2*ry + 1j*dt/2*Vij)
        # G matrix entries
        for (dm, coeff) in [((0,0), a), ((1,0), rx), ((-1,0), rx), ((0,1), ry), ((0,-1), ry)]:
            ii, jj = i+dm[0], j+dm[1]
            if 0 < ii < Nx and 0 < jj < Ny:
                mm = idx(ii,jj)
                data_G.append(coeff); row_G.append(m); col_G.append(mm)
        # H matrix entries (conjugate signs)
        b0 = b
        rxH = 1j * dt/(2*dx**2)
        ryH = 1j * dt/(2*dy**2)
        for (dm, coeff) in [((0,0), b0), ((1,0), rxH), ((-1,0), rxH), ((0,1), ryH), ((0,-1), ryH)]:
            ii, jj = i+dm[0], j+dm[1]
            if 0 < ii < Nx and 0 < jj < Ny:
                mm = idx(ii,jj)
                data_H.append(coeff); row_H.append(m); col_H.append(mm)

G = sp.csr_matrix((data_G, (row_G, col_G)), shape=(N, N))
H = sp.csr_matrix((data_H, (row_H, col_H)), shape=(N, N))

# Pre-factorize G for solving G psi_{k+1} = H psi_k
solver = spla.factorized(G)

# Initial wavefunction psi0 on inner grid
psi = np.zeros((Nx+1, Ny+1), dtype=complex)
psi[1:-1,1:-1] = (1/np.sqrt(np.pi*sigma_x*sigma_y) *
                  np.exp(-((X[1:-1,1:-1]-x0)**2/(2*sigma_x**2) +
                           (Y[1:-1,1:-1]-y0)**2/(2*sigma_y**2))) *
                  np.exp(1j*p0*(X[1:-1,1:-1]-x0)))
# Normalize
norm = np.sqrt(np.sum(np.abs(psi)**2)*dx*dy)
psi /= norm

# Time evolution and animation
fig, ax = plt.subplots()
frames = []

for k in range(K):
    # Flatten inner psi
    psi_vec = psi[1:-1,1:-1].flatten()
    # Compute next time step
    psi_next_vec = solver(H.dot(psi_vec))
    # Reshape back
    psi[1:-1,1:-1] = psi_next_vec.reshape((Nx-1, Ny-1))
    # Probability density
    prob = np.abs(psi)**2
    # Plot snapshot every few steps
    if k % (K//100) == 0:
        im = ax.imshow(prob.T, origin='lower', extent=(0, Lx, 0, Ly), vmin=0, vmax=prob.max())
        frames.append([im])

ani = animation.ArtistAnimation(fig, frames, interval=50)
plt.colorbar(im, ax=ax)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title('Probability density')
plt.show()
