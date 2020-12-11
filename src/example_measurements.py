import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec

# Normalising velocity
v0 = 1.0

def gaussian(x,mu,sig):
	return np.exp(-((x-mu)/sig)**2/2.0)/np.sqrt(2*np.pi)/sig

# Number of emitted particles of velocity v at time t
def double_gaussian_f(mu_v,sig_v,mu_t,sig_t):
	def f(v,t):
		return gaussian(v,mu_v,sig_v)*gaussian(t,mu_t,sig_t)
	return f

def S_continuous(x,T_norm,vmin,vmax,f):
	ff = lambda v : f(v,T_norm-x/v+x/v0)
	y,err = quad_vec(ff,vmin,vmax)
	return y

def S_discrete(x,T,vmin,vmax,tmin,tmax,f,n_v=100,n_t=100):
	n_T = T.shape[0]
	P   = np.zeros((n_T,n_v,n_t))

	v   = np.linspace(vmin,vmax,n_v)
	t   = np.linspace(tmin,tmax,n_t)
	
	dv = v[1]-v[0]
	dt = t[1]-t[0]
	dT = T[1]-T[0]

	T3ph,v3ph,t3ph = np.meshgrid(T+0.5*dT,v+0.5*dv,t+0.5*dt,indexing='ij')
	T3mh,v3mh,t3mh = np.meshgrid(T-0.5*dT,v-0.5*dv,t-0.5*dt,indexing='ij')
	T3ph += x/v0
	T3mh += x/v0
	for n in range(4):
		if(n == 0):
			a_iln = T3ph-t3mh
		elif(n == 1):
			a_iln = T3ph-t3ph
		elif(n == 2):
			a_iln = T3mh-t3ph
		else:
			a_iln = T3mh-t3mh
		y_kph = a_iln-x/v3ph
		y_kmh = a_iln-x/v3mh
		mask = y_kph > 0.0
		P[mask] += (-1.0)**n*(np.log((a_iln[mask]-y_kph[mask])/a_iln[mask])+y_kph[mask]/(a_iln[mask]-y_kph[mask]))
		mask = y_kmh > 0.0
		P[mask] -= (-1.0)**n*(np.log((a_iln[mask]-y_kmh[mask])/a_iln[mask])+y_kmh[mask]/(a_iln[mask]-y_kmh[mask]))
	P       *= x/dT

	v2,t2    = np.meshgrid(v,t,indexing='ij')
	F        = f(v2,t2)

	S        = np.einsum('ijk,jk->i',P,F)
	return S,F

def init_recon(x_arr,T,vmin,vmax,tmin,tmax,n_v=100,n_t=100):
	n_x = x_arr.shape[0]
	n_T = T.shape[0]
	P   = np.zeros((n_x,n_T,n_v,n_t))

	v   = np.linspace(vmin,vmax,n_v)
	t   = np.linspace(tmin,tmax,n_t)
	
	dv = v[1]-v[0]
	dt = t[1]-t[0]
	dT = T[1]-T[0]

	T3ph_norm,v3ph,t3ph = np.meshgrid(T+0.5*dT,v+0.5*dv,t+0.5*dt,indexing='ij')
	T3mh_norm,v3mh,t3mh = np.meshgrid(T-0.5*dT,v-0.5*dv,t-0.5*dt,indexing='ij')

	for i,x in enumerate(x_arr):
		T3ph = T3ph_norm+x/v0
		T3mh = T3mh_norm+x/v0
		for n in range(4):
			if(n == 0):
				a_iln = T3ph-t3mh
			elif(n == 1):
				a_iln = T3ph-t3ph
			elif(n == 2):
				a_iln = T3mh-t3ph
			else:
				a_iln = T3mh-t3mh
			y_kph = a_iln-x/v3ph
			y_kmh = a_iln-x/v3mh
			mask = y_kph > 0.0
			P[i,mask] += (-1.0)**n*(np.log((a_iln[mask]-y_kph[mask])/a_iln[mask])+y_kph[mask]/(a_iln[mask]-y_kph[mask]))
			mask = y_kmh > 0.0
			P[i,mask] -= (-1.0)**n*(np.log((a_iln[mask]-y_kmh[mask])/a_iln[mask])+y_kmh[mask]/(a_iln[mask]-y_kmh[mask]))
		P[i,...] *= x/dT

	P = P.reshape(n_x,n_T,n_v*n_t)
	P = P.reshape(-1,P.shape[-1])

	return P,(n_x,n_T,n_v,n_t),(v,t)

def LR_reconstruction(N,S,P,f_guess=None,iter_max=1000):
	n_m,n_n = P.shape
	if(f_guess is None):
		f = np.ones(n_n)*N
	else:
		f = f_guess

	# Normalise PSF
	norm = np.sum(P,axis=0)
	norm[norm == 0.0] = 1.0e-10
	for i in range(iter_max):
		c = np.einsum('ij,j->i',P,f)
		c[c == 0.0] = 1.0e-10
		d = S/c
		delta = np.einsum('i,ij->j',d,P)/norm
		f = f*delta
	return f

# Number of measurements
n_d = 30

# x positions
#x_arr = np.array([5.0,10.0,20.0])
x_arr = np.linspace(5.0,30.0,n_d)
print(x_arr)

# Normalised arrival time array
nT = 50
T  = np.linspace(-20,20.0,nT)
tmin = -1.0; tmax = 1.0

# Velocity min and max
vmin = 0.5*v0; vmax = 1.5*v0

f_dg = double_gaussian_f(1.0,0.1,0.0,0.5)

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(111)

for x in x_arr:
	# S_res = S_continuous(x,T,vmin,vmax,f_dg)
	# ax1.plot(T,S_res)
	S_res,F_discrete = S_discrete(x,T,vmin,vmax,tmin,tmax,f_dg)
	ax1.plot(T,S_res)

ax1.set_ylabel(r"$S(x,T')$")
ax1.set_xlabel(r"$T'=T-\frac{x}{v_0}$")

P_mn,dims,arrays = init_recon(x_arr,T,vmin,vmax,tmin,tmax)
S_m = np.einsum('ij,j',P_mn,F_discrete.flatten())
f_proxy = np.einsum('ij,j',P_mn.T,S_m).reshape(dims[-2],dims[-1])

v,t      = arrays
v2,t2    = np.meshgrid(v,t,indexing='ij')
f_guess  = double_gaussian_f(1.0,0.2,0.0,0.2)(v2,t2).flatten()
f_n = LR_reconstruction(np.sum(S_m)/x_arr.shape[0],S_m,P_mn)

S_m = np.einsum('ij,j',P_mn,f_n)
S_m = S_m.reshape(x_arr.shape[0],T.shape[0])
for i in range(x_arr.shape[0]):
	ax1.plot(T,S_m[i,:],'k--')

S_m = np.einsum('ij,j',P_mn,f_guess)
S_m = S_m.reshape(x_arr.shape[0],T.shape[0])
for i in range(x_arr.shape[0]):
	ax1.plot(T,S_m[i,:],'r:')

f_recon = f_n.reshape(dims[-2],dims[-1])

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

im1 = ax1.imshow(f_recon,origin='lower',extent=[t[0],t[-1],v[0],v[-1]])
im2 = ax2.imshow(f_proxy,origin='lower',cmap='coolwarm',extent=[t[0],t[-1],v[0],v[-1]])
ax1.set_xlabel('t')
ax1.set_ylabel('v')

fig.colorbar(im1,ax=ax1,orientation='horizontal')
fig.colorbar(im2,ax=ax2,orientation='horizontal')

plt.show()