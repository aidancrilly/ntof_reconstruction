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

def S_discrete(x,T,vmin,vmax,tmin,tmax,f,n_v=102,n_t=103):
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
	return S

# Number of measurements
n_d = 3

# x positions
x_arr = np.array([5.0,10.0,20.0])

# Normalised arrival time array
nT = 200
T  = np.linspace(-10,10.0,nT)
tmin = -1.0; tmax = 1.0

# Velocity min and max
vmin = 0.5*v0; vmax = 1.5*v0

f_dg = double_gaussian_f(1.0,0.1,0.0,0.1)

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(111)

for x in x_arr:
	S_res = S_continuous(x,T,vmin,vmax,f_dg)
	ax1.plot(T,S_res)
	S_res = S_discrete(x,T,vmin,vmax,tmin,tmax,f_dg)
	ax1.plot(T,S_res,'k--')

ax1.set_ylabel(r"$S(x,T')$")
ax1.set_xlabel(r"$T'=T-\frac{x}{v_0}$")


plt.show()