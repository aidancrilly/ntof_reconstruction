from deep_reconstruction_model import nn_model
import numpy as np
import tensorflow as tf
from keras.utils import plot_model
import matplotlib.pyplot as plt

v0 = 1.0

def gaussian(x,mu,sig):
	return np.exp(-((x-mu)/sig)**2/2.0)/np.sqrt(2*np.pi)/sig

# Number of emitted particles of velocity v at time t
def double_gaussian_f(mu_v,sig_v,mu_t,sig_t):
	def f(v,t):
		return gaussian(v,mu_v,sig_v)*gaussian(t,mu_t,sig_t)
	return f

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

N_train = 10
n_v = 40
n_t = 40
n_x = 3
n_T = 200

vmin = 0.5*v0; vmax = 1.5*v0
tmin = -1.0; tmax = 1.0

x_arr = np.linspace(1e-3,5.0,n_x)
T     = np.linspace(-5.0,5.0,n_T)

A,dims,arrs = init_recon(x_arr,T,vmin,vmax,tmin,tmax,n_v=n_v,n_t=n_t)
v,t         = arrs[0],arrs[1]
v2,t2       = np.meshgrid(v,t,indexing='ij')

A_train = np.zeros((N_train,n_x*n_T,n_v*n_t))
X_train = np.zeros((N_train,n_v,n_t,1))
y_train = np.zeros((N_train,n_v*n_t))
rands   = np.random.random((2,N_train))
for i in range(N_train):
	A_train[i,...]	 = A
	f                = double_gaussian_f(1.0,0.1+0.2*rands[0,i],0.0,0.1+0.2*rands[1,i])
	F_discrete       = f(v2,t2)
	y_train[i,:]     = F_discrete.flatten()
	S_m              = np.einsum('ij,j',A,y_train[i,:])
	f_proxy          = np.einsum('ij,j',A.T,S_m).reshape(n_v,n_t)
	X_train[i,:,:,0] = f_proxy

fig = plt.figure()
for i in range(n_x):
	plt.plot(T,S_m.reshape(n_x,n_T)[i,:],'x')

n_iters = 1
n_cnn_layers = 3

input_shape = (n_v,n_t,1)
A_shape     = (n_x*n_T,n_v*n_t)

model_deep  = nn_model(input_shape,A_shape,n_cnn_layers,n_iters,10,training=True)
model_deep.compile(optimizer='adam',loss='mean_squared_error')
print(model_deep.summary())
model_deep.fit([X_train,A_train],y_train,epochs=10,batch_size=10)
y_pred = model_deep.predict([X_train[0:1,...],A_train[0:1,...]])

model_CG  = nn_model(input_shape,A_shape,0,0,10,training=True)
model_CG.compile(optimizer='adam',loss='mean_squared_error')
y_CG = model_CG.predict([X_train[0:1,...],A_train[0:1,...]])

f_pred = y_pred.reshape(n_v,n_t)
f_CG   = y_CG.reshape(n_v,n_t)
f_train = y_train[0,:].reshape(n_v,n_t)

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.set_title("CG Inv")
ax2.set_title("Deep Model")
ax3.set_title("Truth")

im1 = ax1.imshow(f_CG,origin='lower',extent=[t[0],t[-1],v[0],v[-1]])
im2 = ax2.imshow(f_pred,origin='lower',extent=[t[0],t[-1],v[0],v[-1]])
im3 = ax3.imshow(f_train,origin='lower',extent=[t[0],t[-1],v[0],v[-1]])

fig.colorbar(im1,ax=ax1,orientation='horizontal')
fig.colorbar(im2,ax=ax2,orientation='horizontal')
fig.colorbar(im3,ax=ax3,orientation='horizontal')

plt.show()