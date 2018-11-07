import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


dt=0.0001
num_points=1000;
dx=10.0/num_points;

x=np.linspace(0,10,num_points)


#Test cases

def hump():
	u=np.full(num_points,0.5)
	quart=num_points//4;
	u[0:quart]+=0.125*(1-np.cos(0.8*np.pi*x[0:quart]))
	return u

def slope():
	u=np.full(num_points,1.0)
	u[0]=1.0
	quart=num_points//4;
	u[1:quart]+=x[1:quart]/2.5
	u[quart+1:2*quart]+=(5-x[quart+1:2*quart])/2.5
	return u

def square():
	u=np.full(num_points,0.05)
	quart=num_points//4;
	u[0:quart]=np.full(quart,1.5)
	return u

def f(u):
	return np.stack([u[1],(((u[1]*u[1])/u[0])+0.5*9.81*u[0]*u[0])]);


def compute_lax_friedrich_flux(u,dt):
	#According to Wikipedia, for the lax-friedrich method diff should be 1, but this seems to be far more diffusion than necessary.
	#However, the optimal value of diff seems to vary with the grid spacing, so I think there is an error in my implementation.
	diff=0.025
	center_fluxes=f(u);
	return 0.5*(np.roll(center_fluxes,1,axis=1)+center_fluxes)-diff*(dx/dt)*(u-np.roll(u,1,axis=1))

def compute_lax_wendroff_flux(u,dt):
	center_flux=f(u)
	u2=(0.5*dt/dx)*(np.roll(center_flux,1,axis=1)-center_flux)+0.5*(u+np.roll(u,1,axis=1))
	return f(u2)

def superbee(r):
	return np.maximum(0,np.maximum(np.minimum(2*r,1),np.minimum(r,2)))

def compute_limited_flux(u,dt):
	lax_friedrich_flux=compute_lax_friedrich_flux(u,dt)
	lax_wendroff_flux=compute_lax_wendroff_flux(u,dt)
	#This is the formula for r given in the notes, but it is not symmetric and it seems like it should be, since shocks may propogate
	#in either direction.
	r_num=(np.roll(u,1,axis=1)-np.roll(u,2,axis=1))
	r_den=(u-np.roll(u,1,axis=1));
	r=np.where(np.abs(r_den)<0.0001,np.full_like(r_den,1.0),r_num/r_den)
	psi=superbee(r)
	return psi*lax_wendroff_flux+(1-psi)*lax_friedrich_flux


def compute_step(u,dt,flux_func):
	flux=flux_func(u,dt)
	u+=(dt/dx)*(flux-np.roll(flux,-1,axis=1));

#Main plotting routine

def plot_solution(initial_condition,flux_func):

	u=np.stack([initial_condition,np.zeros(len(initial_condition))])

	fig=plt.figure()
	ax=plt.axes(xlim=(0,10),ylim=(0,2))
	ax.set_aspect("equal")
	line,=ax.plot([],[],lw=2)

	def animate(i):
		for i in range(0,50):
			compute_step(u,dt,flux_func)
		line.set_data(x,u[0])
		return line,
	anim=animation.FuncAnimation(fig,animate,frames=100,interval=int(dt*50000),blit=True)	
	plt.show()

#This method works, but it's too diffusive and only first order
plot_solution(square(),compute_lax_friedrich_flux)
#This method works well with a higher grid resolution, but with 1000 points it introduces severe spurious oscillations
plot_solution(square(),compute_lax_wendroff_flux)
#This method just doesn't work at all. I've only included it because I'd like to know what's wrong with it
plot_solution(square(),compute_limited_flux)
