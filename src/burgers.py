import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


dt=0.0001
num_points=10000;
dx=1.0/num_points;
c=0.1*dt/dx

x=np.linspace(0,1,num_points)


#Test cases

def hump():
	u=np.zeros(num_points)
	quart=num_points//4;
	u[0:quart]=0.5*(1-np.cos(8*np.pi*x[0:quart]))
	return u

def slope():
	u=np.zeros(num_points)
	u[0]=1.0
	half=num_points//2;
	u[1:half]=2.0*(0.5-x[1:half])
	return u

def square():
	u=np.full(num_points,-0.3333)
	quart=num_points//4;
	u[1:quart]=np.full(quart-1,1.0)
	return u

def burgers_flux(u):
	return 0.5*u*u

def upwind(f,u):
	return f(np.roll(u,1)) #Assumes "wind" is from left to right

def lax_friedrich(f,u,diff):
	return 0.5*(f(u)+f(np.roll(u,1)))-diff*(u-np.roll(u,1))


def compute_step(u,dt):
	interface_fluxes=lax_friedrich(burgers_flux,u,0.5)
	u[:]+=dt*(interface_fluxes-np.roll(interface_fluxes,-1))/dx
	


#Main plotting routine

def plot_solution(initial_condition):

	u=initial_condition

	fig=plt.figure()
	ax=plt.axes(xlim=(0,1),ylim=(-1,1))
	line,=ax.plot([],[],lw=2)

	def init():
		line.set_data([],[])
		return line,
	def animate(i):
		for i in range(0,100):
			compute_step(u,dt)
		line.set_data(x,u)
		return line,
	anim=animation.FuncAnimation(fig,animate,init_func=init,frames=100,interval=int(dt*100000),blit=True)	
	plt.show()

plot_solution(square())
