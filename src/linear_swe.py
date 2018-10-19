import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


dt=0.00001
num_points=10000;
dx=1.0/num_points;
g=9.81;
H=1

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
	u=np.zeros(num_points)
	quart=num_points//4;
	u[1:quart]=np.full(quart-1,1.0)
	return u

def compute_h_flux(h,m):
	return m

def compute_m_flux(h,m):
	return g*H*h

def compute_step(h,m,dt):
	f_m=compute_m_flux(h,m)
	m[:]=m-(dt/dx)*(f_m-np.roll(f_m,1))
	f_h=compute_h_flux(h,m)	
	h[:]=h-(dt/dx)*(np.roll(f_h,-1)-f_h)
	


#Main plotting routine

def plot_solution(initial_condition):

	h=initial_condition
	m=np.zeros(len(initial_condition))

	fig=plt.figure()
	ax=plt.axes(xlim=(0,1),ylim=(0,2))
	line,=ax.plot([],[],lw=2)

	def animate(i):
		for i in range(0,10):
			compute_step(h,m,dt)
		line.set_data(x,h)
		return line,
	anim=animation.FuncAnimation(fig,animate,frames=100,interval=int(dt*100000),blit=True)	
	plt.show()

plot_solution(hump())