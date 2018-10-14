import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


dt=0.0001
num_points=10000;
dx=10.0/num_points;
g=9.81;
H=1

x=np.linspace(0,10,num_points)


#Test cases

def hump():
	u=np.full(num_points,1.0)
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
	u=np.full(num_points,0.5)
	quart=num_points//2;
	u[1:quart]=np.full(quart-1,1.5)
	return u

def g(domain):
	return np.stack([-domain[1],-(((domain[1]*domain[1])/domain[0])+0.5*9.81*domain[0]*domain[0])]);

def compute_step(domain,dt):
	domain2=(0.5*dt/dx)*(g(np.roll(domain,-1,axis=1))-g(domain))+0.5*(domain+np.roll(domain,-1,axis=1))
	domain+=(dt/dx)*(g(domain2)-g(np.roll(domain2,1,axis=1)));


#Main plotting routine

def plot_solution(initial_condition):

	domain=np.stack([initial_condition,np.zeros(len(initial_condition))])

	fig=plt.figure()
	ax=plt.axes(xlim=(0,10),ylim=(0,2))
	ax.set_aspect("equal")
	line,=ax.plot([],[],lw=2)

	def animate(i):
		for i in range(0,50):
			compute_step(domain,dt)
		line.set_data(x,domain[0])
		return line,
	anim=animation.FuncAnimation(fig,animate,frames=100,interval=int(dt*100000),blit=True)	
	plt.show()

plot_solution(hump())
