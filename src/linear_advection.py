import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


dt=0.0002
num_points=2000;
dx=1.0/num_points;
c=0.2*dt/dx

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


def upwind(u):
	return u

def lax_wendroff(u):
	return 0.5*(1+c)*u+0.5*(1-c)*np.roll(u,-1)
	
def van_leer(r):
	return (r+np.abs(r))/(1+abs(r))
	

def compute_step(u,dt):
	r_num=(u-np.roll(u,1))
	r_den=(np.roll(u,-1)-u);
	r=np.where(np.abs(r_den)<0.0001,np.full(len(r_den),1.0),r_num/r_den)
	psi=van_leer(r)
	flux=psi*lax_wendroff(u)+(1-psi)*upwind(u)
	du=-0.1*(flux-np.roll(flux,1))/dx
	u[:]=u+dt*du
	


#Main plotting routine

def plot_solution(initial_condition):

	u=initial_condition

	fig=plt.figure()
	ax=plt.axes(xlim=(0,1),ylim=(0,2))
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

plot_solution(hump())
