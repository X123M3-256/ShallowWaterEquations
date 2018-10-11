import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


dt=0.0002
num_points=100;
dx=1.0/num_points;

xs=np.linspace(0,1,num_points)


#Test cases

def hump():
	us=np.zeros(num_points)
	quart=num_points//4;
	us[0:quart]=0.5*(1-np.cos(8*np.pi*xs[0:quart]))
	return us

def slope():
	us=np.zeros(num_points)
	us[0]=1.0
	half=num_points//2;
	us[1:half]=2.0*(0.5-xs[1:half])
	return us

def square():
	us=np.zeros(num_points)
	quart=num_points//4;
	us[1:quart]=np.full(quart-1,1.0)
	return us

def step():
	us=np.zeros(num_points)
	us[0]=1.0;
	return us

#Difference schemes

def cs(us):
	nus=np.zeros(len(us)-1)
	nus[:-1]=(us[2:]-us[0:-2])/(2*dx)
	nus[-1:]=(us[-1:]-us[-2:-1])/dx
	return nus

def bs(us):
	return (us[1:]-us[0:-1])/dx

#Equations

def linear_advection(c):
	return lambda us,difference_scheme:-c*difference_scheme(us)

def burgers():
	return lambda us,difference_scheme:-us[1:]*difference_scheme(us)



#Main plotting routine
t=0.0;

def plot_solution(initial_condition,equation,difference_scheme):

	us=initial_condition

	fig=plt.figure()
	ax=plt.axes(xlim=(0,1),ylim=(0,2))
	line,=ax.plot([],[],lw=2)

	def init():
		line.set_data([],[])
		return line,
	def animate(i):
		global t
		for i in range(0,100):
			us[1:]=us[1:]+dt*equation(us,difference_scheme)
			t+=dt
		line.set_data(xs,us)
		return line,
	anim=animation.FuncAnimation(fig,animate,init_func=init,frames=100,interval=int(dt*100000),blit=True)	
	plt.show()

plot_solution(step(),linear_advection(0.1),cs)
