import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


dt=0.0001
num_points=1000;
dx=10.0/num_points;
g=9.81;
H=1

x=np.linspace(-5,5,num_points)


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
	u=np.full(num_points,0.5)
	quart=num_points//2;
	u[0:quart]=np.full(quart,1.0)
	return u

def g(domain):
	return np.stack([domain[1],(((domain[1]*domain[1])/domain[0])+0.5*9.81*domain[0]*domain[0])]);

def compute_lax_friedrich_flux(domain,dt):
	center_fluxes=g(domain);
	return 0.5*(np.roll(center_fluxes,1,axis=1)+center_fluxes)-0.5*(dx/dt)*(domain-np.roll(domain,1,axis=1))

def compute_lax_wendroff_flux(domain,dt):
	center_flux=g(domain)
	domain2=(0.5*dt/dx)*(np.roll(center_flux,1,axis=1)-center_flux)+0.5*(domain+np.roll(domain,1,axis=1))
	return g(domain2)

def compute_step(domain,dt):
	lax_friedrich_flux=compute_lax_wendroff_flux(domain,dt)
	#lax_friedrich_flux=compute_lax_friedrich_flux(domain,dt)
	domain+=(dt/dx)*(lax_friedrich_flux-np.roll(lax_friedrich_flux,-1,axis=1));


def analytic(x,t):
	x+=0.5
	g=9.81;
	S=2.957918120187525
	u2=S-(g/(8.0*S))*(1.0+np.sqrt(1.0+(16.0*S**2)/g))
	c2=np.sqrt(0.25*g*(np.sqrt(1.0+(16.0*S**2)/g)-1))
	downstream_depth=0.5
	bore_depth=0.25*(np.sqrt(1+(16*S**2)/g)-1)
	if(x<0.5-t*np.sqrt(g)):
		return 1
	elif(x<(u2-c2)*t+0.5):
		return (2.0*np.sqrt(g)-(2.0*x-1)/(2.0*t))**2/(9.0*g)
	elif(x<S*t+0.5):
		return bore_depth
	else:
		return downstream_depth


#Main plotting routine

def plot_solution(initial_condition):

	domain=np.stack([initial_condition,np.zeros(len(initial_condition))])

	fig=plt.figure()
	ax=plt.axes(xlim=(-1,1),ylim=(-1,1))
	ax.set_aspect("equal")
	line,=ax.plot([],[],lw=1)
	line2,=ax.plot([],[],lw=1)
	
	t=[0];

	def animate(i):
		for i in range(0,25):
			compute_step(domain,dt)
		t[0]+=25*dt;
		exact=np.array(list(map(lambda xn:analytic(xn,t[0]),x)))
		line.set_data(x,exact)
		line2.set_data(x,domain[0])
		return line,line2
	anim=animation.FuncAnimation(fig,animate,frames=100,interval=int(dt*50000),blit=True)	
	plt.show()
plot_solution(square())
