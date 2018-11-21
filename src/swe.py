import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
g=9.81



def dam_break(x):
	return np.stack([np.where(x<=0,1.0,0.5),np.zeros_like(x)])

def hump(x):
	return np.stack([np.where(np.abs(x)<=0.5,0.5+0.1*(1+np.cos(2.0*np.pi*x)),0.5),np.zeros_like(x)])

def analytic(x,t):
	x+=0.5
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

def integrate(u,dx):
	return sum(dx*u)
	

def l2_norm(x,u,domain):
	(start,end)=domain
	dx=x[1]-x[0]
	u=np.where(np.logical_and(x>start,x<end),u,0.0)
	return np.sqrt(integrate(u*u,dx))


def do_convergence_test(solvers):
	print("Running convergence test")
	time=0.2
	domain=(-2,2)
	courant=4.0;
	
	f=open("convergence.txt","w")
	for num_cells in [500,750,1000,1500,2000,3000,4000,6000,8000,12000]:
		
		dx=(domain[1]-domain[0])/num_cells;
		num_timesteps=int(courant*num_cells*time/(domain[1]-domain[0]))
		dt=time/num_timesteps
		
		f.write("%f"%dx);
		for solver in solvers:
			solution=solve(domain,dam_break,num_cells,dt,solver);

			x=[]
			u=[]
			for i in range(num_timesteps+1):
				(t,x,u)=next(solution)
			exact=np.array(list(map(lambda xn:analytic(xn,time),x)))
			f.write(" %f"%l2_norm(x,u[0]-exact,(-1,1)))
		f.write("\n");
		print("Computed errors for dx=%f"%dx)
	f.close();
	print("Output written to convergence.txt")

def do_conservation_test(solvers):
	print("Running conservation test")
	domain=(-1,1)
	num_cells=1000
	num_timesteps=2000
	dx=(domain[1]-domain[0])/num_cells;
	dt=0.0005
	
	f=open("conservation.txt","w")
	for solver in solvers:
		solution=solve(domain,hump,num_cells,dt,solver);

		x=[]
		u=[]
		for i in range(num_timesteps+1):
			(t,x,u)=next(solution)
			f.write("%f %f %f %f\n"%(t,integrate(u[0],dx),integrate(u[1],dx),integrate(0.5*u[1]*u[1]/u[0]+0.5*g*u[0]*u[0],dx)))
		f.write("\n");
	f.close();
	print("Output written to conservation.txt")

def f(u):
	return np.stack([u[1],(((u[1]*u[1])/u[0])+0.5*9.81*u[0]*u[0])]);

def lax_friedrich_flux(u,dx,dt):
	center_fluxes=f(u);
	return 0.5*(np.roll(center_fluxes,1,axis=1)+center_fluxes)-0.5*(dx/dt)*(u-np.roll(u,1,axis=1))

def lax_wendroff_flux(u,dx,dt):
	center_flux=f(u)
	u2=(0.5*dt/dx)*(np.roll(center_flux,1,axis=1)-center_flux)+0.5*(u+np.roll(u,1,axis=1))
	return f(u2)

def minmod(u,v):
	return np.where(u*v<0,0.0,np.where(np.abs(u)<np.abs(v),u,v))


def compute_velocity(u):
	return u[1]/u[0]

def compute_wave_speed(h):
	return np.sqrt(g*h) 

def kurganov_petrova_flux(u,dx,dt):
	#Compute numerical derivatives using slope limiter
	left_slope=(u-np.roll(u,1,axis=1))/dx
	right_slope=(np.roll(u,-1,axis=1)-u)/dx
	u_x=np.stack([minmod(left_slope[0],right_slope[0]),minmod(left_slope[1],right_slope[1])])
	#Compute left and right values of u
	u_right=u+0.5*dx*u_x
	u_left=np.roll(u,1,axis=1)-0.5*dx*np.roll(u_x,1,axis=1)
	#Compute velocities
	v_left=compute_velocity(u_left)
	v_right=compute_velocity(u_right)	
	#Compute local propogation speeds
	c_left=compute_wave_speed(u_left[0]);
	c_right=compute_wave_speed(u_right[0]);
	a_left=np.minimum(np.minimum(u_right-c_right,u_left-c_left),0);
	a_right=np.maximum(np.maximum(u_right+c_right,u_left+c_left),0);
	return ((a_right*f(u_left)-a_left*f(u_right))+a_left*a_right*(u_right-u_left))/(a_right-a_left)

def kurganov_petrova_derivative(u,dx,dt):
	flux=kurganov_petrova_flux(u,dx,dt)
	return (flux-np.roll(flux,-1,axis=1))/dx;

def kurganov_petrova(u,dx,dt):
	u1=u+dt*kurganov_petrova_derivative(u,dx,dt);
	u2=0.75*u+0.25*(u1+dt*kurganov_petrova_derivative(u1,dx,dt));
	u[:]=u/3.0+(2.0/3.0)*(u2+dt*kurganov_petrova_derivative(u2,dx,dt))

def finite_volume(flux_func):
	def compute_step(u,dx,dt):
		flux=flux_func(u,dx,dt)
		u+=(dt/dx)*(flux-np.roll(flux,-1,axis=1));
	return compute_step



def solve(domain,initial_condition,num_cells,dt,solver):
	(start,end)=domain
	#Compute cell width
	dx=(end-start)/num_cells
	#Compute coordinates of cell centers
	x=np.linspace(start+dx/2,end-dx/2,num_cells)
	#Initialize solution with initial condition	
	t=0
	u=initial_condition(x)
	#Generate sequence of timesteps
	while(True):
		yield (t,x,u)
		solver(u,dx,dt);
		t+=dt			


#Main plotting routine
def plot_solution(solution):

	fig=plt.figure()
	ax=plt.axes(xlim=(-1,1),ylim=(-1,1))
	ax.set_aspect("equal")
	line,=ax.plot([],[],lw=1)
	line2,=ax.plot([],[],lw=1)
	
	def animate(i):
		for i in range(50):
			(t,x,u)=next(solution)
		exact=np.array(list(map(lambda xn:analytic(xn,t),x)))
		line.set_data(x,exact)
		line2.set_data(x,u[0])
		return line,line2
	anim=animation.FuncAnimation(fig,animate,frames=100,interval=30,blit=True)	
	plt.show()

solvers=[finite_volume(kurganov_petrova_flux),finite_volume(lax_wendroff_flux),kurganov_petrova]
do_convergence_test(solvers)
do_conservation_test(solvers)
