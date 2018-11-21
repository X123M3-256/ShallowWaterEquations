import numpy as np

g=9.81


def f(u):
	return np.stack([u[1],(((u[1]*u[1])/u[0])+0.5*g*u[0]*u[0])]);

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

lax_friedrich=finite_volume(lax_friedrich_flux)
lax_wendroff=finite_volume(lax_wendroff_flux)

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


