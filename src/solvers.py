import numpy as np

#The value of acceleration due to gravity
g=9.81

"""
This function compute the flux of fluid volume and momentum from the height and discharge.
"""
def f(u):
	return np.stack([u[1],(((u[1]*u[1])/u[0])+0.5*g*u[0]*u[0])]);

"""
This computes the numerical flux used in the Lax-Friedrich scheme
"""
def lax_friedrich_flux(u,dx,dt):
	center_fluxes=f(u);
	return 0.5*(np.roll(center_fluxes,1,axis=1)+center_fluxes)-0.5*(dx/dt)*(u-np.roll(u,1,axis=1))

"""
This computes the numerical_flux used in the Lax-Wendroff scheme
"""
def lax_wendroff_flux(u,dx,dt):
	center_flux=f(u)
	u2=(0.5*dt/dx)*(np.roll(center_flux,1,axis=1)-center_flux)+0.5*(u+np.roll(u,1,axis=1))
	return f(u2)

"""
This evaluates the minmod slope limiter used to prevent spurious oscillations from developing 
in the solution.

The values left_slope and right_slope should be numerical derivative computed using backward and forward differencing respectively.
"""
def minmod(left_slope,right_slope):
	return np.where(left_slope*right_slope<0,0.0,np.where(np.abs(left_slope)<np.abs(right_slope),left_slope,right_slope))

"""
Compute the fluid velocity from the fluid depth and discharge.

Note that in the original Kurganov-Petrova scheme, this is implemented differently
in order to avoid the singularity when h=0. However, I don't test cases where this
occurs, and the original formula was causing problems.
"""
def compute_velocity(u):
	return u[1]/u[0]

"""
This computes the speed of surface gravity waves from the fluid depth
""" 
def compute_wave_speed(h):
	return np.sqrt(g*h) 


"""
This computes the numerical flux used in the Kurganov-Petrova scheme

I suspect there may be issues with this implementation as it isn't giving the 
convergence rate that it ought to have.
"""
def kurganov_petrova_flux(u,dx,dt):
	#Compute numerical derivatives using minmod slope limiter
	left_slope=(u-np.roll(u,1,axis=1))/dx
	right_slope=(np.roll(u,-1,axis=1)-u)/dx
	u_x=np.stack([minmod(left_slope[0],right_slope[0]),minmod(left_slope[1],right_slope[1])])
	#Compute the left and right values of u at the cell edges
	u_right=u+0.5*dx*u_x
	u_left=np.roll(u,1,axis=1)-0.5*dx*np.roll(u_x,1,axis=1)
	#Compute velocities at the cell edges
	v_left=compute_velocity(u_left)
	v_right=compute_velocity(u_right)	
	#Compute wave speeds at the cell edges
	c_left=compute_wave_speed(u_left[0]);
	c_right=compute_wave_speed(u_right[0]);
	#Compute the minimum and maximum propogation speeds at the cell edges
	a_left=np.minimum(np.minimum(u_right-c_right,u_left-c_left),0);
	a_right=np.maximum(np.maximum(u_right+c_right,u_left+c_left),0);
	#Compute Kurganov-Petrova flux using the calculated propogation speeds
	return ((a_right*f(u_left)-a_left*f(u_right))+a_left*a_right*(u_right-u_left))/(a_right-a_left)


"""
Time stepping schemes.

These functions take as input a function which computes the numerical flux, and return a function which computes a timestep.
"""

"""
Implements the first order Euler method
"""
def euler(flux_func):
	def compute_step(u,dx,dt):
		flux=flux_func(u,dx,dt)
		return u+(dt/dx)*(flux-np.roll(flux,-1,axis=1));
	return compute_step

"""
Implements the third order strong stability preserving Runge-Kutta method
"""
def ssp_rk3(flux_func):
	euler_step=euler(flux_func)
	def compute_step(u,dx,dt):
		u1=euler_step(u,dx,dt);
		u2=0.75*u+0.25*euler_step(u1,dx,dt);
		return u/3.0+(2.0/3.0)*euler_step(u2,dx,dt)
	return compute_step



"""
Define a generator which takes as input the problem domain, initial condition, number of cells and timestep to use, and a numerical solver, and produces the sequence of solutions at each successive timestep
"""
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
		u=solver(u,dx,dt);
		t+=dt			
"""
For convenience, define the three combinations of space and time discretizations that we look at
"""
lax_friedrich=euler(lax_friedrich_flux)
lax_wendroff=euler(lax_wendroff_flux)
kurganov_petrova=ssp_rk3(kurganov_petrova_flux)



