import numpy as np

from solvers import g,lax_wendroff,solve
from analytic import analytic


"""
Initial conditions for the test cases we will consider - one smooth and one discontinous

These functions take as input an array of x coordinates and return the initial values
of the fluid height and discharge at those points
"""
def dam_break(x):
	return np.stack([np.where(x<=0,1.0,0.5),np.zeros_like(x)])

def hump(x):
	return np.stack([np.where(np.abs(x)<=0.5,0.5+0.1*(1+np.cos(2.0*np.pi*x)),0.5),np.zeros_like(x)])

"""
Compute a numerical quadrature by the midpoint rule.
"""
def integrate(u,dx):
	return sum(dx*u)
	
"""
Compute the L2 norm
"""
def l2_norm(u,dx):
	return np.sqrt(integrate(u*u,dx))

"""
Compute dam break solution at time t=0.2

This function computes the solution of the dam break problem at time t=0.2 using each
numerical scheme, so that they can be plotted. The results are written to the file dam_break.txt
"""

def do_dam_break_computation(solvers):
	print("Computing solutions to dam break problem")
	domain=(-2,2)
	num_cells=1000
	start=num_cells//4
	end=num_cells-start
	num_timesteps=400
	dx=(domain[1]-domain[0])/num_cells;
	dt=0.0005
	
	f=open("dam_break.txt","w")

	x=np.linspace(domain[0]+0.5*dx,domain[1]-0.5*dx,num_cells)
	exact=analytic(x,num_timesteps*dt)
	solutions=[]
	for solver in solvers:
		solution=solve(domain,dam_break,num_cells,dt,solver);
		for i in range(num_timesteps+1):
			(t,x,u)=next(solution)
		solutions.append(u[0])

	for i in range(start,end):
		f.write("%f %f "%(x[i],exact[i]))
		for solution in solutions:
			f.write(" %f"%solution[i])
		f.write("\n");
	f.close();
	print("Output written to dam_break.txt\n")




"""
Test the convergence rate of a list of solvers on the dam break case

This function computes the numerical solution with each solver for a number of different
values of dx and dt such that the Courant number is constant. The L2 norms of the errors
are written to the file convergence.txt
"""
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

			for i in range(num_timesteps+1):
				(t,x,u)=next(solution)
			f.write(" %f"%l2_norm(np.where(np.logical_and(x>-1,x<1),u[0]-analytic(x,time),0),dx))
		f.write("\n");
		print("Computed errors for dx=%f"%dx)
	f.close();
	print("Output written to convergence.txt\n")

"""
Test the convergence rate of a list of solvers on the smooth initial condition

Since no analytic solution is availabe in this case, the L2 norms are estimated using a
numerical solution computed with a very high resolution. The results are written to the
file convergence_smooth.txt
"""
def do_smooth_convergence_test(solvers):
	print("Running smooth convergence test")

	#Compute high resolution solution
	time=0.05
	domain=(-1,1)
	courant=4.0
	high_res_num_cells=21870*3
	high_res_num_timesteps=int(courant*high_res_num_cells*time/(domain[1]-domain[0]))
	high_res_dt=time/high_res_num_timesteps
	high_res_solution=solve(domain,hump,high_res_num_cells,high_res_dt,lax_wendroff);
	high_res_u=[]
	for i in range(high_res_num_timesteps+1):
		(t,x,high_res_u)=next(high_res_solution)
	print("Computed high resolution solution (%d points)"%(high_res_num_cells))

	downsampled_high_res_h=high_res_u[0]
	downsampled_high_res_h=downsampled_high_res_h[1::3]

	f=open("convergence_smooth.txt","w")
	for steps in [9,27,27*3,27*9,27*27,27*27*3]:#,27*3]:
		num_cells=high_res_num_cells//steps	
		dx=(domain[1]-domain[0])/num_cells;
		num_timesteps=int(courant*num_cells*time/(domain[1]-domain[0]))
		dt=time/num_timesteps
		#Downsample high res solution	
		downsampled_high_res_h=downsampled_high_res_h[1::3]

		f.write("%f"%dx);
		for solver in solvers:
			solution=solve(domain,hump,num_cells,dt,solver);

			for i in range(num_timesteps+1):
				(t,x,u)=next(solution)
			f.write(" %f"%l2_norm(u[0]-downsampled_high_res_h,dx))
		f.write("\n");
		print("Computed errors for dx=%f"%dx)
	f.close();
	print("Output written to convergence_smooth.txt\n")

"""
Test how well the numerical solvers preserve conserved quantities

This function computes the solution of the cosine bell problem with each 
numerical scheme, and computes values of the mass, momentum, and energy at
 each time step. The results are written to the file conservation.txt
"""
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

		for i in range(num_timesteps+1):
			(t,x,u)=next(solution)
			f.write("%f %f %f %f\n"%(t,integrate(u[0],dx),integrate(u[1],dx),integrate(0.5*u[1]*u[1]/u[0]+0.5*g*u[0]*u[0],dx)))
		f.write("\n");
	f.close();
	print("Output written to conservation.txt\n")

"""
Compute the total variation of u
"""
def total_variation(u):
	return sum(np.abs(u[1:]-u[0:-1]))

"""
Test whether the numerical methods introduce spurious oscillations

This function computes the numerical solution of the dam break problem with each
numerical scheme, and computes the total variation at each time step. The results
are written to the file total_variation.txt
"""

def do_total_variation_test(solvers):
	print("Running total variation test")
	domain=(-2,2)
	num_cells=1000
	start=num_cells//4
	end=num_cells-start
	num_timesteps=200
	dx=(domain[1]-domain[0])/num_cells;
	dt=0.0005
	
	f=open("total_variation.txt","w")
	solutions=[]
	for solver in solvers:
		solutions.append(solve(domain,dam_break,num_cells,dt,solver));

	for t in np.linspace(0,dt*num_timesteps,num_timesteps+1):
		f.write("%f"%t)
		for solution in solutions:
			(t2,x,u)=next(solution)
			f.write(" %f"%(total_variation(u[0][start:end])))
		f.write("\n");
	f.close();
	print("Output written to total_variation.txt\n")

