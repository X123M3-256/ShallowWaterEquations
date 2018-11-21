import numpy as np
from solvers import lax_wendroff,solve

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

def do_smooth_convergence_test(solvers):
	print("Running smooth convergence test")

	#Compute high resolution solution
	time=0.05
	domain=(-1,1)
	courant=16.0
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

			x=[]
			u=[]
			for i in range(num_timesteps+1):
				(t,x,u)=next(solution)
			f.write(" %f"%l2_norm(x,u[0]-downsampled_high_res_h,(-1,1)))
		f.write("\n");
		print("Computed errors for dx=%f"%dx)
	f.close();
	print("Output written to convergence_smooth.txt")


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

def total_variation(u):
	return sum(np.abs(u[1:]-u[0:-1]))

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
	print("Output written to total_variation.txt")


