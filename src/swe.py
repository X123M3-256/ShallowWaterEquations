import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

g=9.81



def dam_break(x):
	return np.stack([np.where(x<=0,1.0,0.5),np.zeros_like(x)])

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

def l2_norm(x,u,domain):
	(start,end)=domain
	dx=x[1]-x[0]
	u=np.where(np.logical_and(x>start,x<end),u,0.0)
	return np.sqrt(sum(dx*(u*u)))


def do_convergence_test(solvers):
	print("Running convergence test")
	time=0.2
	domain=(-5,5)
	
	f=open("convergence.txt","w")
	for num_cells in [2000,4000,8000,16000,32000,64000]:
		
		dx=(domain[1]-domain[0])/num_cells;
		num_timesteps=int(4.0*time/dx)
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

def f(u):
	return np.stack([u[1],(((u[1]*u[1])/u[0])+0.5*9.81*u[0]*u[0])]);

def lax_friedrich_flux(u,dx,dt):
	center_fluxes=f(u);
	return 0.5*(np.roll(center_fluxes,1,axis=1)+center_fluxes)-0.5*(dx/dt)*(u-np.roll(u,1,axis=1))

def lax_wendroff_flux(u,dx,dt):
	center_flux=f(u)
	u2=(0.5*dt/dx)*(np.roll(center_flux,1,axis=1)-center_flux)+0.5*(u+np.roll(u,1,axis=1))
	return f(u2)

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

plot_solution(solve((-5,5),dam_break,5000,0.0001,finite_volume(lax_friedrich_flux)))
plot_solution(solve((-5,5),dam_break,5000,0.0001,finite_volume(lax_wendroff_flux)))
do_convergence_test([finite_volume(lax_friedrich_flux),finite_volume(lax_wendroff_flux)])
