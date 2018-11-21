import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from solvers import solve
from tests import dam_break
from analytic import analytic

#Plot numerical against analytic solution for dam break problem
def plot_dam_break(num_cells,dt,solver):
	solution=solve((-5,5),dam_break,num_cells,dt,solver)
	fig=plt.figure()
	ax=plt.axes(xlim=(-1,1),ylim=(0,1))
	ax.set_aspect("equal")
	line,=ax.plot([],[],lw=1)
	line2,=ax.plot([],[],lw=1)
	
	def animate(i):
		for i in range(50):
			(t,x,u)=next(solution)
		line.set_data(x,analytic(x,t))
		line2.set_data(x,u[0])
		return line,line2
	anim=animation.FuncAnimation(fig,animate,frames=100,interval=30,blit=True)	
	plt.show()
