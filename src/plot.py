import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from solvers import solve
from analytic import analytic

#Main plotting routine
def plot_solution(solution):

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
