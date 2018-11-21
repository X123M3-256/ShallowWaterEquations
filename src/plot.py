import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from solvers import solve

#Main plotting routine
def plot_solution(solution):

	fig=plt.figure()
	ax=plt.axes(xlim=(-1,1),ylim=(-1,1))
	ax.set_aspect("equal")
	line,=ax.plot([],[],lw=1)
	line2,=ax.plot([],[],lw=1)
	
	def animate(i):
		for i in range(1):
			(t,x,u)=next(solution)
		exact=np.array(list(map(lambda xn:analytic(xn,t),x)))
		line.set_data(x,exact)
		line2.set_data(x,u[0])
		return line,line2
	anim=animation.FuncAnimation(fig,animate,frames=100,interval=30,blit=True)	
	plt.show()
