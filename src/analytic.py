import numpy as np
from solvers import g

"""
Computes the value of the analytic solution of the dam break case with initial conditions:

h=1   for x<0
h=0.5 for x>=0
m=0
"""

def analytic_pointwise(x,t):
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

"""
This does the same as the above, except x can be an array.
"""
def analytic(x,t):
	return np.array(list(map(lambda xn:analytic_pointwise(xn,t),x)))
