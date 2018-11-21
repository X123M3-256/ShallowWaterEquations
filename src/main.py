from solvers import lax_friedrich,lax_wendroff,kurganov_petrova,solve
from tests import dam_break,do_convergence_test,do_smooth_convergence_test,do_conservation_test,do_total_variation_test
from plot import plot_solution


#Display solution of dam break case with each of the three schemes 
print("Displaying solution of dam break case using Lax-Friedrich scheme with dx=0.005 and dt=0.0001");
plot_solution(solve((-5,5),dam_break,5000,0.0001,lax_friedrich))
print("Displaying solution of dam break case using Lax-Wendroff scheme with dx=0.005 and dt=0.0001");
plot_solution(solve((-5,5),dam_break,5000,0.0001,lax_wendroff))
print("Displaying solution of dam break case using Kurganov-Petrova scheme with dx=0.005 and dt=0.0001");
plot_solution(solve((-5,5),dam_break,5000,0.0001,kurganov_petrova))

#Run test cases
solvers=[lax_friedrich,lax_wendroff,kurganov_petrova]
do_convergence_test(solvers)
do_smooth_convergence_test(solvers)
do_conservation_test(solvers)
do_total_variation_test(solvers)
