from solvers import lax_friedrich,lax_wendroff,kurganov_petrova,solve
from tests import dam_break,do_convergence_test,do_smooth_convergence_test,do_conservation_test,do_total_variation_test
from plot import plot_dam_break


#Display solution of dam break problem with each of the three schemes 
num_cells=5000
dt=0.0001
print("Displaying solution of dam break case using Lax-Friedrich scheme with dx=0.005 and dt=0.0001");
plot_dam_break(num_cells,dt,lax_friedrich)
print("Displaying solution of dam break case using Lax-Wendroff scheme with dx=0.005 and dt=0.0001");
plot_dam_break(num_cells,dt,lax_wendroff)
print("Displaying solution of dam break case using Kurganov-Petrova scheme with dx=0.005 and dt=0.0001");
plot_dam_break(num_cells,dt,kurganov_petrova)

#Run test cases
solvers=[lax_friedrich,lax_wendroff,kurganov_petrova]
do_convergence_test(solvers)
do_smooth_convergence_test(solvers)
do_conservation_test(solvers)
do_total_variation_test(solvers)
