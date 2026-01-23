#include "Poisson2D.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  Poisson2D problem_0(0);
  Poisson2D problem_1(1);

  problem_0.setup();
  problem_1.setup();

  std::cout << "Setup completed" << std::endl;

  const double       tolerance_increment = 1e-6;
  const unsigned int n_max_iter          = 15;
  const double       lambda_0            = 1.0;  // Relaxation for domain 0
  const double       lambda_1            = 0.25; // Relaxation for domain 1

  double       solution_increment_norm = tolerance_increment + 1;
  unsigned int n_iter                  = 0;

  // Domain decomposition iterations
  while (n_iter < n_max_iter && solution_increment_norm > tolerance_increment)
    {
      Vector<double> solution_0_increment = problem_0.get_solution();
      Vector<double> solution_1_increment = problem_1.get_solution();

      // Solve subdomain 0 with Dirichlet interface condition
      problem_0.assemble();
      problem_0.apply_interface_dirichlet(problem_1);
      problem_0.solve();
      problem_0.apply_relaxation(solution_0_increment, lambda_0);

      // Solve subdomain 1 with Neumann interface condition
      problem_1.assemble();
      problem_1.apply_interface_neumann(problem_0);
      problem_1.solve();
      problem_1.apply_relaxation(solution_1_increment, lambda_1);

      // Check convergence
      solution_1_increment -= problem_1.get_solution();
      solution_increment_norm = solution_1_increment.l2_norm();

      std::cout << "DD iteration " << n_iter
                << " - solution increment = " << solution_increment_norm
                << std::endl;

      problem_0.output(n_iter);
      problem_1.output(n_iter);

      ++n_iter;
    }

  std::cout << "\nConverged in " << n_iter << " iterations" << std::endl;

  return 0;
}