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

  // Tolerance and max iterations
  const double       tolerance_increment = 1e-4; // o un valore adeguato per convergenza
  const unsigned int n_max_iter          = 1000;

  double       solution_increment_norm = tolerance_increment + 1;
  unsigned int n_iter                  = 0;

  // Relaxation coefficient as per Exercise 3.3
  const double lambda = 0.25;

  while (n_iter < n_max_iter && solution_increment_norm > tolerance_increment)
    {
      auto solution_1_increment = problem_1.get_solution();

      // Sequential DN: 
      // 1. Solve Domain 0 (Dirichlet on interface from Domain 1)
      problem_0.assemble();
      problem_0.apply_interface_dirichlet(problem_1);
      problem_0.solve();

      // 2. Solve Domain 1 (Neumann on interface from Domain 0)
      problem_1.assemble();
      problem_1.apply_interface_neumann(problem_0);
      problem_1.solve();

      // 3. Relaxation on Domain 1
      problem_1.apply_relaxation(solution_1_increment, lambda);

      // Check convergence
      solution_1_increment -= problem_1.get_solution();
      solution_increment_norm = solution_1_increment.l2_norm();

      std::cout << "iteration " << n_iter
                << " - solution increment = " << solution_increment_norm
                << std::endl;

      // Output for first and potentially last iteration (filtering done later or overwrite)
      problem_0.output(n_iter);
      problem_1.output(n_iter);

      ++n_iter;
    }

  return 0;
}