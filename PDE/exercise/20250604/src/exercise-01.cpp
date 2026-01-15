#include <iostream>
#include <string>

#include "Poisson1D.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = Poisson1D::dim;

  // Parameters defaults
  double epsilon = 1.0;
  unsigned int N_el = 10; // h = 0.1

  // Parse command line arguments
  if (argc > 1)
    epsilon = std::stod(argv[1]);
  if (argc > 2)
    N_el = std::stoi(argv[2]);

  std::cout << "Solving for epsilon = " << epsilon << " and h = " << 1.0/N_el << " (N_el = " << N_el << ")" << std::endl;

  const unsigned int r    = 1;

  // [MODIFICA] Definizione del coefficiente di diffusione e del termine forzante
  // Diffusion coefficient: epsilon * (1 + x^2)
  const auto         mu   = [epsilon](const Point<dim> &p) {
    return epsilon * (1.0 + p[0] * p[0]);
  };

  // Forcing term: f = 0
  const auto         f    = [](const Point<dim> & /*p*/) {
    return 0.0;
  };

  Poisson1D problem(N_el, r, mu, f);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}
