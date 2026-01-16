#include <iostream>

#include "Poisson1D.hpp"

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  constexpr unsigned int dim = Poisson1D::dim;

  const unsigned int N_el = 10;
  const unsigned int r    = 1;
  const double       kappa = 25.0;
  const double       alpha = 0.0;
  const double       beta  = 1.0;
  const double       dt    = 0.01;
  const double       T_final = 1.0;

  Poisson1D problem(N_el, r, kappa,  alpha, beta, dt, T_final);

  problem.run();

  return 0;
}
