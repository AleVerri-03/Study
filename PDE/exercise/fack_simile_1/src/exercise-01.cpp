#include "DiffusionReaction.hpp"

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  constexpr unsigned int dim = DiffusionReaction::dim;
  const unsigned int degree        = 2;

  const auto mu = [](const Point<dim> & /*p*/) {
    return 1.0;
  };

  const auto sigma = [](const Point<dim> & /*p*/) { return 1.0; };

  const auto f = [](const Point<dim> &p) { 
    return (1.0 + M_PI * M_PI / 4.0) * 
           std::sin((M_PI / 2.0) * p[0]) * p[1]; 
  };

  const auto g = [](const Point<dim> &p) { 
    return std::sin((M_PI / 2.0) * p[0]); 
  };

  std::vector<double> h_values = {0.1, 0.05, 0.025, 0.0125};
  std::vector<double> errors_L2;
  std::vector<double> errors_H1;

  for (int i=0; i<h_values.size(); i++) {
    double h = h_values[i];
    DiffusionReaction problem(h, degree, mu, sigma, f, g);
    std::cout << "================ h = " << h << " ================" << std::endl;
    problem.setup();
    problem.assemble();
    problem.solve();
    problem.output();

    const double error_L2 = problem.compute_error(
      VectorTools::L2_norm, DiffusionReaction::ExactSolution());
    std::cout << "L2 error: " << error_L2 << std::endl;
    errors_L2.push_back(error_L2);

    const double error_H1 = problem.compute_error(
      VectorTools::H1_norm, DiffusionReaction::ExactSolution());
    std::cout << "H1 error: " << error_H1 << std::endl;
    errors_H1.push_back(error_H1);
    
    if(i>0){
      // EOC
      double eoc_L2 = std::log(errors_L2[i-1]/errors_L2[i]) / std::log(2.0);
      double eoc_H1 = std::log(errors_H1[i-1]/errors_H1[i]) / std::log(2.0);
      std::cout << "EOC L2: " << eoc_L2 << std::endl;
      std::cout << "EOC H1: " << eoc_H1 << std::endl;
    }
  }

  return 0;
}