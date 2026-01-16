#include <deal.II/base/convergence_table.h>

#include <iostream>

#include "Poisson1D.hpp"


class ExactSolution : public Function<1>
{
public:
  virtual double value(const Point<1> &p, const unsigned int = 0) const override
  {
    return p[0] * (1.0 - p[0]); // x - x^2
  }

  virtual Tensor<1, 1> gradient(const Point<1> &p, const unsigned int = 0) const override
  {
    Tensor<1, 1> result;
    result[0] = 1.0 - 2.0 * p[0]; // 1 - 2x
    return result;
  }
};

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  const unsigned int dim = 1;

  ConvergenceTable table;

  const std::vector<unsigned int> N_el_values = {10};
  const unsigned int              degree      = 2;
  const auto mu = [](const Point<dim> & /*p*/) { return 1.0; };
  const auto f  = [](const Point<dim> &p) {
    return p[0] * p[0] + 2.0;
  };

  const ExactSolution exact_solution;

  for (const unsigned int &N_el : N_el_values)
    {
      Poisson1D problem(N_el, degree, f); // MODIFICA (instanziamo il problema)

      problem.setup();
      problem.assemble();
      problem.solve();
      problem.output();

      const double h = 1.0 / N_el;

      const double error_L2 =
        problem.compute_error(VectorTools::L2_norm, exact_solution);
      const double error_H1 =
        problem.compute_error(VectorTools::H1_norm, exact_solution);

      table.add_value("h", h);
      table.add_value("L2", error_L2);
      table.add_value("H1", error_H1);

    }

  // Calcolo rate convergenza
  table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);

  table.set_scientific("L2", true);
  table.set_scientific("H1", true);

  table.write_text(std::cout);

  return 0;
}
