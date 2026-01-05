#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>

#include <iostream>
#include <filesystem>

#include "DiffusionReaction.hpp"

static constexpr unsigned int dim = DiffusionReaction::dim;

// Exact solution.
// [MODIFICA] Definizione della soluzione esatta u(x,y) = sin(pi/2 * x) * y
class ExactSolution : public Function<dim>
{
public:
  // Constructor.
  ExactSolution()
  {}

  // Evaluation.
  virtual double
  value(const Point<dim> &p,
        const unsigned int /*component*/ = 0) const override
  {
    return std::sin(M_PI / 2.0 * p[0]) * p[1];
  }

  // Gradient evaluation.
  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int /*component*/ = 0) const override
  {
    Tensor<1, dim> result;

    result[0] = M_PI / 2.0 * std::cos(M_PI / 2.0 * p[0]) * p[1];
    result[1] = std::sin(M_PI / 2.0 * p[0]);

    return result;
  }
};

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  ConvergenceTable table;

  // [MODIFICA] Mesh sizes h = 0.1, 0.05, 0.025, 0.0125 corrispondono a N = 10, 20, 40, 80
  const std::vector<unsigned int> N_el_values = {10, 20, 40, 80};
  
  // [MODIFICA] Elementi finiti P2 (r=2) come richiesto
  const unsigned int r = 2;

  // [MODIFICA] Coefficienti mu = 1, sigma = 1 come richiesto
  const auto mu = [](const Point<dim> & /*p*/) { return 1.0; };
  const auto sigma = [](const Point<dim> & /*p*/) { return 1.0; };

  // [MODIFICA] Forzante f calcolata da -Delta u + u con la soluzione esatta data
  const auto f = [](const Point<dim> &p) {
    return (1.0 + M_PI * M_PI / 4.0) * std::sin(M_PI / 2.0 * p[0]) * p[1];
  };

  const ExactSolution exact_solution;

  std::ofstream convergence_file("convergence.csv");
  convergence_file << "h,eL2,eH1" << std::endl;

  for (const auto &N_el : N_el_values)
    {
      const std::string mesh_file_name =
        "../mesh/mesh-square-" + std::to_string(N_el) + ".msh";

      if (!std::filesystem::exists(mesh_file_name))
        {
          std::cerr << "File mesh non trovato: " << mesh_file_name << std::endl;
          continue;
        }

      // [MODIFICA] Passiamo exact_solution come funzione di bordo 'g'
      DiffusionReaction problem(mesh_file_name, r, mu, sigma, f, exact_solution);

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

      convergence_file << h << "," << error_L2 << "," << error_H1 << std::endl;
    }

  table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);

  table.set_scientific("L2", true);
  table.set_scientific("H1", true);

  table.write_text(std::cout);

  return 0;
}