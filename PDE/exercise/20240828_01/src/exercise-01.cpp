#include "Heat.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = Heat::dim;
  const double PI = 3.14159265358979323846;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const double epsilon_val = 1.0;
  const double k_val       = 1.0;
  const auto b_func = [](const Point<dim> &p) { 
    return p[0] - 1.0; 
  };

  const auto f_func = [PI](const Point<dim> &p, const double &t) {
    const double x = p[0];
    const double sin_pix_2 = std::sin(PI / 2.0 * x);
    const double cos_pix_2 = std::cos(PI / 2.0 * x);
    const double sin_pit_2 = std::sin(PI / 2.0 * t);
    const double cos_pit_2 = std::cos(PI / 2.0 * t);

    double term1 = (PI / 2.0) * sin_pix_2 * cos_pit_2;
    double term2 = (std::pow(PI, 2) / 4.0 + 2.0) * sin_pix_2 * sin_pit_2;
    double term3 = (PI / 2.0) * (x - 1.0) * cos_pix_2 * sin_pit_2;

    return term1 + term2 + term3;
  };

  std::vector<double> deltas = {0.1, 0.05, 0.025, 0.0125};
  std::vector<double> errors;

  const unsigned int N_elements = 40;
  const unsigned int degree     = 2;
  const double       T          = 1.0;
  const double       theta      = 0.5;

  std::cout << "dt\t\tError L2\tEOC" << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  for (unsigned int i = 0; i < deltas.size(); ++i)
  {
      double dt = deltas[i];

      Heat problem(N_elements, degree, T, theta, dt, epsilon_val, k_val, b_func, f_func);
      
      // Eseguiamo e catturiamo l'errore
      double error = problem.run();
      errors.push_back(error);

      // Calcolo EOC (Experimental Order of Convergence)
      double eoc = 0.0;
      if (i > 0)
      {
          // formula: log(err_old / err_new) / log(dt_old / dt_new)
          // Dato che dt si dimezza sempre, log(2) al denominatore
          eoc = std::log(errors[i-1] / errors[i]) / std::log(deltas[i-1] / deltas[i]);
      }

      std::cout << dt << "\t" << error << "\t" << (i > 0 ? std::to_string(eoc) : "-") << std::endl;
  }

  return 0;
}