#include "Heat.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = Heat::dim;
  const double PI = 3.14159265358979323846;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const double mu = 1.0;
  const auto b_func = [](const Point<dim> & /*p*/) { 
    Tensor<1, dim> b;
    b[0] = 0.1;
    b[1] = 0.2;
    return b; 
  };

  const auto f_func = [PI](const Point<dim> &p, const double &t) {
    const double x = p[0];
    const double y = p[1];
    
    const double sin_pix = std::sin(PI * x);
    const double cos_pix = std::cos(PI * x);
    const double sin_2piy = std::sin(2.0 * PI * y);
    const double cos_2piy = std::cos(2.0 * PI * y);
    const double sin_2pit = std::sin(2.0 * PI * t);
    const double cos_2pit = std::cos(2.0 * PI * t);
    
    double term1 = sin_pix * sin_2piy * (2.0 * PI * cos_2pit + 5.0 * PI * PI * sin_2pit);
    double term2 = PI * sin_2pit * (0.1 * cos_pix * sin_2piy + 0.4 * sin_pix * cos_2piy);
    
    return term1 + term2;
  };

  std::vector<double> deltas = {0.05};
  std::vector<double> errors;

  const std::string mesh_file = "../mesh/square.msh";
  const unsigned int degree     = 2;
  const double       T          = 1.0;
  const double       theta      = 0.5;

  std::cout << "dt\t\tError L2\tEOC" << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  for (unsigned int i = 0; i < deltas.size(); ++i)
  {
      double dt = deltas[i];

      Heat problem(mesh_file, degree, T, theta, dt, mu, b_func, f_func);
      
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