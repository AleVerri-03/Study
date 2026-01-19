#include "Heat.hpp"

int main(int argc, char *argv[])
{
  constexpr unsigned int dim = Heat::dim;
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  // Funzione forzante f (la stessa di prima)
  const auto f = [](const Point<dim> &p, const double &t) {
    const double x = p[0];
    const double pi_2 = M_PI / 2.0;
    return pi_2 * std::sin(pi_2 * x) * std::cos(pi_2 * t)
         + (M_PI * M_PI / 4.0 - 1.0) * std::sin(pi_2 * x) * std::sin(pi_2 * t)
         + pi_2 * std::cos(pi_2 * x) * std::sin(pi_2 * t);
  };

  // Lista dei Delta t da testare
  std::vector<double> deltas = {0.1, 0.05, 0.025, 0.0125};
  
  // Per memorizzare l'errore precedente e calcolare l'ordine
  double previous_error = 0.0;

  std::cout << "  dt      |   Error L2   |  Order" << std::endl;
  std::cout << "------------------------------------" << std::endl;

  for (unsigned int i = 0; i < deltas.size(); ++i)
  {
      double dt = deltas[i];

      // Istanziamo il problema con il dt corrente
      // N=40 Ã¨ gestito internamente in setup() come abbiamo modificato prima
      Heat problem("", 2, 1.0, 0.5, dt, 1.0, 1.0, 1.0, f);

      // Disabilitiamo l'output VTU per risparmiare tempo e spazio, 
      // oppure lo lasciamo se vogliamo controllare.
      // (Se vuoi disabilitarlo, commenta output() dentro run() o aggiungi un flag).
      problem.run();

      // Prepariamo la soluzione esatta al tempo finale T=1.0
      ExactSolution sol_ex;
      sol_ex.set_time(1.0);

      double error = problem.compute_error(sol_ex);

      // Calcolo ordine di convergenza sperimentale (EOC)
      // p = log2( error_old / error_new ) poichÃ© dimezziamo dt
      double order = 0.0;
      if (i > 0)
      {
          order = std::log2(previous_error / error);
      }

      std::cout << std::fixed << std::setprecision(4) << dt << "  |  " 
                << std::scientific << std::setprecision(4) << error << "  |  " 
                << std::fixed << std::setprecision(2) << order << std::endl;

      previous_error = error;
  }

  return 0;
}