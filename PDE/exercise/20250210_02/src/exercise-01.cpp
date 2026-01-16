#include "Stokes.hpp"

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  // Usa il mesh 2D quadrato dalla cartella lab-02 del professore.
  const std::string  mesh_file_name  = "../../../prof_files/lab-02/mesh/mesh-square-10.msh";
  const unsigned int degree_velocity = 2;
  const unsigned int degree_pressure = 1;

  Stokes problem(mesh_file_name, degree_velocity, degree_pressure);

  problem.setup();
  // MODIFICA: Chiamata al metodo run per avviare la simulazione tempo-dipendente
  problem.run();

  return 0;
}