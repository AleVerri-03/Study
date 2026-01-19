#ifndef HEAT_HPP
#define HEAT_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class Heat
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 1;

  // Initial condition.
  class FunctionU0 : public Function<dim>
  {
  public:
    // Constructor.
    FunctionU0() = default;

    // Evaluation of the function.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution() : Function<dim>() {} // Costruttore di default

    virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      const double PI = 3.14159265358979323846;
      double t = this->get_time(); // Recupera il tempo impostato
      return std::sin(PI / 2.0 * p[0]) * std::sin(PI / 2.0 * t);
    }
  };

  // Constructor.
  Heat(const unsigned int                              &N_elements_,
       const unsigned int                              &r_,
       const double                                    &T_,
       const double                                    &theta_,
       const double                                    &delta_t_,
       const double                                    &epsilon_,
       const double                                    &k_,
       const std::function<double(const Point<dim> &)> &b_,
       const std::function<double(const Point<dim> &, const double &)> &f_)
    : N_elements(N_elements_)
    , r(r_)
    , T(T_)
    , theta(theta_)
    , delta_t(delta_t_)
    , epsilon(epsilon_)
    , k(k_)
    , b(b_)
    , f(f_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , mesh(MPI_COMM_WORLD)
    , pcout(std::cout, mpi_rank == 0)
  {}

  // Run the time-dependent simulation.
  double
  run();

protected:
  void setup();
  void assemble();
  void solve_linear_system();
  void output() const;

  double compute_error();
  const unsigned int N_elements;
  const unsigned int r;
  const double T;
  const double theta;
  const double delta_t;
  double time = 0.0;
  unsigned int timestep_number = 0;

  const double epsilon;
  const double k;
  std::function<double(const Point<dim> &)> b;
  std::function<double(const Point<dim> &, const double &)> f;

  const unsigned int mpi_size;
  const unsigned int mpi_rank;

  parallel::fullydistributed::Triangulation<dim> mesh;
  std::unique_ptr<FiniteElement<dim>> fe;
  std::unique_ptr<Quadrature<dim>> quadrature;
  DoFHandler<dim> dof_handler;
  
  AffineConstraints<double> constraints;

  TrilinosWrappers::SparseMatrix system_matrix;
  TrilinosWrappers::MPI::Vector system_rhs;
  TrilinosWrappers::MPI::Vector solution_owned;
  TrilinosWrappers::MPI::Vector solution;

  ConditionalOStream pcout;
};

#endif