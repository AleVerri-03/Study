#ifndef POISSON_1D_HPP
#define POISSON_1D_HPP

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class Poisson1D
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 1;

  // Constructor.
  Poisson1D(const unsigned int                              N_el_,
            const unsigned int                              r_,
            const double                                    kappa_,
            const double                                    alpha_,
            const double                                    beta_,
            const double                                    dt_,
            const double                                    T_final_);

public:
    void run();
    double compute_error(const VectorTools::NormType &norm_type,
                         const Function<dim>         &exact_solution) const;

private:
    void setup();
    void assemble();
    void solve();
    void output(unsigned int step) const;

    const unsigned int N_el;
    const unsigned int r;
    
    const double kappa;
    const double alpha_bc;
    const double beta_bc;
    const double delta_t;
    const double T_final;

    double time;

    Vector<double> solution_old;
    Triangulation<dim> mesh;


    // Number of elements.
    std::unique_ptr<FiniteElement<dim>> fe;
    std::unique_ptr<Quadrature<dim>> quadrature;
    DoFHandler<dim> dof_handler;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> system_rhs;
    Vector<double> solution;
};
#endif