#ifndef DIFFUSION_REACTION_HPP
#define DIFFUSION_REACTION_HPP

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class DiffusionReaction
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;

  // Helper function to format h without trailing zeros.
  static std::string format_h(double h)
  {
    std::ostringstream oss;
    oss << std::noshowpoint << h;
    return oss.str();
  }

  // Constructor.
  DiffusionReaction(const double h_,
                    const unsigned int &r_,
                    const std::function<double(const Point<dim> &)> &mu_,
                    const std::function<double(const Point<dim> &)> &sigma_,
                    const std::function<double(const Point<dim> &)> &f_,
                    const std::function<double(const Point<dim> &)> &g_)
    : mesh_file_name("../mesh/mesh-square-h" + format_h(h_) + ".msh")
    , r(r_)
    , mu(mu_)
    , sigma(sigma_)
    , f(f_)
    , g(g_)
  {}

  // Initialization.
  void
  setup();

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve();

  // Output.
  void
  output() const;

  // Compute the error against a given exact solution.
  double
  compute_error(const VectorTools::NormType &norm_type,
                const Function<dim>         &exact_solution) const;

  class ExactSolution : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override
    {
      return std::sin((M_PI / 2.0) * p[0]) * p[1];
    }

    virtual Tensor<1, dim> gradient(const Point<dim> &p,
                                    const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> grad;
      // du/dx = (pi/2) * cos(pi/2 * x) * y
      grad[0] = (M_PI / 2.0) * std::cos((M_PI / 2.0) * p[0]) * p[1];
      // du/dy = sin(pi/2 * x)
      grad[1] = std::sin((M_PI / 2.0) * p[0]);
      return grad;
    }
  };

protected:
  // Name of the mesh.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Diffusion coefficient.
  std::function<double(const Point<dim> &)> mu;

  // Reaction coefficient.
  std::function<double(const Point<dim> &)> sigma;

  // Forcing term.
  std::function<double(const Point<dim> &)> f;

  // Dirichlet boundary condition.
  std::function<double(const Point<dim> &)> g;

  // Triangulation.
  Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // Quadrature formula for boundary integrals.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // Sparsity pattern.
  SparsityPattern sparsity_pattern;

  // System matrix.
  SparseMatrix<double> system_matrix;

  // System right-hand side.
  Vector<double> system_rhs;

  // System solution.
  Vector<double> solution;
};

#endif