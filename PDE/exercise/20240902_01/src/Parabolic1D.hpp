#ifndef PARABOLIC_1D_HPP
#define PARABOLIC_1D_HPP

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
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

class Parabolic1D
{
public:
  static constexpr unsigned int dim = 1;

  Parabolic1D(const unsigned int &subdomain_id_)
    : subdomain_id(subdomain_id_)
  {}

  void
  setup();

  void
  assemble();

  void
  solve();

  void
  output(const unsigned int &iter) const;

  void
  apply_interface_dirichlet(const Parabolic1D &other);

  void
  apply_interface_neumann(const Parabolic1D &other);

  const Vector<double> &
  get_solution() const
  {
    return solution;
  }

  void
  apply_relaxation(const Vector<double> &old_solution, const double &lambda);

protected:
  std::map<types::global_dof_index, types::global_dof_index>
  compute_interface_map(const Parabolic1D &other) const;

  const unsigned int subdomain_id;

  Triangulation<dim> mesh;

  std::map<types::global_dof_index, Point<dim>> support_points;

  std::unique_ptr<FiniteElement<dim>> fe;

  std::unique_ptr<Quadrature<dim>> quadrature;

  DoFHandler<dim> dof_handler;

  SparsityPattern sparsity_pattern;

  SparseMatrix<double> system_matrix;

  Vector<double> system_rhs;

  Vector<double> solution;
};

#endif