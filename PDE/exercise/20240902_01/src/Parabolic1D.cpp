#include "Parabolic1D.hpp"

void
Parabolic1D::setup()
{
  {
    std::vector<unsigned int> repetitions = {5};
    Point<dim> p1, p2;

    if (subdomain_id == 0)
      {
        p1 = Point<dim>(0.0);
        p2 = Point<dim>(0.5);
      }
    else
      {
        p1 = Point<dim>(0.5);
        p2 = Point<dim>(1.0);
      }

    GridGenerator::subdivided_hyper_rectangle(mesh, repetitions, p1, p2);
  }

  {
    fe         = std::make_unique<FE_SimplexP<dim>>(1);
    quadrature = std::make_unique<QGaussSimplex<dim>>(2);
  }

  {
    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    FE_SimplexP<dim> fe_linear(1);
    MappingFE        mapping(fe_linear);
    support_points = DoFTools::map_dofs_to_support_points(mapping, dof_handler);
  }

  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());
  }
}

void
Parabolic1D::assemble()
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_matrix = 0.0;
  system_rhs    = 0.0;

  const double mu     = 1.0;
  const double kappa  = 1.0;
  const double dt     = 0.1;
  const double f_val  = 1.0;

  const double reaction_coeff = kappa + 1.0 / dt;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) +=
                    (mu * fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) +
                     reaction_coeff * fe_values.shape_value(i, q) * fe_values.shape_value(j, q)) *
                    fe_values.JxW(q);
                }
              
              cell_rhs(i) += f_val * fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);

      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  {
    std::map<types::global_dof_index, double> boundary_values;
    
    if (subdomain_id == 0)
      {
        // Applica BC solo su x=0, NON sull'interfaccia (boundary_id=1)
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 0,  // ← Solo boundary 0
                                                 Functions::ZeroFunction<dim>(),
                                                 boundary_values);
      }
    // Per subdomain_id == 1, NON applicare nessuna BC qui
    // (verranno applicate da apply_interface_neumann)

    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, true);
  }
}

void
Parabolic1D::solve()
{
  SolverControl            solver_control(1000, 1e-12 * system_rhs.l2_norm());
  SolverCG<Vector<double>> solver(solver_control);

  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
}

void
Parabolic1D::output(const unsigned int &iter) const
{
  DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler, solution, "solution");
  data_out.build_patches();

  const std::string output_file_name = "output-" +
                                       std::to_string(subdomain_id) + "-" +
                                       std::to_string(iter) + ".vtk";
  std::ofstream output_file(output_file_name);
  data_out.write_vtk(output_file);
}

void
Parabolic1D::apply_interface_dirichlet(const Parabolic1D &other)
{
  const auto interface_map = compute_interface_map(other);

  std::map<types::global_dof_index, double> boundary_values;
  for (const auto &dof : interface_map)
    boundary_values[dof.first] = other.solution[dof.second];

  MatrixTools::apply_boundary_values(
    boundary_values, system_matrix, solution, system_rhs, false);
}

void
Parabolic1D::apply_interface_neumann(const Parabolic1D &other)
{
  const auto interface_map = compute_interface_map(other);

  // Calcola il residuo SENZA ri-assemblare
  Vector<double> residual(other.solution.size());
  other.system_matrix.vmult(residual, other.solution);
  residual -= other.system_rhs;
  residual *= -1.0;

  for (const auto &dof : interface_map)
    system_rhs[dof.first] += residual[dof.second];
}

std::map<types::global_dof_index, types::global_dof_index>
Parabolic1D::compute_interface_map(const Parabolic1D &other) const
{
  IndexSet current_interface_dofs;
  IndexSet other_interface_dofs;

  if (subdomain_id == 0)
    {
      current_interface_dofs =
        DoFTools::extract_boundary_dofs(dof_handler, ComponentMask(), {1});
      other_interface_dofs = DoFTools::extract_boundary_dofs(other.dof_handler,
                                                             ComponentMask(),
                                                             {0});
    }
  else
    {
      current_interface_dofs =
        DoFTools::extract_boundary_dofs(dof_handler, ComponentMask(), {0});
      other_interface_dofs = DoFTools::extract_boundary_dofs(other.dof_handler,
                                                             ComponentMask(),
                                                             {1});
    }

  std::map<types::global_dof_index, types::global_dof_index> interface_map;
  for (const auto &dof_current : current_interface_dofs)
    {
      const Point<dim> &p = support_points.at(dof_current);

      types::global_dof_index nearest = *other_interface_dofs.begin();
      for (const auto &dof_other : other_interface_dofs)
        {
          if (p.distance_square(other.support_points.at(dof_other)) <
              p.distance_square(other.support_points.at(nearest)))
            nearest = dof_other;
        }

      interface_map[dof_current] = nearest;
    }

  return interface_map;
}

void
Parabolic1D::apply_relaxation(const Vector<double> &old_solution,
                              const double         &lambda)
{
  solution *= lambda;
  solution.add(1.0 - lambda, old_solution);
}