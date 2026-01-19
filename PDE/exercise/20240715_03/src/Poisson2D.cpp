#include "Poisson2D.hpp"

void
Poisson2D::setup()
{
  // Create the mesh (1D).
  // Problem domain is (0, 1). Gamma = 0.75.
  // Subdomain 0: (0, 0.75). Subdomain 1: (0.75, 1).
  // We choose N based on roughly equal spacing h.
  // Let's use 30 elements for dom 0 and 10 for dom 1 (total 40).
  {
    if (subdomain_id == 0)
      {
         GridGenerator::subdivided_hyper_cube(mesh, 30, 0.0, 0.75);
      }
    else
      {
         GridGenerator::subdivided_hyper_cube(mesh, 10, 0.75, 1.0);
      }
    // In 1D GridGenerator: Left boundary is 0, Right boundary is 1.
  }

  // Initialize the finite element space.
  {
    // Use FE_Q for segments/hypercubes
    fe         = std::make_unique<FE_Q<dim>>(1);
    quadrature = std::make_unique<QGauss<dim>>(2);
  }

  // Initialize the DoF handler.
  {
    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // Compute support points for the DoFs.
    FE_Q<dim> fe_linear(1);
    MappingFE<dim> mapping(fe_linear);
    support_points = DoFTools::map_dofs_to_support_points(mapping, dof_handler);
  }

  // Initialize the linear system.
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
Poisson2D::assemble()
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

  // Problem parameters from Exercise 3.3
  const double epsilon = 1.0;
  const double b       = 2.0;
  const double c       = 1.0;
  const double f       = 1.0;

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
                  // Weak form of -eps*u'' + b*u' + c*u = f
                  // (eps*u', v') + (b*u', v) + (c*u, v)
                  // In 1D, shape_grad returns Tensor<1,1>, so we extract [0] for scalar operations
                  cell_matrix(i, j) +=
                    (epsilon * fe_values.shape_grad(j, q) * fe_values.shape_grad(i, q) +
                     b * fe_values.shape_grad(j, q)[0] * fe_values.shape_value(i, q) +
                     c * fe_values.shape_value(j, q) * fe_values.shape_value(i, q)) *
                    fe_values.JxW(q);
                }
              
              cell_rhs(i) += f * fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);

      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  // Boundary conditions.
  // Physical boundaries are at x=0 and x=1.
  // For Subdomain 0: Left (ID 0) is Physical, Right (ID 1) is Interface.
  // For Subdomain 1: Left (ID 0) is Interface, Right (ID 1) is Physical.
  // We need homogeneous Dirichlet (u=0) on physical boundaries.
  {
    std::map<types::global_dof_index, double>           boundary_values;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    // Use ZeroFunction for u=0
    Functions::ZeroFunction<dim> function_bc;
    
    // Apply only to physical boundaries
    boundary_functions[subdomain_id == 0 ? 0 : 1] = &function_bc;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, false);
  }
}

void
Poisson2D::solve()
{
  // Use GMRES instead of CG because the matrix is non-symmetric
  // due to the convection term (b*u', v)
  SolverControl               solver_control(1000, 1e-12 * system_rhs.l2_norm());
  SolverGMRES<Vector<double>> solver(solver_control);

  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
}

void
Poisson2D::output(const unsigned int &iter) const
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
Poisson2D::apply_interface_dirichlet(const Poisson2D &other)
{
  const auto interface_map = compute_interface_map(other);

  // We use the interface map to build a boundary values map for interface DoFs.
  std::map<types::global_dof_index, double> boundary_values;
  for (const auto &dof : interface_map)
    boundary_values[dof.first] = other.solution[dof.second];

  // Then, we apply those boundary values.
  MatrixTools::apply_boundary_values(
    boundary_values, system_matrix, solution, system_rhs, false);
}

void
Poisson2D::apply_interface_neumann(Poisson2D &other)
{
  const auto interface_map = compute_interface_map(other);

  // We assemble the interface residual of the other subproblem. Indeed,
  // directly computing the normal derivative of the solution on the other
  // subdomain has extremely poor accuracy. This is due to the fact that the
  // trace of the derivative has very low regularity. Therefore, we compute the
  // (weak) normal derivative as the residual of the system of the other
  // subdomain, excluding interface conditions.
  Vector<double> interface_residual;
  other.assemble();
  interface_residual = other.system_rhs;
  interface_residual *= -1;
  other.system_matrix.vmult_add(interface_residual, other.solution);

  // Then, we add the elements of the residual corresponding to interface DoFs
  // to the system rhs for current subproblem.
  for (const auto &dof : interface_map)
    system_rhs[dof.first] -= interface_residual[dof.second];
}

std::map<types::global_dof_index, types::global_dof_index>
Poisson2D::compute_interface_map(const Poisson2D &other) const
{
  // Retrieve interface DoFs on the current and other subdomain.
  IndexSet current_interface_dofs;
  IndexSet other_interface_dofs;

  // In 1D:
  // Sub 0: Interface is at right boundary (ID 1)
  // Sub 1: Interface is at left boundary (ID 0)

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

  // For each interface DoF on current subdomain, we find the corresponding one
  // on the other subdomain.
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
Poisson2D::apply_relaxation(const Vector<double> &old_solution,
                            const double         &lambda)
{
  solution *= lambda;
  solution.add(1.0 - lambda, old_solution);
}