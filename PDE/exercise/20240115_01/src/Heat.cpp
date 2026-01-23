#include "Heat.hpp"

void Heat::setup()
{
  //pcout << "===============================================" << std::endl;

  // Create the mesh.
  {
    //pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;
    
    // Legge la mesh da file
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);
    std::ifstream input_file(mesh_file);
    grid_in.read_msh(input_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    //pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;
  }

  //pcout << "-----------------------------------------------" << std::endl;

  // Initialize FE space.
  {
    //pcout << "Initializing the finite element space" << std::endl;
    fe = std::make_unique<FE_SimplexP<dim>>(r); // O FE_Q<dim>(r) in 1D
    //pcout << "  Degree                     = " << fe->degree << std::endl;
    
    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);
    //pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
  }

  //pcout << "-----------------------------------------------" << std::endl;

  // Initialize DoF handler.
  {
    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);
    //pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;

    const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    // Setup Constraints (Dirichlet at x=0)
    // In 1D HyperCube: left face is boundary 0, right face is boundary 1.
    // u(0) = 0 => Boundary 0 is homogeneous Dirichlet.
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             constraints);
    constraints.close();

    // Initialize linear system
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity, constraints, false);
    sparsity.compress();

    system_matrix.reinit(sparsity);
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}

void Heat::assemble()
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_matrix = 0.0;
  system_rhs    = 0.0;

  std::vector<double> solution_old_values(n_q);
  std::vector<Tensor<1, dim>> solution_old_grads(n_q);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned()) continue;

      fe_values.reinit(cell);
      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      fe_values.get_function_values(solution, solution_old_values);
      fe_values.get_function_gradients(solution, solution_old_grads);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const Point<dim> &xq = fe_values.quadrature_point(q);
          const Tensor<1, dim> b_loc = b(xq);

          const double f_old_loc = f(xq, time - delta_t);
          const double f_new_loc = f(xq, time);
          const double JxW = fe_values.JxW(q);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              // Test function values and gradients
              const double phi_i = fe_values.shape_value(i, q);
              const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                   const double phi_j = fe_values.shape_value(j, q);
                   const Tensor<1, dim> grad_phi_j = fe_values.shape_grad(j, q);

                   // Mass term: (phi_j, phi_i)
                   const double mass_term = phi_j * phi_i;

                   // Stiffness term a(phi_j, phi_i) = mu * (grad_phi_j, grad_phi_i) + (b . grad_phi_j) * phi_i
                   const double stiffness_term = mu * scalar_product(grad_phi_j, grad_phi_i)
                                                 + (b_loc * grad_phi_j) * phi_i;

                   // K_sys = M + (dt/2) * A  (theta = 0.5 for Crank-Nicolson)
                   cell_matrix(i, j) += ( mass_term + theta * delta_t * stiffness_term ) * JxW;
                }

              // RHS: (M - (dt/2)*A) * U_n + (dt/2) * (F_n+1 + F_n)
              const double u_old_val = solution_old_values[q];
              const Tensor<1, dim> &grad_u_old = solution_old_grads[q];
              
              // (M - (1-theta)*dt*A) applied to u_old
              // M*u_old term
              double rhs_term = u_old_val * phi_i;
              
              // -(1-theta)*dt*A*u_old term = -(1-theta)*dt * [mu*(grad_u_old, grad_phi_i) + (b.grad_u_old)*phi_i]
              rhs_term -= (1.0 - theta) * delta_t * (mu * scalar_product(grad_u_old, grad_phi_i) 
                                                      + (b_loc * grad_u_old) * phi_i);

              // Forcing term: (dt/2) * (f_n+1 + f_n) * phi_i  (for theta=0.5)
              // General: dt * (theta * f_n+1 + (1-theta) * f_n) * phi_i
              rhs_term += delta_t * (theta * f_new_loc + (1.0 - theta) * f_old_loc) * phi_i;

              cell_rhs(i) += rhs_term * JxW;
            }
        }

      cell->get_dof_indices(dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, dof_indices, system_matrix, system_rhs);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}

void Heat::solve_linear_system()
{
  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(system_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  ReductionControl solver_control(10000, 1.0e-12, 1.0e-6);

  // Usiamo GMRES perché la matrice non è simmetrica a causa del termine convettivo b != 0
  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);

  solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
  
  // Applichiamo i vincoli al vettore soluzione (per i nodi Dirichlet)
  constraints.distribute(solution_owned);
  
  //pcout << solver_control.last_step() << " GMRES iterations" << std::endl;
}

void Heat::output() const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "solution");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::string output_file_name = "solution";
  data_out.write_vtu_with_pvtu_record("./", output_file_name, timestep_number, MPI_COMM_WORLD);
}

double Heat::compute_error(){
  // Impostiamo il tempo corretto nella soluzione esatta
  ExactSolution exact_sol;
  exact_sol.set_time(time);

  Vector<double> difference_per_cell(mesh.n_active_cells());

  // Integriamo la differenza tra soluzione numerica ed esatta in norma L2
  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    exact_sol,
                                    difference_per_cell,
                                    QGauss<dim>(r + 2), // Quadratura un po' più accurata per l'errore
                                    VectorTools::L2_norm);

  const double local_error_sq = difference_per_cell.l2_norm() * difference_per_cell.l2_norm(); // somma dei quadrati
  
  // Raccogliamo l'errore globale su tutti i processi MPI
  double global_error_sq = Utilities::MPI::sum(local_error_sq, MPI_COMM_WORLD);

  return std::sqrt(global_error_sq);
}

double Heat::run()
{
  setup();

  VectorTools::interpolate(dof_handler, FunctionU0(), solution_owned);
  solution = solution_owned;

  time = 0.0;
  timestep_number = 0;

  // Ciclo temporale...
  while (time < T - 0.5 * delta_t)
    {
      time += delta_t;
      ++timestep_number;
      assemble();
      solve_linear_system();
      solution = solution_owned;
    }
    
  // Al tempo finale T, calcoliamo l'errore
  return compute_error();
}