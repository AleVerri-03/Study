#include "Heat.hpp"

void Heat::setup()
{
  //pcout << "===============================================" << std::endl;

  // Create the mesh.
  {
    //pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;
    // Genera un ipercubo (linea) suddiviso in N_elements da 0 a 1
    GridGenerator::subdivided_hyper_cube(mesh_serial, N_elements, 0.0, 1.0);

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
          const double b_loc = b(xq);
          // Epsilon and k are constant members

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

                   // Mass term: (u, v)
                   const double mass_term = phi_j * phi_i;

                   // Stiffness term a(u, v) = (eps*u_x - b*u)*v_x + k*u*v
                   // a(phi_j, phi_i)
                   // Nota: il termine convettivo è - b * u * v_x
                   const double stiffness_term = epsilon * scalar_product(grad_phi_j, grad_phi_i) // Diffusion
                                                 - b_loc * phi_j * grad_phi_i[0]                  // Convection (scalar in 1D)
                                                 + k * phi_j * phi_i;                             // Reaction

                   // Matrix M/dt + theta*A
                   cell_matrix(i, j) += ( (1.0 / delta_t) * mass_term + theta * stiffness_term ) * JxW;
                }

              // RHS assembly
              // Term 1: (M/dt - (1-theta)*A) * u_old
              // Ricostruiamo l'azione locale su u_old
              double u_old_val = solution_old_values[q];
              double u_old_grad = solution_old_grads[q][0];
              
              // A applied to u_old
              double A_u_old = epsilon * u_old_grad * grad_phi_i[0] 
                               - b_loc * u_old_val * grad_phi_i[0] 
                               + k * u_old_val * phi_i;

              double mass_u_old = u_old_val * phi_i;

              cell_rhs(i) += ( (1.0 / delta_t) * mass_u_old - (1.0 - theta) * A_u_old ) * JxW;

              // Term 2: Forcing (theta*f_new + (1-theta)*f_old)
              cell_rhs(i) += ( theta * f_new_loc + (1.0 - theta) * f_old_loc ) * phi_i * JxW;
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