#include "Heat.hpp"

void
Heat::setup()
{
  //pcout << "===============================================" << std::endl;

  // Create the mesh.
  {
    //pcout << "Initializing the mesh" << std::endl;

    // Read serial mesh.
    Triangulation<dim> mesh_serial;

    // MODIFICA QUI: Generazione interna invece di leggere da file
      // Crea un cubo (linea in 1D) suddiviso in 40 elementi tra 0.0 e 1.0.
      // L'ultimo argomento 'true' attiva la colorazione dei bordi:
      // x=0 avrÃ  boundary_id = 0 (Left)
      // x=1 avrÃ  boundary_id = 1 (Right)
    GridGenerator::subdivided_hyper_cube(mesh_serial, 40, 0.0, 1.0, true);

    // Copy the serial mesh into the parallel one.
    {
      GridTools::partition_triangulation(mpi_size, mesh_serial);

      const auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
      mesh.create_triangulation(construction_data);
    }

    //pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;
  }

  //pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    //pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    //pcout << "  Degree                     = " << fe->degree << std::endl;
    //pcout << "  DoFs per cell              = " << fe->dofs_per_cell << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    //pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
  }

  //pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    //pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    //pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  //pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    //pcout << "Initializing the linear system" << std::endl;

    const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    //pcout << "  Initializing the sparsity pattern" << std::endl;
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    //pcout << "  Initializing the system matrix" << std::endl;
    system_matrix.reinit(sparsity);

    //pcout << "  Initializing vectors" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}

void
Heat::assemble()
{
  // Number of local DoFs for each element.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  // Local matrix and vector.
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Reset the global matrix and vector, just in case.
  system_matrix = 0.0;
  system_rhs    = 0.0;

  // Evaluation of the old solution on quadrature nodes of current cell.
  std::vector<double> solution_old_values(n_q);

  // Evaluation of the gradient of the old solution on quadrature nodes of
  // current cell.
  std::vector<Tensor<1, dim>> solution_old_grads(n_q);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      // Evaluate the old solution and its gradient on quadrature nodes.
      fe_values.get_function_values(solution, solution_old_values);
      fe_values.get_function_gradients(solution, solution_old_grads);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double f_old_loc = f(fe_values.quadrature_point(q), time - delta_t);
          const double f_new_loc = f(fe_values.quadrature_point(q), time);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // Time derivative.
                  cell_matrix(i, j) += (1.0 / delta_t) *             //
                                       fe_values.shape_value(i, q) * //
                                       fe_values.shape_value(j, q) * //
                                       fe_values.JxW(q);

                  // Diffusion.
                  cell_matrix(i, j) +=
                    theta * epsilon *                             //
                    fe_values.shape_grad(i, q) *                //
                    fe_values.shape_grad(j, q) *                  //
                    fe_values.JxW(q);

                  cell_matrix(i, j) +=
                    theta * b *                      //
                    fe_values.shape_grad(j, q)[0] *                //
                    fe_values.shape_value(i, q) *                  //
                    fe_values.JxW(q);

                  cell_matrix(i, j) -=
                    theta * kappa *                      //
                    fe_values.shape_value(j, q) *                //
                    fe_values.shape_value(i, q) *                  //
                    fe_values.JxW(q);
                }

              // Time derivative.
              cell_rhs(i) += (1.0 / delta_t) *             //
                             fe_values.shape_value(i, q) * //
                             solution_old_values[q] *      //
                             fe_values.JxW(q);

              // Diffusion.
              cell_rhs(i) -= (1.0 - theta) * epsilon *                   //
                             solution_old_grads[q] *                //
                             fe_values.shape_grad(i, q) *          //
                             fe_values.JxW(q);

              cell_rhs(i) -= (1.0 - theta) * b *                   //
                             solution_old_grads[q][0] *                //
                             fe_values.shape_value(i, q) *          //
                             fe_values.JxW(q);

              cell_rhs(i) += (1.0 - theta) * kappa *                   //
                             solution_old_values[q] *      //
                             fe_values.shape_value(i, q) *          //
                             fe_values.JxW(q);

              // Forcing term.
              cell_rhs(i) +=
                (theta * f_new_loc + (1.0 - theta) * f_old_loc) * //
                fe_values.shape_value(i, q) *                     //
                fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);

      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  {
    // Applicazione boundary contitions di Dirichlet
    std::map<types::global_dof_index, double> boundary_values;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    Functions::ZeroFunction<dim> zero_function;
    boundary_functions[0] = &zero_function;

    VectorTools::interpolate_boundary_values(dof_handler,
                                               boundary_functions,
                                               boundary_values);

    MatrixTools::apply_boundary_values(boundary_values,
                                         system_matrix,
                                         solution_owned, // vettore fittizio per la struttura
                                         system_rhs);
  }
}

void
Heat::solve_linear_system()
{
  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(
    system_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  ReductionControl solver_control(/* maxiter = */ 10000,
                                  /* tolerance = */ 1.0e-16,
                                  /* reduce = */ 1.0e-6);

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
  //pcout << solver_control.last_step() << " CG iterations" << std::endl;
}

void
Heat::output() const
{
  DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler, solution, "solution");

  // Add vector for parallel partition.
  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::filesystem::path mesh_path(mesh_file_name);
  const std::string output_file_name = "output-" + mesh_path.stem().string();

  data_out.write_vtu_with_pvtu_record(/* folder = */ "./",
                                      /* basename = */ output_file_name,
                                      /* index = */ timestep_number,
                                      MPI_COMM_WORLD);
}

void
Heat::run()
{
  // Setup initial conditions.
  {
    setup();

    VectorTools::interpolate(dof_handler, FunctionU0(), solution_owned);
    solution = solution_owned;

    time            = 0.0;
    timestep_number = 0;

    // Output initial condition.
    output();
  }

  //pcout << "===============================================" << std::endl;

  // Time-stepping loop.
  while (time < T - 0.5 * delta_t)
    {
      time += delta_t;
      ++timestep_number;

      //pcout << "Timestep " << std::setw(3) << timestep_number
      //      << ", time = " << std::setw(4) << std::fixed << std::setprecision(2)
      //      << time << " : ";

      assemble();
      solve_linear_system();

      // Perform parallel communication to update the ghost values of the
      // solution vector.
      solution = solution_owned;

      output();
    }
}

double Heat::compute_error(const Function<dim> &exact_solution)
{
  Vector<double> difference_per_cell(mesh.n_active_cells());

  // Integrazione della differenza (u_h - u_ex)^2
  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    exact_solution,
                                    difference_per_cell,
                                    *quadrature,
                                    VectorTools::L2_norm);

  // Somma globale degli errori locali (norma L2 globale)
  const double global_error = VectorTools::compute_global_error(mesh,
                                                                difference_per_cell,
                                                                VectorTools::L2_norm);
  return global_error;
}