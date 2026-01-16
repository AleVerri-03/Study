#include "Poisson1D.hpp"

class InitialCondition : public Function<1>
{
  public:
    virtual double value(const Point<1> &p, const unsigned int = 0) const override
    {
      return p[0]; // u0(x) = x
    }
};

Poisson1D::Poisson1D(const unsigned int N_el_,
                     const unsigned int r_,
                     const double kappa_,
                     const double alpha_,
                     const double beta_,
                     const double dt_,
                     const double T_final_)
  : N_el(N_el_)
  , r(r_)
  , kappa(kappa_)
  , alpha_bc(alpha_)
  , beta_bc(beta_)
  , delta_t(dt_)
  , T_final(T_final_)
  , time(0.0)
{}


void
Poisson1D::setup()
{
  std::cout << "===============================================" << std::endl;

  {
    std::cout << "Initializing the mesh" << std::endl;
    GridGenerator::subdivided_hyper_cube(mesh, N_el, 0.0, 1.0, true);
    std::cout << "  Number of elements = " << mesh.n_active_cells()
              << std::endl;

    const std::string mesh_file_name = "mesh-" + std::to_string(N_el) + ".vtk";
    GridOut           grid_out;
    std::ofstream     grid_out_file(mesh_file_name);
    grid_out.write_vtk(mesh, grid_out_file);
    std::cout << "  Mesh saved to " << mesh_file_name << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  {
    std::cout << "Initializing the finite element space" << std::endl;
    fe = std::make_unique<FE_SimplexP<dim>>(r);

    std::cout << "  Degree                     = " << fe->degree << std::endl;
    std::cout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    std::cout << "  Quadrature points per cell = " << quadrature->size()
              << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  {
    std::cout << "Initializing the DoF handler" << std::endl;

    // Initialize the DoF handler with the mesh we constructed.
    dof_handler.reinit(mesh);

    // "Distribute" the degrees of freedom. For a given finite element space,
    // initializes info on the control variables (how many they are, where
    // they are collocated, their "global indices", ...).
    dof_handler.distribute_dofs(*fe);

    std::cout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    std::cout << "Initializing the linear system" << std::endl;

    // We first initialize a "sparsity pattern", i.e. a data structure that
    // indicates which entries of the matrix are zero and which are different
    // from zero. To do so, we construct first a DynamicSparsityPattern (a
    // sparsity pattern stored in a memory- and access-inefficient way, but
    // fast to write) and then convert it to a SparsityPattern (which is more
    // efficient, but cannot be modified).
    std::cout << "  Initializing the sparsity pattern" << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    // Then, we use the sparsity pattern to initialize the system matrix
    std::cout << "  Initializing the system matrix" << std::endl;
    system_matrix.reinit(sparsity_pattern);

    // Finally, we initialize the right-hand side and solution vectors.
    std::cout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(dof_handler.n_dofs());
    std::cout << "  Initializing the solution vector" << std::endl;
    solution.reinit(dof_handler.n_dofs());

    solution_old.reinit(dof_handler.n_dofs());
    solution_old = 0.0; 
  }
}

#include <cmath> // Necessario per std::tanh

void Poisson1D::assemble()
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe, *quadrature,
                          update_values | update_gradients | 
                          update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
  std::vector<double> old_sol_values(n_q);

  system_matrix = 0.0;
  system_rhs    = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell_matrix = 0.0;
    cell_rhs    = 0.0;

    fe_values.get_function_values(solution_old, old_sol_values);

    // --- Calcolo di Tau per SUPG ---
    // h = lunghezza della cella
    double h = cell->measure(); 
    
    // Numero di Peclet locale: theta = (h * |kappa|) / (2 * mu). Qui mu=1.
    double theta = (h * std::abs(kappa)) / 2.0;
    
    // Funzione xi(theta) = coth(theta) - 1/theta
    // Nota: coth(x) = 1/tanh(x)
    double xi = 1.0 / std::tanh(theta) - 1.0 / theta;
    
    // Parametro tau
    double tau = (h / (2.0 * std::abs(kappa))) * xi;
    // -------------------------------

    for (unsigned int q = 0; q < n_q; ++q)
    {
      const double dx = fe_values.JxW(q);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const double phi_i = fe_values.shape_value(i, q);      // v
        const double grad_phi_i = fe_values.shape_grad(i, q)[0]; // v_x

        // Termine di test per SUPG: (kappa * v_x)
        double supg_test = kappa * grad_phi_i;

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          const double phi_j = fe_values.shape_value(j, q);      // u
          const double grad_phi_j = fe_values.shape_grad(j, q)[0]; // u_x

          // --- Termini Standard (Galerkin) ---
          double mass_term = (1.0 / delta_t) * phi_j * phi_i;
          double diff_term = grad_phi_j * grad_phi_i; // mu * u_x * v_x (mu=1)
          double adv_term  = kappa * grad_phi_j * phi_i;

          // --- Termini SUPG ---
          // 1. Parte temporale: (tau / dt) * u * (kappa * v_x)
          double supg_mass = (tau / delta_t) * phi_j * supg_test;
          
          // 2. Parte advezione: tau * (kappa * u_x) * (kappa * v_x)
          double supg_adv  = tau * (kappa * grad_phi_j) * supg_test;

          cell_matrix(i, j) += (mass_term + diff_term + adv_term + supg_mass + supg_adv) * dx;
        }

        // --- RHS ---
        // Standard: (1/dt) * u_old * v
        double rhs_standard = (1.0 / delta_t) * old_sol_values[q] * phi_i;
        
        // SUPG: (1/dt) * u_old * (tau * kappa * v_x)
        double rhs_supg     = (1.0 / delta_t) * old_sol_values[q] * tau * supg_test;

        cell_rhs(i) += (rhs_standard + rhs_supg) * dx;
      }
    }

    cell->get_dof_indices(dof_indices);
    system_matrix.add(dof_indices, cell_matrix);
    system_rhs.add(dof_indices, cell_rhs);
  }

  // Boundary conditions
  {
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ConstantFunction<dim>(alpha_bc), boundary_values);
    VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ConstantFunction<dim>(beta_bc), boundary_values);
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
  }
}
6
void
Poisson1D::solve()
{
  std::cout << "===============================================" << std::endl;

  ReductionControl solver_control(/* maxiter = */ 1000,
                                  /* tolerance = */ 1.0e-16,
                                  /* reduce = */ 1.0e-6);

  SolverGMRES<Vector<double>> solver(solver_control);

  std::cout << "  Solving the linear system" << std::endl;
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  std::cout << "  " << solver_control.last_step() << " CG iterations"
            << std::endl;
}

void
Poisson1D::output(unsigned int step) const
{
  std::cout << "===============================================" << std::endl;

  // The DataOut class manages writing the results to a file.
  DataOut<dim> data_out;

  // It can write multiple variables (defined on the same mesh) to a single
  // file. Each of them can be added by calling add_data_vector, passing the
  // associated DoFHandler and a name.
  data_out.add_data_vector(dof_handler, solution, "solution");

  // Once all vectors have been inserted, call build_patches to finalize the
  // DataOut object, preparing it for writing to file.
  data_out.build_patches();

  // Then, use one of the many write_* methods to write the file in an
  // appropriate format.
  const std::string output_file_name =
    "output-" + std::to_string(step) + ".vtk";
  std::ofstream output_file(output_file_name);
  data_out.write_vtk(output_file);

  std::cout << "Output written to " << output_file_name << std::endl;

  std::cout << "===============================================" << std::endl;
}

double
Poisson1D::compute_error(const VectorTools::NormType &norm_type,
                         const Function<dim>         &exact_solution) const
{
  // The error is an integral, and we approximate that integral using a
  // quadrature formula. To make sure we are accurate enough, we use a
  // quadrature formula with one node more than what we used in assembly.
  const QGaussSimplex<dim> quadrature_error(r + 2);

  // First we compute the norm on each element, and store it in a vector.
  Vector<double> error_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  // Then, we add out all the cells.
  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}

void Poisson1D::run()
{
  setup();

  std::cout << "Interpolating initial condition..." << std::endl;
  VectorTools::interpolate(dof_handler, InitialCondition(), solution_old);
  
  solution = solution_old; 

  std::cout << "Starting time loop..." << std::endl;
  time = 0.0;
  unsigned int step = 0;

  while (time < T_final - 1e-9)
  {
    time += delta_t;
    step++;

    assemble();
    solve();

    solution_old = solution;
    output(step);
  }
}