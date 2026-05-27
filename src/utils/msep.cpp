#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace 
{
    double rate_at(const std::vector<double>& rates, int d, int alpha, int beta)
    {
        return rates[static_cast<std::size_t>(alpha) * d + beta];
    }

    std::vector<double> flatten_rates(py::array_t<double, py::array::c_style | py::array::forcecast> rates, int d)
    {
        py::buffer_info info = rates.request();

        const auto* ptr = static_cast<const double*>(info.ptr);
        return std::vector<double>(ptr, ptr + static_cast<std::size_t>(d) * d);
    }
}

class MultiSpeciesExclusionProcess {
public:
    int dimension;
    std::vector<double> density;
    std::vector<double> rates_matrix; 
    int length;
    double max_rate;

    std::mt19937 rng;
    std::vector<int> chain;
    std::vector<double> proj_vectors; 

    MultiSpeciesExclusionProcess(int dim, const std::vector<double>& dens, py::array_t<double, py::array::c_style | py::array::forcecast> rates, int len, unsigned int seed, bool do_shuffle = true) : dimension(dim), density(dens), rates_matrix(flatten_rates(rates, dim)), length(len), max_rate(1.0), rng(seed)
    {
        max_rate = *std::max_element(rates_matrix.begin(), rates_matrix.end());

        proj_vectors = build_projected_vectors();
        chain = build_chain();

        if (do_shuffle) 
        {
            shuffle_chain(chain);
        }
    }

    MultiSpeciesExclusionProcess(int dim, const std::vector<double>& dens, py::array_t<double, py::array::c_style | py::array::forcecast> rates, int len, bool do_shuffle = true) : MultiSpeciesExclusionProcess(dim, dens, rates, len, std::random_device{}(), do_shuffle)
    {

    }

    void update() 
    {
        update_state(chain);
    }

    py::array_t<int> get_chain() const 
    {
        py::array_t<int> out(length);
        std::copy(chain.begin(), chain.end(), static_cast<int*>(out.request().ptr));
        return out;
    }

    void set_chain(const std::vector<int>& new_chain)
    {
        if (static_cast<int>(new_chain.size()) != length)
        {
            throw std::runtime_error("chain length must equal length");
        }

        for (int species : new_chain)
        {
            if (species < 0 || species >= dimension)
            {
                throw std::runtime_error("chain entries must be integers from 0 to dimension - 1");
            }
        }

        chain = new_chain;
    }

    py::array_t<double> get_projected_vectors_array() const 
    {
        py::array_t<double> out({dimension, dimension - 1});
        std::copy(proj_vectors.begin(), proj_vectors.end(), static_cast<double*>(out.request().ptr));
        return out;
    }


    py::array simulate(int steps = 100000, bool store_history = false, bool get_projection = false, int skip = 1)
    {
        const int path_dim = dimension - 1;

        if (store_history) 
        {
            const int n_saved = steps / skip + 1;

            if (get_projection) 
            {
                py::array_t<double> out({n_saved, length + 1, path_dim});
                auto* path = static_cast<double*>(out.request().ptr);

                std::fill(path, path + static_cast<std::size_t>(n_saved) * (length + 1) * path_dim, 0.0);

                auto write_projection = [&](int save_index)
                {
                    const std::size_t step_offset = static_cast<std::size_t>(save_index) * (length + 1) * path_dim;

                    for (int j = 0; j < length; ++j)
                    {
                        const int species = chain[j];

                        for (int k = 0; k < path_dim; ++k) 
                        {
                            path[step_offset + static_cast<std::size_t>(j + 1) * path_dim + k] = path[step_offset + static_cast<std::size_t>(j) * path_dim + k] + proj_vectors[static_cast<std::size_t>(species) * path_dim + k];
                        }
                    }
                };

                // Save projected path for initial chain at simulation step 0.
                int save_index = 0;
                write_projection(save_index);
                ++save_index;

                for (int step = 1; step <= steps; ++step) 
                {
                    update_state(chain);

                    if (step % skip == 0)
                    {
                        write_projection(save_index);
                        ++save_index;
                    }
                }
                return out;
            }
            else 
            {
                py::array_t<int> out({n_saved, length});
                auto* history = static_cast<int*>(out.request().ptr);

                // Save initial chain.
                std::copy(chain.begin(), chain.end(), history);

                int save_index = 1;

                for (int step = 1; step <= steps; ++step) 
                {
                    update_state(chain);

                    if (step % skip == 0)
                    {
                        std::copy(chain.begin(), chain.end(), history + static_cast<std::size_t>(save_index) * length);
                        ++save_index;
                    }
                }
                return out;
            }
        }
        else 
        {
            for (int step = 0; step < steps; ++step) 
            {
                update_state(chain);
            }
            if (get_projection)
            {
                py::array_t<double> out({length + 1, path_dim});
                auto* path = static_cast<double*>(out.request().ptr);
                std::fill(path, path + static_cast<std::size_t>(length + 1) * path_dim, 0.0);

                for (int j = 0; j < length; ++j)
                {
                    const int species = chain[j];
                    for (int k = 0; k < path_dim; ++k) 
                    {
                        path[static_cast<std::size_t>(j + 1) * path_dim + k] = path[static_cast<std::size_t>(j) * path_dim + k] + proj_vectors[static_cast<std::size_t>(species) * path_dim + k];
                    }
                }
                return out;
            }
            else
            {
                py::array_t<int> out(length);
                std::copy(chain.begin(), chain.end(), static_cast<int*>(out.request().ptr));
                return out;
            }
        }
    }

    py::array_t<double> get_path_projection() const 
    {
        const int path_dim = dimension - 1;
        py::array_t<double> out({length + 1, path_dim});
        auto* path = static_cast<double*>(out.request().ptr);
        std::fill(path, path + static_cast<std::size_t>(length + 1) * path_dim, 0.0);

        for (int i = 0; i < length; ++i) 
        {
            const int species = chain[i];
            for (int k = 0; k < path_dim; ++k) 
            {
                path[static_cast<std::size_t>(i + 1) * path_dim + k] = path[static_cast<std::size_t>(i) * path_dim + k] + proj_vectors[static_cast<std::size_t>(species) * path_dim + k];
            }
        }
        return out;
    }

    py::array_t<std::complex<double>> fourier_time_series(int n_samples = 60000, int sample_every = 1, int mode = 1)
    {
        /*
            Returns the specified Fourier mode of the hydrodynamic normal-mode
            density fields.

            Output shape: (n_samples, dimension - 1)

            Construction:
                1. Use species 0 as the dependent species, since sum_a n_a(x,t)=1.
                2. Build independent centered density fields
                       u_a(x,t) = 1_{state[x] == a} - density[a],
                   for a = 1,...,dimension-1.
                3. Build the current Jacobian J for these independent densities:
                       A_ab = g_ab - g_ba,
                       j_a = rho_a [ A_a0 + sum_b (A_ab - A_a0) rho_b ],
                       J_ab = d j_a / d rho_b.
                4. Diagonalize J^T using Eigen. The right eigenvectors of J^T are
                   the left eigenvectors of J. These rows form the normal-mode
                   transformation R.
                5. Compute phi_hat_alpha(k,t) = sum_a R_{alpha,a} u_hat_a(k,t).

            The normalization of eigenvectors does not affect the relaxation exponent.
            For asymmetric systems, these modes may carry a ballistic phase
                exp(-i v_alpha k t),
            so demodulate by exp(+i v_alpha k t) before fitting the decay envelope.
        */
        const int n_modes = dimension - 1;

        // Build current Jacobian J for independent species 1,...,dimension-1.
        Eigen::MatrixXd J(n_modes, n_modes);

        for (int a_ind = 0; a_ind < n_modes; ++a_ind)
        {
            const int a = a_ind + 1;  // physical species label
            const double rho_a = density[a];
            const double A_a0 = rate_at(rates_matrix, dimension, a, 0)
                              - rate_at(rates_matrix, dimension, 0, a);

            double bracket = A_a0;
            for (int c_ind = 0; c_ind < n_modes; ++c_ind)
            {
                const int c = c_ind + 1;
                const double A_ac = rate_at(rates_matrix, dimension, a, c)
                                  - rate_at(rates_matrix, dimension, c, a);
                bracket += (A_ac - A_a0) * density[c];
            }

            for (int b_ind = 0; b_ind < n_modes; ++b_ind)
            {
                const int b = b_ind + 1;
                const double A_ab = rate_at(rates_matrix, dimension, a, b)
                                  - rate_at(rates_matrix, dimension, b, a);

                J(a_ind, b_ind) = ((a_ind == b_ind) ? bracket : 0.0)
                                + rho_a * (A_ab - A_a0);
            }
        }

        // Diagonalize J^T. If v_alpha is a right eigenvector of J^T,
        // then v_alpha^T is a left eigenvector of J.
        Eigen::ComplexEigenSolver<Eigen::MatrixXd> solver(J.transpose());
        const Eigen::VectorXcd eigenvalues = solver.eigenvalues();
        const Eigen::MatrixXcd eigenvectors = solver.eigenvectors();

        // Sort modes by velocity Re(eigenvalue), so columns are reproducible.
        std::vector<int> order(n_modes);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int i, int j)
        {
            const double vi = eigenvalues[i].real();
            const double vj = eigenvalues[j].real();
            if (std::abs(vi - vj) > 1e-12)
            {
                return vi < vj;
            }
            return eigenvalues[i].imag() < eigenvalues[j].imag();
        });

        // R[alpha, a_ind] is the left eigenvector component for mode alpha.
        std::vector<std::complex<double>> R(static_cast<std::size_t>(n_modes) * n_modes);
        for (int alpha = 0; alpha < n_modes; ++alpha)
        {
            const int col = order[alpha];
            Eigen::VectorXcd left_vec = eigenvectors.col(col);

            const double vec_norm = left_vec.norm();
            if (vec_norm < 1e-14)
            {
                throw std::runtime_error("Eigen returned a near-zero eigenvector");
            }
            left_vec /= vec_norm;

            for (int a_ind = 0; a_ind < n_modes; ++a_ind)
            {
                R[static_cast<std::size_t>(alpha) * n_modes + a_ind] =
                    std::complex<double>(left_vec[a_ind].real(), left_vec[a_ind].imag());
            }
        }

        std::vector<int> state = chain;
        const double q = 2.0 * mode * std::acos(-1.0) / static_cast<double>(length);

        std::vector<double> cos_q(length);
        std::vector<double> sin_q(length);
        for (int j = 0; j < length; ++j)
        {
            cos_q[j] = std::cos(q * j);
            sin_q[j] = std::sin(q * j);
        }

        py::array_t<std::complex<double>> out({n_samples, n_modes});
        auto* X = static_cast<std::complex<double>*>(out.request().ptr);

        for (int n = 0; n < n_samples; ++n)
        {
            std::vector<std::complex<double>> u_hat(n_modes, std::complex<double>(0.0, 0.0));

            for (int j = 0; j < length; ++j)
            {
                const std::complex<double> phase(cos_q[j], sin_q[j]);

                for (int a_ind = 0; a_ind < n_modes; ++a_ind)
                {
                    const int a = a_ind + 1;
                    const double centered_occ = ((state[j] == a) ? 1.0 : 0.0) - density[a];
                    u_hat[a_ind] += centered_occ * phase;
                }
            }

            for (int alpha = 0; alpha < n_modes; ++alpha)
            {
                std::complex<double> phi_hat(0.0, 0.0);

                for (int a_ind = 0; a_ind < n_modes; ++a_ind)
                {
                    phi_hat += R[static_cast<std::size_t>(alpha) * n_modes + a_ind] * u_hat[a_ind];
                }

                X[static_cast<std::size_t>(n) * n_modes + alpha] = phi_hat;
            }

            for (int s = 0; s < sample_every; ++s)
            {
                for (int step = 0; step < length; ++step)
                {
                    update_state(state);
                }
            }
        }

        return out;
    }

private:
    std::vector<int> build_chain() const 
    {
        std::vector<int> out;
        out.reserve(length);

        for (int species = 0; species < dimension; ++species) 
        {
            const int count = static_cast<int>(std::llround(density[species] * length));
            for (int i = 0; i < count; ++i) 
            {
                out.push_back(species);
            }
        }

        if (static_cast<int>(out.size()) != length) 
        {
            throw std::runtime_error("internal error while building chain");
        }
        return out;
    }

    void shuffle_chain(std::vector<int>& values) 
    {
        for (int i = static_cast<int>(values.size()) - 1; i > 0; --i) 
        {
            std::uniform_int_distribution<int> dist(0, i);
            const int j = dist(rng);
            std::swap(values[i], values[j]);
        }
    }

    void update_state(std::vector<int>& state, bool same_state_swaps = false) 
    {
        std::uniform_int_distribution<int> index_dist(0, length - 1);
        std::uniform_real_distribution<double> real_dist(0.0, 1.0);

        int j, k;
        int alpha, beta;

        do 
        {
            j = index_dist(rng);
            k = (j + 1 == length) ? 0 : j + 1;

            alpha = state[j];
            beta = state[k];

        } while (!same_state_swaps && alpha == beta);
        
        const double rate = rate_at(rates_matrix, dimension, alpha, beta);

        if (real_dist(rng) < rate / max_rate) 
        {
            state[j] = beta;
            state[k] = alpha;
        }
    }

    static double dot(const std::vector<double>& a, const std::vector<double>& b) 
    {
        double s = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) 
        {
            s += a[i] * b[i];
        }
        return s;
    }

    static double norm(const std::vector<double>& a) 
    {
        return std::sqrt(dot(a, a));
    }


    // gram schmidt process
    std::vector<std::vector<double>> build_plane_basis() const 
    {
        std::vector<std::vector<double>> basis;
        basis.reserve(dimension - 1);

        for (int k = 0; k < dimension - 1; ++k) {
            std::vector<double> v(dimension, 0.0);
            v[k] = 1.0;
            v[dimension - 1] = -1.0;

            for (const auto& b : basis) 
            {
                const double coeff = dot(v, b);
                for (int i = 0; i < dimension; ++i) 
                {
                    v[i] -= coeff * b[i];
                }
            }

            const double n = norm(v);
            for (double& x : v) 
            {
                x /= n;
            }
            basis.push_back(std::move(v));
        }
        return basis;
    }

    std::vector<double> build_projected_vectors() const 
    {
        const int path_dim = dimension - 1;
        const auto basis = build_plane_basis();
        std::vector<double> coords(static_cast<std::size_t>(dimension) * path_dim, 0.0);

        // coordinate of projected e_i along basis vector b_k since b_k lies in the
        // plane sum(x_i)=0, dot(e_i - n_hat/dot correction, b_k) = dot(e_i, b_k)
        for (int species = 0; species < dimension; ++species) 
        {
            double row_norm_sq = 0.0;
            for (int k = 0; k < path_dim; ++k) 
            {
                const double c = basis[k][species];
                coords[static_cast<std::size_t>(species) * path_dim + k] = c;
                row_norm_sq += c * c;
            }

            const double row_norm = std::sqrt(row_norm_sq);
            if (row_norm > 1e-14) 
            {
                for (int k = 0; k < path_dim; ++k) 
                {
                    coords[static_cast<std::size_t>(species) * path_dim + k] /= row_norm;
                }
            }
        }
        return coords;
    }
};

PYBIND11_MODULE(msep, m) {
    py::class_<MultiSpeciesExclusionProcess>(m, "MultiSpeciesExclusionProcess")
        .def(py::init<int,
                const std::vector<double>&,
                py::array_t<double, py::array::c_style | py::array::forcecast>,
                int,
                unsigned int,
                bool>(),
            py::arg("dimension"),
            py::arg("density"),
            py::arg("rates_matrix"),
            py::arg("length"),
            py::arg("seed"),
            py::arg("shuffle") = true)
        .def(py::init<int, const std::vector<double>&,
                py::array_t<double, py::array::c_style | py::array::forcecast>,
                int, bool>(),
            py::arg("dimension"),
            py::arg("density"),
            py::arg("rates_matrix"),
            py::arg("length"),
            py::arg("shuffle") = true)
        .def_readonly("dimension", &MultiSpeciesExclusionProcess::dimension)
        .def_readonly("density", &MultiSpeciesExclusionProcess::density)
        .def_readonly("length", &MultiSpeciesExclusionProcess::length)
        .def_readonly("max_rate", &MultiSpeciesExclusionProcess::max_rate)
        .def("update", &MultiSpeciesExclusionProcess::update)
        .def("simulate", &MultiSpeciesExclusionProcess::simulate,
            py::arg("steps") = 100000,
            py::arg("store_history") = false, 
            py::arg("get_projection") = false, 
            py::arg("skip") = 1)
        .def("get_chain", &MultiSpeciesExclusionProcess::get_chain)
        .def("set_chain", &MultiSpeciesExclusionProcess::set_chain,
            py::arg("chain"))
        .def("get_path_projection", &MultiSpeciesExclusionProcess::get_path_projection)
        .def("get_projected_vectors", &MultiSpeciesExclusionProcess::get_projected_vectors_array)
        .def("fourier_time_series", &MultiSpeciesExclusionProcess::fourier_time_series,
            py::kw_only(),
            py::arg("n_samples") = 60000,
            py::arg("sample_every") = 1,
            py::arg("mode") = 1);
}