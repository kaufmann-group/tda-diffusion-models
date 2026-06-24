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
#include <variant>
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

        if (info.ndim != 2 || info.shape[0] != d || info.shape[1] != d)
        {
            throw std::runtime_error("rates_matrix must have shape (dimension, dimension)");
        }

        const auto* ptr = static_cast<const double*>(info.ptr);
        return std::vector<double>(ptr, ptr + static_cast<std::size_t>(d) * d);
    }
}

class MultiSpeciesExclusionProcess
{
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
        validate_parameters();

        max_rate = *std::max_element(rates_matrix.begin(), rates_matrix.end());
        if (max_rate <= 0.0)
        {
            throw std::runtime_error("at least one rate must be positive");
        }

        proj_vectors = build_projected_vectors();
        chain = build_chain();

        if (do_shuffle)
        {
            shuffle_chain(chain);
        }
    }

    MultiSpeciesExclusionProcess( int dim, const std::vector<double>& dens, py::array_t<double, py::array::c_style | py::array::forcecast> rates, int len, bool do_shuffle = true ) : MultiSpeciesExclusionProcess(dim, dens, rates, len, std::random_device{}(), do_shuffle)
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
        if (steps < 0)
        {
            throw std::runtime_error("steps must be nonnegative");
        }
        if (skip <= 0)
        {
            throw std::runtime_error("skip must be positive");
        }

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
                            path[step_offset + static_cast<std::size_t>(j + 1) * path_dim + k] =
                                path[step_offset + static_cast<std::size_t>(j) * path_dim + k]
                                + proj_vectors[static_cast<std::size_t>(species) * path_dim + k];
                        }
                    }
                };

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
                return get_path_projection();
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
                path[static_cast<std::size_t>(i + 1) * path_dim + k] =
                    path[static_cast<std::size_t>(i) * path_dim + k]
                    + proj_vectors[static_cast<std::size_t>(species) * path_dim + k];
            }
        }
        return out;
    }

    // Returns the hydrodynamic current Jacobian J for the independent density
    // fields species 1, ..., dimension - 1, using species 0 as reference.
    py::array_t<double> get_current_jacobian() const
    {
        const int n_modes = dimension - 1;
        Eigen::MatrixXd J = build_current_jacobian_matrix();

        py::array_t<double> out({n_modes, n_modes});
        auto* ptr = static_cast<double*>(out.request().ptr);

        for (int i = 0; i < n_modes; ++i)
        {
            for (int j = 0; j < n_modes; ++j)
            {
                ptr[static_cast<std::size_t>(i) * n_modes + j] = J(i, j);
            }
        }

        return out;
    }

    // Returns the real part of the left eigenvector matrix R.
    // Row gamma gives the left eigenvector for normal mode gamma.
    // For three species, R has shape (2, 2).
    py::array_t<double> get_left_eigenvector_matrix() const
    {
        const int n_modes = dimension - 1;
        Eigen::MatrixXcd R_complex = build_left_eigenvector_matrix_complex();

        py::array_t<double> out({n_modes, n_modes});
        auto* R = static_cast<double*>(out.request().ptr);

        for (int gamma = 0; gamma < n_modes; ++gamma)
        {
            for (int a_ind = 0; a_ind < n_modes; ++a_ind)
            {
                if (std::abs(R_complex(gamma, a_ind).imag()) > 1e-10)
                {
                    throw std::runtime_error(
                        "Left eigenvector has non-negligible imaginary part. "
                        "Use get_left_eigenvector_matrix_complex if you need complex normal modes."
                    );
                }

                R[static_cast<std::size_t>(gamma) * n_modes + a_ind] =
                    R_complex(gamma, a_ind).real();
            }
        }

        return out;
    }

    py::array_t<std::complex<double>> get_left_eigenvector_matrix_complex() const
    {
        const int n_modes = dimension - 1;
        Eigen::MatrixXcd R_complex = build_left_eigenvector_matrix_complex();

        py::array_t<std::complex<double>> out({n_modes, n_modes});
        auto* R = static_cast<std::complex<double>*>(out.request().ptr);

        for (int gamma = 0; gamma < n_modes; ++gamma)
        {
            for (int a_ind = 0; a_ind < n_modes; ++a_ind)
            {
                R[static_cast<std::size_t>(gamma) * n_modes + a_ind] =
                    std::complex<double>(R_complex(gamma, a_ind).real(), R_complex(gamma, a_ind).imag());
            }
        }

        return out;
    }

    // Returns normal-mode density fields phi_gamma(x) in real space.
    // Shape: (length, dimension - 1). For dimension = 3, columns are phi_1, phi_2.
    py::array_t<double> get_normal_mode_fields() const
    {
        const int n_modes = dimension - 1;
        Eigen::MatrixXcd R_complex = build_left_eigenvector_matrix_complex();
        Eigen::MatrixXd R = real_matrix_or_throw(R_complex, "normal-mode fields");

        py::array_t<double> out({length, n_modes});
        auto* phi = static_cast<double*>(out.request().ptr);
        std::fill(phi, phi + static_cast<std::size_t>(length) * n_modes, 0.0);

        for (int x = 0; x < length; ++x)
        {
            for (int gamma = 0; gamma < n_modes; ++gamma)
            {
                double phi_gamma = 0.0;

                for (int a_ind = 0; a_ind < n_modes; ++a_ind)
                {
                    const int species = a_ind + 1;
                    const double u_a = ((chain[x] == species) ? 1.0 : 0.0) - density[species];
                    phi_gamma += R(gamma, a_ind) * u_a;
                }

                phi[static_cast<std::size_t>(x) * n_modes + gamma] = phi_gamma;
            }
        }

        return out;
    }

    // Returns normal-mode height functions h_gamma(x).
    // Shape: (length + 1, dimension - 1). For dimension = 3, columns are h_1, h_2.
    // h_gamma(0) = 0 and h_gamma(x + 1) - h_gamma(x) = phi_gamma(x).
    py::array_t<double> get_normal_mode_heights() const
    {
        const int n_modes = dimension - 1;
        Eigen::MatrixXcd R_complex = build_left_eigenvector_matrix_complex();
        Eigen::MatrixXd R = real_matrix_or_throw(R_complex, "normal-mode heights");

        py::array_t<double> out({length + 1, n_modes});
        auto* h = static_cast<double*>(out.request().ptr);
        std::fill(h, h + static_cast<std::size_t>(length + 1) * n_modes, 0.0);

        write_normal_mode_height_sample(h, R);

        return out;
    }

    // Returns a time series of normal-mode height functions.
    // Shape: (n_samples, length + 1, dimension - 1).
    // sample_every is measured in Monte Carlo sweeps, where one sweep = length attempted updates.
    py::array_t<double> normal_mode_height_time_series(int n_samples = 1000, int sample_every = 1)
    {
        if (n_samples <= 0)
        {
            throw std::runtime_error("n_samples must be positive");
        }
        if (sample_every <= 0)
        {
            throw std::runtime_error("sample_every must be positive");
        }

        const int n_modes = dimension - 1;
        Eigen::MatrixXcd R_complex = build_left_eigenvector_matrix_complex();
        Eigen::MatrixXd R = real_matrix_or_throw(R_complex, "normal-mode height time series");

        py::array_t<double> out({n_samples, length + 1, n_modes});
        auto* h = static_cast<double*>(out.request().ptr);
        std::fill(h, h + static_cast<std::size_t>(n_samples) * (length + 1) * n_modes, 0.0);

        for (int n = 0; n < n_samples; ++n)
        {
            double* sample_ptr = h + static_cast<std::size_t>(n) * (length + 1) * n_modes;
            write_normal_mode_height_sample(sample_ptr, R);
            advance_sweeps(sample_every);
        }

        return out;
    }

    std::variant<py::array_t<std::complex<double>>, std::pair<py::array_t<std::complex<double>>, py::array>> fourier_time_series(int n_samples = 60000, bool store_history = false, bool get_projection = false, int sample_every = 1, int mode = 1)
    {
        if (n_samples <= 0)
        {
            throw std::runtime_error("n_samples must be positive");
        }
        if (sample_every <= 0)
        {
            throw std::runtime_error("sample_every must be positive");
        }
        if (mode < 0 || mode >= length)
        {
            throw std::runtime_error("mode must satisfy 0 <= mode < length");
        }

        const int n_modes = dimension - 1;
        const int path_dim = n_modes;

        Eigen::MatrixXcd R_matrix = build_left_eigenvector_matrix_complex();

        const double q = 2.0 * static_cast<double>(mode) * std::acos(-1.0) / static_cast<double>(length);

        std::vector<double> cos_q(length);
        std::vector<double> sin_q(length);
        for (int j = 0; j < length; ++j)
        {
            cos_q[j] = std::cos(q * static_cast<double>(j));
            sin_q[j] = std::sin(q * static_cast<double>(j));
        }

        auto write_fourier_sample = [&](std::complex<double>* X, int sample_index)
        {
            std::vector<std::complex<double>> u_hat(n_modes, std::complex<double>(0.0, 0.0));

            for (int j = 0; j < length; ++j)
            {
                // Uses exp(+iqj). If your Python analysis assumes exp(-iqj), conjugate the result.
                const std::complex<double> phase(cos_q[j], sin_q[j]);

                for (int a_ind = 0; a_ind < n_modes; ++a_ind)
                {
                    const int a = a_ind + 1;
                    const double centered_occ = ((chain[j] == a) ? 1.0 : 0.0) - density[a];
                    u_hat[a_ind] += centered_occ * phase;
                }
            }

            for (int gamma = 0; gamma < n_modes; ++gamma)
            {
                std::complex<double> phi_hat(0.0, 0.0);

                for (int a_ind = 0; a_ind < n_modes; ++a_ind)
                {
                    phi_hat += R_matrix(gamma, a_ind) * u_hat[a_ind];
                }

                X[static_cast<std::size_t>(sample_index) * n_modes + gamma] = phi_hat;
            }
        };

        auto write_chain_history = [&](int* history, int sample_index)
        {
            std::copy(chain.begin(), chain.end(),
                      history + static_cast<std::size_t>(sample_index) * length);
        };

        auto write_projection_history = [&](double* path, int sample_index)
        {
            const std::size_t sample_offset =
                static_cast<std::size_t>(sample_index) * (length + 1) * path_dim;

            for (int k = 0; k < path_dim; ++k)
            {
                path[sample_offset + k] = 0.0;
            }

            for (int j = 0; j < length; ++j)
            {
                const int species = chain[j];
                for (int k = 0; k < path_dim; ++k)
                {
                    path[sample_offset + static_cast<std::size_t>(j + 1) * path_dim + k] =
                        path[sample_offset + static_cast<std::size_t>(j) * path_dim + k]
                        + proj_vectors[static_cast<std::size_t>(species) * path_dim + k];
                }
            }
        };

        py::array_t<std::complex<double>> fourier_out({n_samples, n_modes});
        auto* X = static_cast<std::complex<double>*>(fourier_out.request().ptr);

        if (!store_history)
        {
            for (int n = 0; n < n_samples; ++n)
            {
                write_fourier_sample(X, n);
                advance_sweeps(sample_every);
            }
            return fourier_out;
        }

        if (get_projection)
        {
            py::array_t<double> history_out({n_samples, length + 1, path_dim});
            auto* path = static_cast<double*>(history_out.request().ptr);
            std::fill(path,
                      path + static_cast<std::size_t>(n_samples) * (length + 1) * path_dim,
                      0.0);

            for (int n = 0; n < n_samples; ++n)
            {
                write_fourier_sample(X, n);
                write_projection_history(path, n);
                advance_sweeps(sample_every);
            }

            return std::make_pair(fourier_out, py::array(history_out));
        }
        else
        {
            py::array_t<int> history_out({n_samples, length});
            auto* history = static_cast<int*>(history_out.request().ptr);

            for (int n = 0; n < n_samples; ++n)
            {
                write_fourier_sample(X, n);
                write_chain_history(history, n);
                advance_sweeps(sample_every);
            }

            return std::make_pair(fourier_out, py::array(history_out));
        }
    }

private:
    void validate_parameters() const
    {
        if (dimension < 2)
        {
            throw std::runtime_error("dimension must be at least 2");
        }
        if (static_cast<int>(density.size()) != dimension)
        {
            throw std::runtime_error("density must have length equal to dimension");
        }
        if (length <= 0)
        {
            throw std::runtime_error("length must be positive");
        }
        if (static_cast<int>(rates_matrix.size()) != dimension * dimension)
        {
            throw std::runtime_error("rates_matrix must have dimension * dimension entries");
        }

        double density_sum = 0.0;
        for (double rho : density)
        {
            if (rho < 0.0)
            {
                throw std::runtime_error("densities must be nonnegative");
            }
            density_sum += rho;
        }

        if (std::abs(density_sum - 1.0) > 1e-8)
        {
            throw std::runtime_error("densities must sum to 1");
        }

        int rounded_total = 0;
        for (double rho : density)
        {
            rounded_total += static_cast<int>(std::llround(rho * length));
        }
        if (rounded_total != length)
        {
            throw std::runtime_error(
                "rounded density counts do not sum to length; choose densities and length so rho_a * length gives integer counts"
            );
        }

        for (double rate : rates_matrix)
        {
            if (rate < 0.0)
            {
                throw std::runtime_error("rates must be nonnegative");
            }
        }
    }

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

    void advance_sweeps(int sample_every)
    {
        for (int sweep = 0; sweep < sample_every; ++sweep)
        {
            for (int step = 0; step < length; ++step)
            {
                update_state(chain);
            }
        }
    }

    Eigen::MatrixXd build_current_jacobian_matrix() const
    {
        const int n_modes = dimension - 1;
        Eigen::MatrixXd J(n_modes, n_modes);

        for (int a_ind = 0; a_ind < n_modes; ++a_ind)
        {
            const int a = a_ind + 1;
            const double rho_a = density[a];

            const double A_a0 =
                rate_at(rates_matrix, dimension, a, 0)
                - rate_at(rates_matrix, dimension, 0, a);

            double bracket = A_a0;

            for (int c_ind = 0; c_ind < n_modes; ++c_ind)
            {
                const int c = c_ind + 1;

                const double A_ac =
                    rate_at(rates_matrix, dimension, a, c)
                    - rate_at(rates_matrix, dimension, c, a);

                bracket += (A_ac - A_a0) * density[c];
            }

            for (int b_ind = 0; b_ind < n_modes; ++b_ind)
            {
                const int b = b_ind + 1;

                const double A_ab =
                    rate_at(rates_matrix, dimension, a, b)
                    - rate_at(rates_matrix, dimension, b, a);

                J(a_ind, b_ind) =
                    ((a_ind == b_ind) ? bracket : 0.0)
                    + rho_a * (A_ab - A_a0);
            }
        }

        return J;
    }

    Eigen::MatrixXcd build_left_eigenvector_matrix_complex() const
    {
        const int n_modes = dimension - 1;
        Eigen::MatrixXd J = build_current_jacobian_matrix();

        // A right eigenvector of J^T is a left eigenvector of J.
        Eigen::ComplexEigenSolver<Eigen::MatrixXd> solver(J.transpose());
        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error("Eigen diagonalization failed");
        }

        const Eigen::VectorXcd eigenvalues = solver.eigenvalues();
        const Eigen::MatrixXcd eigenvectors = solver.eigenvectors();

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

        Eigen::MatrixXcd R(n_modes, n_modes);

        for (int gamma = 0; gamma < n_modes; ++gamma)
        {
            const int col = order[gamma];
            Eigen::VectorXcd left_vec = eigenvectors.col(col);

            const double vec_norm = left_vec.norm();
            if (vec_norm < 1e-14)
            {
                throw std::runtime_error("Eigen returned a near-zero eigenvector");
            }
            left_vec /= vec_norm;

            for (int a_ind = 0; a_ind < n_modes; ++a_ind)
            {
                R(gamma, a_ind) = left_vec[a_ind];
            }
        }

        return R;
    }

    Eigen::MatrixXd real_matrix_or_throw(const Eigen::MatrixXcd& M, const std::string& context) const
    {
        Eigen::MatrixXd out(M.rows(), M.cols());

        for (int i = 0; i < M.rows(); ++i)
        {
            for (int j = 0; j < M.cols(); ++j)
            {
                if (std::abs(M(i, j).imag()) > 1e-10)
                {
                    throw std::runtime_error(
                        "Cannot compute real " + context +
                        ": left eigenvectors have non-negligible imaginary parts."
                    );
                }
                out(i, j) = M(i, j).real();
            }
        }

        return out;
    }

    void write_normal_mode_height_sample(double* h, const Eigen::MatrixXd& R) const
    {
        const int n_modes = dimension - 1;

        for (int gamma = 0; gamma < n_modes; ++gamma)
        {
            h[gamma] = 0.0;
        }

        for (int x = 0; x < length; ++x)
        {
            for (int gamma = 0; gamma < n_modes; ++gamma)
            {
                double phi_gamma = 0.0;

                for (int a_ind = 0; a_ind < n_modes; ++a_ind)
                {
                    const int species = a_ind + 1;
                    const double u_a = ((chain[x] == species) ? 1.0 : 0.0) - density[species];
                    phi_gamma += R(gamma, a_ind) * u_a;
                }

                h[static_cast<std::size_t>(x + 1) * n_modes + gamma] =
                    h[static_cast<std::size_t>(x) * n_modes + gamma] + phi_gamma;
            }
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

    std::vector<std::vector<double>> build_plane_basis() const
    {
        std::vector<std::vector<double>> basis;
        basis.reserve(dimension - 1);

        for (int k = 0; k < dimension - 1; ++k)
        {
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

PYBIND11_MODULE(msep, m)
{
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
        .def(py::init<int,
                const std::vector<double>&,
                py::array_t<double, py::array::c_style | py::array::forcecast>,
                int,
                bool>(),
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
        .def("get_current_jacobian", &MultiSpeciesExclusionProcess::get_current_jacobian)
        .def("get_left_eigenvector_matrix", &MultiSpeciesExclusionProcess::get_left_eigenvector_matrix)
        .def("get_left_eigenvector_matrix_complex", &MultiSpeciesExclusionProcess::get_left_eigenvector_matrix_complex)
        .def("get_normal_mode_fields", &MultiSpeciesExclusionProcess::get_normal_mode_fields)
        .def("get_normal_mode_heights", &MultiSpeciesExclusionProcess::get_normal_mode_heights)
        .def("normal_mode_height_time_series", &MultiSpeciesExclusionProcess::normal_mode_height_time_series,
            py::arg("n_samples") = 1000,
            py::arg("sample_every") = 1)
        .def("fourier_time_series", &MultiSpeciesExclusionProcess::fourier_time_series,
            py::kw_only(),
            py::arg("n_samples") = 60000,
            py::arg("store_history") = false,
            py::arg("get_projection") = false,
            py::arg("sample_every") = 1,
            py::arg("mode") = 1);
}