#include "sparseMatrix.hpp"

void initialise_sparse_matrices(py::module& m)
{
    using real_type = double;
    using complex_type = linalg::complex<real_type>;
    init_csr_matrix<real_type>(m, "csr_matrix_real");
    init_csr_matrix<complex_type>(m, "csr_matrix_complex");
    init_diagonal_matrix<real_type>(m, "diagonal_matrix_real");
    init_diagonal_matrix<complex_type>(m, "diagonal_matrix_complex");
}
