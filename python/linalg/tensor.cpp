#include "tensor.hpp"

void initialise_tensors(py::module& m)
{
    using real_type = double;
    using complex_type = linalg::complex<real_type>;
    init_tensor<real_type, 1>(m, "vector_real");     
    init_tensor<real_type, 2>(m, "matrix_real");     
    init_tensor<real_type, 3>(m, "tensor_3_real");     
    init_tensor<real_type, 4>(m, "tensor_4_real");     

    init_tensor<complex_type, 1>(m, "vector_complex");     
    init_tensor<complex_type, 2>(m, "matrix_complex");     
    init_tensor<complex_type, 3>(m, "tensor_3_complex");     
    init_tensor<complex_type, 4>(m, "tensor_4_complex");     
}
