#ifndef UTILS_QUAD_GAUSS_GEGENBAUER_QUADRATURE_HPP_
#define UTILS_QUAD_GAUSS_GEGENBAUER_QUADRATURE_HPP_

#include <common/exception_handling.hpp>
#include "gauss_jacobi_quadrature.hpp"

namespace utils{
namespace quad{
namespace gauss{

template <typename T, typename backend = linalg::blas_backend>
class gegenbauer : public jacobi<T, backend>
{
public:
    gegenbauer(size_t N, T alpha, bool m = true) : jacobi<T, backend>(N, alpha, alpha, m) {}

    gegenbauer(const gegenbauer& o) = default;
    gegenbauer(gegenbauer&& o) = default;

    gegenbauer& operator=(const gegenbauer& o) = default;
    gegenbauer& operator=(gegenbauer&& o) = default;
};

}
}
}
   

#endif 

