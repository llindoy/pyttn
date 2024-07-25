#ifndef UTILS_QUAD_GAUSS_CHEBYSHEV_QUADRATURE_HPP_
#define UTILS_QUAD_GAUSS_CHEBYSHEV_QUADRATURE_HPP_

#include <common/exception_handling.hpp>
#include "gauss_gegenbauer_quadrature.hpp"

namespace utils{
namespace quad{
namespace gauss{

template <typename T, typename backend = linalg::blas_backend>
class chebyshev : public gegenbauer<T, backend>
{
public:
    chebyshev(size_t N, bool m = true) : gegenbauer<T, backend>(N, -0.5, m) {}

    chebyshev(const chebyshev& o) = default;
    chebyshev(chebyshev&& o) = default;

    chebyshev& operator=(const chebyshev& o) = default;
    chebyshev& operator=(chebyshev&& o) = default;
};

template <typename T, typename backend = linalg::blas_backend>
class chebyshev_second : public gegenbauer<T, backend>
{
public:
    chebyshev_second(size_t N, bool m = true) : gegenbauer<T, backend>(N, 0.5, m) {}

    chebyshev_second(const chebyshev_second& o) = default;
    chebyshev_second(chebyshev_second&& o) = default;

    chebyshev_second& operator=(const chebyshev_second& o) = default;
    chebyshev_second& operator=(chebyshev_second&& o) = default;
};

}
}   //quad
}   //namespace utils

#endif  //UTILS_QUAD_GAUSS_CHEBYSHEV_QUADRATURE_HPP_//

