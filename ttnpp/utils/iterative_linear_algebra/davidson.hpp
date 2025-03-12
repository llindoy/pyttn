#ifndef UTILS_DAVIDSON_HPP
#define UTILS_DAVIDSON_HPP

#include <linalg/linalg.hpp>
#include "bicgstabl.hpp"

namespace utils
{
template <typename T, typename backend>
class davidson_solver
{
    using value_type = T;
    using real_type = typename linalg::get_real_type<T>::type;

public:


protected:
    size_t m_max_iter;
    size_t m_max_search_space;
    size_t m_initial_guess_size;

    real_type m_tol;
    real_type m_linear_solver_tol;

};

}   //namespace utils

#endif

