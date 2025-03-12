#ifndef LINALG_ALGEBRA_OVERLOADS_EXPONENTIAL_HPP
#define LINALG_ALGEBRA_OVERLOADS_EXPONENTIAL_HPP

namespace linalg
{

template <typename T> elemental_exp_return_type<T> elemental_exp(const T& a){return elemental_exp_type<T>(elemental_exp_unary_type<T>(a), a.shape());}

}   //namespace linalg

#endif //LINALG_ALGEBRA_OVERLOADS_EXPONENTIAL_HPP//

