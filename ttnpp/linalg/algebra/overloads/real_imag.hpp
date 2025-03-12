#ifndef LINALG_ALGEBRA_OVERLOADS_REAL_IMAG_HPP
#define LINALG_ALGEBRA_OVERLOADS_REAL_IMAG_HPP

namespace linalg
{

template <typename T> real_return_type<T> real(const T& a){return real_type<T>(real_unary_type<T>(a), a.shape());}
template <typename T> imag_return_type<T> imag(const T& a){return imag_type<T>(imag_unary_type<T>(a), a.shape());}
template <typename T> norm_return_type<T> elementwise_norm(const T& a){return norm_type<T>(norm_unary_type<T>(a), a.shape());}
template <typename T> arg_return_type<T>  elementwise_arg(const T& a) {return arg_type<T>(arg_unary_type<T>(a), a.shape());}
template <typename T> unit_polar_return_type<T>  unit_polar(const T& a) {return unit_polar_type<T>(unit_polar_unary_type<T>(a), a.shape());}

}   //namespace linalg

#endif //LINALG_ALGEBRA_OVERLOADS_REAL_IMAG_HPP//

