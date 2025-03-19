/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2025 NPL Management Limited
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License
 */

#ifndef PYTTN_TTNS_LIB_SOP_COEFF_TYPE_HPP_
#define PYTTN_TTNS_LIB_SOP_COEFF_TYPE_HPP_

#include <functional>
#include <utility>

#include <common/tmp_funcs.hpp>

namespace ttns
{

    namespace literal
    {

        template <typename T>
        class coeff;
        template <typename T>
        std::ostream &operator<<(std::ostream &os, const coeff<T> &op);

        template <typename T>
        class coeff
        {
        public:
            using real_type = typename tmp::get_real_type<T>::type;
            using function_type = std::function<T(real_type)>;
            using real_function_type = std::function<real_type(real_type)>;

        public:
            coeff() : m_constant(T(0.0)), m_funcs() {}
            // constructors
            template <typename U, typename = typename std::enable_if<tmp::is_number<U>::value, void>::type>
            coeff(const U &c) : m_constant(c), m_funcs() {}

            template <typename U>
            coeff(const std::function<U(real_type)> &f) : m_constant(T(0.0)), m_funcs()
            {
                m_funcs.push_back(std::make_pair(T(1.0), function_type([f](real_type t)
                                                                       { return T(f(t)); })));
            }

            coeff(function_type &&f) : m_constant(T(0.0)), m_funcs() { m_funcs.push_back(std::make_pair(T(1.0), std::move(f))); }

            template <typename U, typename = typename std::enable_if<tmp::is_number<U>::value, void>::type>
            coeff(const coeff<U> &o)
            {
                m_constant = o.constant();
                m_funcs.resize(o.funcs().size());
                for (size_t i = 0; i < o.funcs().size(); ++i)
                {
                    const auto &ai = o.funcs()[i];
                    const auto &aif = std::get<1>(ai);
                    m_funcs.push_back(std::make_pair(T(std::get<0>(ai)), function_type([aif](real_type t)
                                                                                       { return T(aif(t)); })));
                }
            }

            coeff(const coeff &o) = default;
            coeff(coeff &&o) = default;

            // assignment operators
            coeff &operator=(const coeff &o) = default;
            coeff &operator=(coeff &&o) = default;

            template <typename U, typename = typename std::enable_if<tmp::is_number<U>::value, void>::type>
            coeff &operator=(const coeff<U> &o)
            {
                m_funcs.clear();
                m_constant = o.constant();
                m_funcs.resize(o.funcs().size());
                for (size_t i = 0; i < o.funcs().size(); ++i)
                {
                    const auto &ai = o.funcs()[i];
                    const auto &aif = std::get<1>(ai);
                    m_funcs.push_back(std::make_pair(T(std::get<0>(ai)), function_type([aif](real_type t)
                                                                                       { return T(aif(t)); })));
                }
                return *this;
            }

            template <typename U, typename = typename std::enable_if<tmp::is_number<U>::value, void>::type>
            coeff &operator=(const T &c)
            {
                m_funcs.clear();
                m_constant = c;
                return *this;
            }

            coeff &operator=(const function_type &f)
            {
                m_funcs.clear();
                m_constant = T(0.0);
                m_funcs.push_back(std::make_pair(T(1.0), f));
                return *this;
            }

            coeff &operator=(function_type &&f)
            {
                m_funcs.clear();
                m_constant = T(0.0);
                m_funcs.push_back(std::make_pair(T(1.0), std::move(f)));
                return *this;
            }

            const T &constant() const { return m_constant; }
            T &constant() { return m_constant; }

            const std::vector<std::pair<T, function_type>> &funcs() const { return m_funcs; }
            std::vector<std::pair<T, function_type>> &funcs() { return m_funcs; }

        public:
            // arithemetic operators for updating coefficients types.
            // inplace addition, subtraction, multiplication and division of scalars
            template <typename U, typename = typename std::enable_if<tmp::is_number<U>::value, void>::type>
            coeff &operator+=(const U &c)
            {
                m_constant += c;
                return *this;
            }
            template <typename U, typename = typename std::enable_if<tmp::is_number<U>::value, void>::type>
            coeff &operator-=(const U &c)
            {
                m_constant -= c;
                return *this;
            }
            template <typename U, typename = typename std::enable_if<tmp::is_number<U>::value, void>::type>
            coeff &operator*=(const U &c)
            {
                m_constant *= c;
                for (size_t i = 0; i < m_funcs.size(); ++i)
                {
                    std::get<0>(m_funcs[i]) *= c;
                }
                return *this;
            }
            template <typename U, typename = typename std::enable_if<tmp::is_number<U>::value, void>::type>
            coeff &operator/=(const U &c)
            {
                m_constant /= c;
                for (size_t i = 0; i < m_funcs.size(); ++i)
                {
                    std::get<0>(m_funcs[i]) /= c;
                }
                return *this;
            }

            // inplace addition and subtraction of other coeffs
            coeff &operator+=(const coeff<T> &o)
            {
                m_constant += o.m_constant;
                for (size_t i = 0; i < o.m_funcs.size(); ++i)
                {
                    m_funcs.push_back(o.m_funcs[i]);
                }
                return *this;
            }
            coeff &operator-=(const coeff<T> &o)
            {
                m_constant -= o.m_constant;
                for (size_t i = 0; i < o.m_funcs.size(); ++i)
                {
                    m_funcs.push_back(std::make_pair(-1.0 * std::get<0>(o.m_funcs[i]), std::get<1>(o.m_funcs[i])));
                }
                return *this;
            }

            coeff &operator*=(const coeff<T> &o)
            {
                if (o.m_funcs.size() == 0)
                {
                    m_constant *= o.m_constant;
                    if (linalg::abs(o.m_constant) < 1e-14)
                    {
                        m_funcs.clear();
                    }
                    else
                    {
                        for (size_t i = 0; i < m_funcs.size(); ++i)
                        {
                            std::get<0>(m_funcs[i]) *= o.m_constant;
                        }
                    }
                }
                else if (m_funcs.size() == 0)
                {
                    if (linalg::abs(m_constant) < 1e-14)
                    {
                    }
                    else
                    {
                        for (size_t i = 0; i < o.m_funcs.size(); ++i)
                        {
                            m_funcs[i] = std::make_pair(m_constant * std::get<0>(o.m_funcs[i]), std::get<1>(o.m_funcs[i]));
                        }

                        m_constant *= o.m_constant;
                    }
                }
                else
                {
                    std::vector<std::pair<T, function_type>> funcs;
                    // now add on the acoeff terms
                    if (linalg::abs(m_constant) > 1e-14)
                    {
                        for (size_t i = 0; i < o.m_funcs.size(); ++i)
                        {
                            const auto &bi = o.m_funcs[i];
                            funcs.push_back(std::make_pair(std::get<0>(bi) * m_constant, std::get<1>(bi)));
                        }
                    }
                    if (linalg::abs(o.m_constant) > 1e-14)
                    {
                        for (size_t i = 0; i < m_funcs.size(); ++i)
                        {
                            const auto &ai = m_funcs[i];
                            funcs.push_back(std::make_pair(std::get<0>(ai) * o.m_constant, std::get<1>(ai)));
                        }
                    }
                    for (size_t i = 0; i < m_funcs.size(); ++i)
                    {
                        const auto &ai = m_funcs[i];
                        const auto &aif = std::get<1>(ai);
                        for (size_t j = 0; j < o.m_funcs.size(); ++j)
                        {
                            const auto &bj = o.m_funcs[j];
                            const auto &bjf = std::get<1>(bj);
                            funcs.push_back(std::make_pair(std::get<0>(ai) * std::get<0>(bj), function_type([aif, bjf](real_type t)
                                                                                                            { return aif(t) * bjf(t); })));
                        }
                    }
                    m_funcs.clear();
                    m_funcs = funcs;
                    m_constant *= o.m_constant;
                }
                return *this;
            }

        public:
            void clear()
            {
                m_constant = T(0.0);
                m_funcs.clear();
            }

            bool is_zero(real_type tol = 1e-14) const
            {
                if (m_funcs.size() == 0)
                {
                    return linalg::abs(m_constant) < tol;
                }
                return false;
            }

            bool is_positive() const { return linalg::real(m_constant) >= 0; }

        public:
            // Functions for accessing the coefficient
            T operator()(real_type t) const
            {
                T ret = m_constant;
                for (size_t i = 0; i < m_funcs.size(); ++i)
                {
                    ret += std::get<0>(m_funcs[i]) * (std::get<1>(m_funcs[i])(t));
                }
                return ret;
            }

            bool is_time_dependent() const { return m_funcs.size() > 0; }
            friend std::ostream &operator<< <T>(std::ostream &os, const coeff<T> &op);

#ifdef CEREAL_LIBRARY_FOUND
        public:
            template <typename archive>
            void serialize(archive &ar)
            {
                CALL_AND_HANDLE(ar(cereal::make_nvp("coeff", m_constant)), "Failed to serialise sum of product operator.  Failed to serialise array of product operators.");
                CALL_AND_HANDLE(ar(cereal::make_nvp("funcs", m_funcs)), "Failed to serialise sum of product operator.  Failed to serialise array of product operators.");
            }
#endif
        protected:
            T m_constant;
            std::vector<std::pair<T, function_type>> m_funcs;
        };

        template <typename T>
        std::ostream &operator<<(std::ostream &os, const ttns::literal::coeff<T> &op)
        {
            os << "(" << op.constant() << " + " << op.funcs().size() << " functions)";
            return os;
        }

    }
}

// arithemetic operators for updating coefficients types.
// addition, subtraction, multiplication and division of scalars
template <typename T, typename U, typename V = decltype(T() + U()), typename = typename std::enable_if<tmp::is_number<U>::value and tmp::is_number<T>::value, void>::type>
ttns::literal::coeff<V> operator+(const ttns::literal::coeff<T> &o, const U &c)
{
    ttns::literal::coeff<V> ret(o);
    ret += c;
    return ret;
}
template <typename T, typename U, typename V = decltype(T() + U()), typename = typename std::enable_if<tmp::is_number<U>::value and tmp::is_number<T>::value, void>::type>
ttns::literal::coeff<V> operator+(const U &c, const ttns::literal::coeff<T> &o)
{
    ttns::literal::coeff<V> ret(o);
    ret += c;
    return ret;
}
template <typename T, typename U, typename V = decltype(T() + U()), typename = typename std::enable_if<tmp::is_number<U>::value and tmp::is_number<T>::value, void>::type>
ttns::literal::coeff<V> operator-(const ttns::literal::coeff<T> &o, const U &c)
{
    ttns::literal::coeff<V> ret(o);
    ret -= c;
    return ret;
}
template <typename T, typename U, typename V = decltype(T() + U()), typename = typename std::enable_if<tmp::is_number<U>::value and tmp::is_number<T>::value, void>::type>
ttns::literal::coeff<V> operator-(const U &c, const ttns::literal::coeff<T> &o)
{
    ttns::literal::coeff<V> ret(c);
    ret -= o;
    return ret;
}

template <typename T, typename U, typename V = decltype(T() + U()), typename = typename std::enable_if<tmp::is_number<U>::value and tmp::is_number<T>::value, void>::type>
ttns::literal::coeff<V> operator*(const ttns::literal::coeff<T> &o, const U &c)
{
    ttns::literal::coeff<V> ret(o);
    ret *= c;
    return ret;
}
template <typename T, typename U, typename V = decltype(T() + U()), typename = typename std::enable_if<tmp::is_number<U>::value and tmp::is_number<T>::value, void>::type>
ttns::literal::coeff<V> operator*(const U &c, const ttns::literal::coeff<T> &o)
{
    ttns::literal::coeff<V> ret(o);
    ret *= c;
    return ret;
}

template <typename T, typename U, typename V = decltype(T() + U()), typename = typename std::enable_if<tmp::is_number<U>::value and tmp::is_number<T>::value, void>::type>
ttns::literal::coeff<V> operator/(const ttns::literal::coeff<T> &o, const U &c)
{
    ttns::literal::coeff<V> ret(o);
    ret /= c;
    return ret;
}

// addition of functions
template <typename T, typename U, typename V = decltype(T() + U()), typename = typename std::enable_if<tmp::is_number<U>::value and tmp::is_number<T>::value, void>::type>
ttns::literal::coeff<V> operator+(const ttns::literal::coeff<T> &o, const std::function<U(const typename tmp::get_real_type<T>::type &)> &c)
{
    ttns::literal::coeff<V> ret(o);
    ret += c;
    return ret;
}
template <typename T, typename U, typename V = decltype(T() + U()), typename = typename std::enable_if<tmp::is_number<U>::value and tmp::is_number<T>::value, void>::type>
ttns::literal::coeff<V> operator+(const std::function<T(const typename tmp::get_real_type<T>::type &)> &c, const ttns::literal::coeff<U> &o)
{
    ttns::literal::coeff<V> ret(o);
    ret += c;
    return ret;
}

template <typename T, typename U, typename V = decltype(T() + U()), typename = typename std::enable_if<tmp::is_number<U>::value and tmp::is_number<T>::value, void>::type>
ttns::literal::coeff<V> operator+(const ttns::literal::coeff<T> &o, std::function<U(const typename tmp::get_real_type<T>::type &)> &&c)
{
    ttns::literal::coeff<V> ret(o);
    ret += c;
    return ret;
}
template <typename T, typename U, typename V = decltype(T() + U()), typename = typename std::enable_if<tmp::is_number<U>::value and tmp::is_number<T>::value, void>::type>
ttns::literal::coeff<V> operator+(std::function<T(const typename tmp::get_real_type<T>::type &)> &&c, const ttns::literal::coeff<U> &o)
{
    ttns::literal::coeff<V> ret(o);
    ret += c;
    return ret;
}

template <typename T, typename U, typename V = decltype(T() + U()), typename = typename std::enable_if<tmp::is_number<U>::value and tmp::is_number<T>::value, void>::type>
ttns::literal::coeff<V> operator-(const ttns::literal::coeff<T> &o, const std::function<T(const typename tmp::get_real_type<U>::type &)> &c)
{
    ttns::literal::coeff<V> ret(o);
    ret -= c;
    return ret;
}
template <typename T, typename U, typename V = decltype(T() + U()), typename = typename std::enable_if<tmp::is_number<U>::value and tmp::is_number<T>::value, void>::type>
ttns::literal::coeff<V> operator-(const std::function<T(const typename tmp::get_real_type<T>::type &)> &c, const ttns::literal::coeff<U> &o)
{
    ttns::literal::coeff<V> ret(c);
    ret -= o;
    return ret;
}

template <typename T, typename U, typename V = decltype(T() + U()), typename = typename std::enable_if<tmp::is_number<U>::value and tmp::is_number<T>::value, void>::type>
ttns::literal::coeff<V> operator-(const ttns::literal::coeff<T> &o, std::function<U(const typename tmp::get_real_type<T>::type &)> &&c)
{
    ttns::literal::coeff<V> ret(o);
    ret -= std::move(c);
    return ret;
}
template <typename T, typename U, typename V = decltype(T() + U()), typename = typename std::enable_if<tmp::is_number<U>::value and tmp::is_number<T>::value, void>::type>
ttns::literal::coeff<V> operator-(std::function<T(const typename tmp::get_real_type<T>::type &)> &&c, const ttns::literal::coeff<U> &o)
{
    ttns::literal::coeff<V> ret(std::move(c));
    ret -= o;
    return ret;
}

template <typename T, typename U, typename V = decltype(T() + U()), typename = typename std::enable_if<tmp::is_number<U>::value and tmp::is_number<T>::value, void>::type>
ttns::literal::coeff<V> operator+(const ttns::literal::coeff<T> &c, const ttns::literal::coeff<U> &o)
{
    ttns::literal::coeff<V> ret(o);
    ret += c;
    return ret;
}

template <typename T, typename U, typename V = decltype(T() + U()), typename = typename std::enable_if<tmp::is_number<U>::value and tmp::is_number<T>::value, void>::type>
ttns::literal::coeff<V> operator-(const ttns::literal::coeff<T> &c, const ttns::literal::coeff<U> &o)
{
    ttns::literal::coeff<V> ret(c);
    ret -= o;
    return ret;
}

template <typename T, typename U, typename V = decltype(T() + U()), typename = typename std::enable_if<tmp::is_number<U>::value and tmp::is_number<T>::value, void>::type>
ttns::literal::coeff<V> operator*(const ttns::literal::coeff<T> &a, const ttns::literal::coeff<U> &b)
{
    using real_type = typename tmp::get_real_type<T>::type;
    ttns::literal::coeff<V> ret(a.constant() * b.constant());
    using function_type = typename ttns::literal::coeff<V>::function_type;
    // now add on the acoeff terms
    if (linalg::abs(a.constant()) > 1e-14)
    {
        for (size_t i = 0; i < b.funcs().size(); ++i)
        {
            const auto &bi = b.funcs()[i];
            const auto &bif = std::get<1>(bi);
            ret.funcs().push_back(std::make_pair(std::get<0>(bi) * a.constant(), function_type([bif](real_type t)
                                                                                               { return bif(t); })));
        }
    }
    if (linalg::abs(b.constant()) > 1e-14)
    {
        for (size_t i = 0; i < a.funcs().size(); ++i)
        {
            const auto &ai = a.funcs()[i];
            const auto &aif = std::get<1>(ai);
            ret.funcs().push_back(std::make_pair(std::get<0>(ai) * b.constant(), function_type([aif](real_type t)
                                                                                               { return aif(t); })));
        }
    }
    for (size_t i = 0; i < a.funcs().size(); ++i)
    {
        const auto &ai = a.funcs()[i];
        const auto &aif = std::get<1>(ai);
        for (size_t j = 0; j < b.funcs().size(); ++j)
        {
            const auto &bj = b.funcs()[j];
            const auto &bjf = std::get<1>(bj);
            ret.funcs().push_back(std::make_pair(std::get<0>(ai) * std::get<0>(bj), function_type([aif, bjf](real_type t)
                                                                                                  { return aif(t) * bjf(t); })));
        }
    }
    return ret;
}

#endif // PYTTN_TTNS_LIB_SOP_COEFF_TYPE_HPP_
