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

#ifndef PYTTN_TTNS_LIB_SOP_STATE_HPP_
#define PYTTN_TTNS_LIB_SOP_STATE_HPP_

#include <linalg/linalg.hpp>

#include <iostream>
#include <list>
#include <vector>
#include <string>

namespace ttns
{
    class stateStr
    {
        using elem_type = std::pair<size_t, size_t>;

    public:
        stateStr() {}
        stateStr(const std::vector<size_t> &N)
        {
            prepare_state(N);
        }

        stateStr(const stateStr &o) = default;
        stateStr(stateStr &&o) = default;

        stateStr &operator=(const stateStr &o) = default;
        stateStr &operator=(stateStr &&o) = default;

        void clear()
        {
            m_state.clear();
            m_N = 0;
            m_hash = 0;
        }

        void resize(size_t N) 
        {
            if(m_state.size() > 0)
            {
                if(N < std::get<0>(m_state[m_state.size()-1]))
                {
                    m_state.clear();
                } 
            }
            m_N = N; 
        }

        const std::vector<elem_type> &operator()() const { return m_state; }
        std::vector<elem_type> &operator()() { return m_state; }

        elem_type &operator[](size_t i) { return m_state[i]; }
        const elem_type &operator[](size_t i) const { return m_state[i]; }

        size_t size() const { return m_N; }
        size_t nnz() const { return m_state.size(); }

        std::vector<size_t> state() const
        {
            std::vector<size_t> _state(m_N);
            std::fill(_state.begin(), _state.end(), 0);
            for (size_t i = 0; i < m_state.size(); ++i)
            {
                _state[std::get<0>(m_state[i])] = std::get<1>(m_state[i]);
            }
            return _state;
        }

        bool hasHash() const { return m_has_hash; }
        std::size_t hash() const { return m_hash; }
        void get_hash()
        {
            std::hash<std::string> hashobj;
            m_hash = hashobj(std::string(*this));
            m_has_hash = true;
        }

    public:
        operator std::string() const
        {
            std::string ret;
            for (const auto &t : m_state)
            {
                ret += std::to_string(std::get<0>(t)) + std::string("_") + std::to_string(std::get<1>(t));
            }
            return ret;
        }

    protected:
        void prepare_state(const std::vector<size_t> &N)
        {
            size_t nstate = 0;
            for (size_t i = 0; i < N.size(); ++i)
            {
                if (N[i] != 0)
                {
                    nstate += 1;
                }
            }
            m_state.clear();
            m_state.resize(nstate);

            size_t counter = 0;
            for (size_t i = 0; i < N.size(); ++i)
            {
                if (N[i] != 0)
                {
                    m_state[counter] = std::make_pair(i, N[i]);
                    ++counter;
                }
            }
            m_N = N.size();

            get_hash();
        }

    protected:
        std::vector<std::pair<size_t, size_t>> m_state;
        size_t m_N;

        bool m_has_hash = false;
        std::size_t m_hash;
    };

    inline bool operator==(const stateStr &A, const stateStr &B)
    {
        if (A.nnz() != B.nnz())
        {
            return false;
        }
        for (size_t i = 0; i < A.nnz(); ++i)
        {
            if (std::get<0>(A[i]) != std::get<0>(B[i]) || std::get<1>(A[i]) != std::get<1>(B[i]))
            {
                return false;
            }
        }
        return true;
    }

    inline bool operator!=(const stateStr &A, const stateStr &B) { return !(A == B); }
}

template <>
struct std::hash<ttns::stateStr>
{
    std::size_t operator()(const ttns::stateStr &k) const
    {
        if (k.hasHash())
        {
            return k.hash();
        }
        else
        {
            return std::hash<std::string>()(std::string(k));
        }
    }
};

inline std::ostream &operator<<(std::ostream &os, const ttns::stateStr &_state)
{
    auto state = _state.state();
    std::string sep("");
    for (size_t i = 0; i < state.size(); ++i)
    {
        os << sep << state[i];
        if (i == 0)
        {
            sep = " ";
        }
    }
    return os;
}

namespace ttns
{
    template <typename T>
    class sepState
    {
        using real_type = typename linalg::get_real_type<T>::type;

    public:
        sepState() : m_coeff(T(0.0)) {}
        sepState(const std::vector<size_t> &N) : m_coeff(T(1.0)), m_state(N) {}
        sepState(const T &coeff, const std::vector<size_t> &N) : m_coeff(coeff), m_state(N) {}

        sepState(const stateStr &N) : m_coeff(T(1.0)), m_state(N) {}
        sepState(const T &coeff, const stateStr &N) : m_coeff(coeff), m_state(N) {}

        sepState(const sepState &o) = default;
        sepState(sepState &&o) = default;

        template <typename U>
        sepState(const sepState<U> &o) : m_coeff(o.coeff()), m_state(o.state()) {}

        sepState &operator=(const stateStr &o)
        {
            m_state = o;
            m_coeff = T(1.0);
            return *this;
        }
        sepState &operator=(const sepState &o) = default;
        sepState &operator=(sepState &&o) = default;

        template <typename U>
        sepState &operator=(const sepState<U> &o)
        {
            m_state = o.state();
            return *this;
        }

        void clear()
        {
            m_state.clear();
            m_coeff = T(0);
        }

        void resize(size_t N) { m_state.resize(N); }

        const stateStr &state() const { return m_state; }
        stateStr &state() { return m_state; }

        const T &coeff() const { return m_coeff; }
        T &coeff() { return m_coeff; }

        size_t size() const { return m_state.size(); }
        size_t nnz() const { return m_state.nnz(); }

        sepState<T> &operator*=(const T &b)
        {
            m_coeff *= b;
            return *this;
        }

        sepState<T> &operator/=(const T &b)
        {
            m_coeff /= b;
            return *this;
        }

        std::vector<size_t> stateRep() const
        {
            return m_state.state();
        }

    protected:
        T m_coeff;
        stateStr m_state;
    };

    template <typename T>
    bool operator==(const sepState<T> &A, const sepState<T> &B)
    {
        return A.coeff() == B.coeff() && A.state() == B.state();
    }

} // namespace ttns

template <typename T>
std::ostream &operator<<(std::ostream &os, const ttns::sepState<T> &op)
{
    os << op.coeff() << " " << op.state();
    return os;
}

template <typename T, typename U, typename = typename std::enable_if<linalg::is_number<T>::value and linalg::is_number<U>::value, void>::type>
ttns::sepState<decltype(T() * U())> operator*(const T &b, const ttns::sepState<U> &o)
{
    ttns::sepState<decltype(T() * U())> ret;
    ret.coeff() = b * o.coeff();
    ret.state() = o.state();

    return ret;
}

template <typename T, typename U, typename = typename std::enable_if<linalg::is_number<T>::value and linalg::is_number<U>::value, void>::type>
ttns::sepState<decltype(T() * U())> operator*(const ttns::sepState<T> &o, const U &b)
{
    ttns::sepState<decltype(T() * U())> ret;
    ret.coeff() = b * o.coeff();
    ret.state() = o.state();
    return ret;
}

template <typename T, typename U, typename = typename std::enable_if<linalg::is_number<T>::value and linalg::is_number<U>::value, void>::type>
ttns::sepState<decltype(T() * U())> operator/(const ttns::sepState<T> &o, const U &b)
{
    ttns::sepState<decltype(T() * U())> ret;
    ret.coeff() = o.coeff() / b;
    ret.state() = o.state();

    return ret;
}

template <typename T, typename = typename std::enable_if<linalg::is_number<T>::value, void>::type>
ttns::sepState<T> operator*(const T &b, const ttns::stateStr &o)
{
    ttns::sepState<T> ret(b, o.state());
    return ret;
}

template <typename T, typename = typename std::enable_if<linalg::is_number<T>::value, void>::type>
ttns::sepState<T> operator*(const ttns::stateStr &o, const T &b)
{
    ttns::sepState<T> ret(b, o.state());
    return ret;
}

template <typename T, typename = typename std::enable_if<linalg::is_number<T>::value, void>::type>
ttns::sepState<T> operator/(const ttns::stateStr &o, const T &b)
{
    ttns::sepState<T> ret(T(1.0) / b, o.state());
    return ret;
}

#include <unordered_map>

namespace ttns
{
    template <typename T>
    class ket
    {
    public:
        using elem_type = stateStr;
        using container_type = std::unordered_map<elem_type, T>;
        using iterator = typename container_type::iterator;
        using const_iterator = typename container_type::const_iterator;

    public:
        ket() {}

        ket(const ket &o) = default;

        template <typename U>
        ket(const ket<U> &o)
        {
            for (const auto &t : o)
            {
                m_terms[std::get<0>(t)] = T(std::get<1>(t));
            }
        }


        ket(ket &&o) = default;

        ket &operator=(const ket &o) = default;
        ket &operator=(ket &&o) = default;

        template <typename U>
        ket &operator=(const ket<U> &o)
        {
            m_terms.clear();
            for (const auto &t : o)
            {
                m_terms[std::get<0>(t)] = T(std::get<1>(t));
            }
            return *this;
        }

        void reserve(size_t nt) { m_terms.reserve(nt); }
        void clear()
        {
            m_terms.clear();
        }

        template <typename U>
        ket<T> &operator*=(const U &a)
        {
            for (auto &t : m_terms)
            {
                std::get<1>(t) *= a;
            }
            return *this;
        }

        template <typename U>
        ket<T> &operator/=(const U &a)
        {
            for (auto &t : m_terms)
            {
                std::get<1>(t) /= a;
            }
            return *this;
        }

        ket<T> &operator+=(const stateStr &a)
        {
            insert(sepState<T>(a));
            return *this;
        }

        ket<T> &operator+=(const sepState<T> &a)
        {
            insert(a);
            return *this;
        }

        ket<T> &operator+=(const ket &a)
        {
            for (const auto &t : a)
            {
                insert(t);
            }
            return *this;
        }

        ket<T> &operator-=(const stateStr &a)
        {
            insert(sepState<T>(T(-1), a));
            return *this;
        }

        ket<T> &operator-=(const sepState<T> &a)
        {
            insert(T(-1.0) * a);
            return *this;
        }

        ket<T> &operator-=(const ket &a)
        {
            for (const auto &t : a.m_terms)
            {
                const auto& coeff = std::get<1>(t);
                const auto& term = std::get<0>(t);
                insert(T(-1)*coeff, term);                
            }
            return *this;
        }

        size_t nterms() const { return m_terms.size(); }

        void insert(const elem_type &a) { insert(T(1.0), a); }

        template <typename U>
        void insert(const sepState<U> &a) { insert(a.coeff(), a.state()); }

        template <typename U>
        void insert(const U &v, const elem_type &a)
        {
            m_terms[a] += v;
        }

        void insert(const std::pair<elem_type, T>& a){ insert(std::get<1>(a), std::get<0>(a)); }

        const T& operator[](const elem_type& i)const{return m_terms[i];}
        T& operator[](const elem_type& i){return m_terms[i];}

        bool contains(const elem_type& i)const{return m_terms.find(i) != m_terms.end();}
    public:
        iterator begin() { return iterator(m_terms.begin()); }
        iterator end() { return iterator(m_terms.end()); }
        const_iterator begin() const { return const_iterator(m_terms.begin()); }
        const_iterator end() const { return const_iterator(m_terms.end()); }

    protected:
        container_type m_terms;

    public:
        void prune_zeros(double tol = 1e-15)
        {
            for (auto it = m_terms.begin(); it != m_terms.end();)
            {
                if (linalg::abs(std::get<1>(*it))< tol)
                {
                    it = m_terms.erase(it);
                }
                else
                {
                    it++;
                }
            }
        }
    };

} // namespace ttns

    template <typename T>
    std::ostream &operator<<(std::ostream &os, const ttns::ket<T> &op)
    {
        const auto separator = "";
        const auto *sep = "";
        const auto plus = "+";
        for (const auto &t : op)
        {
            sep = linalg::real(std::get<1>(t))>0 ? plus : separator;
            os << sep << std::get<1>(t) << " " << std::get<0>(t) << std::endl;
        }
        return os;
    }

template <typename T>
inline ttns::ket<T> add(const ttns::stateStr &a, const ttns::stateStr &b)
{
    ttns::ket<T> ret;
    ret.insert(ttns::sepState<T>(a));
    ret.insert(ttns::sepState<T>(b));
    return ret;
}

template <typename T>
ttns::ket<T> operator+(const ttns::sepState<T> &a, const ttns::stateStr &b)
{
    ttns::ket<T> ret;
    ret.insert(a);
    ret.insert(ttns::sepState<T>(b));
    return ret;
}

template <typename T>
ttns::ket<T> operator+(const ttns::stateStr &a, const ttns::sepState<T> &b)
{
    ttns::ket<T> ret;
    ret.insert(ttns::sepState<T>(a));
    ret.insert(b);
    return ret;
}

template <typename T>
ttns::ket<T> operator+(const ttns::stateStr &a, const ttns::ket<T> &b)
{
    ttns::ket<T> ret(b);
    ret.insert(a);
    return ret;
}

template <typename T>
ttns::ket<T> operator+(const ttns::ket<T> &a, const ttns::stateStr &b)
{
    ttns::ket<T> ret(a);
    ret.insert(ttns::sepState<T>(b));
    return ret;
}

template <typename T, typename U>
ttns::ket<decltype(T() * U())> operator+(const ttns::sepState<T> &a, const ttns::sepState<U> &b)
{
    ttns::ket<decltype(T() * U())> ret;
    ret.insert(a);
    ret.insert(b);
    return ret;
}

template <typename T, typename U>
ttns::ket<decltype(T() * U())> operator+(const ttns::sepState<T> &a, const ttns::ket<U> &b)
{
    ttns::ket<decltype(T() * U())> ret(b);
    ret.insert(a);
    return ret;
}

template <typename T, typename U>
ttns::ket<decltype(T() * U())> operator+(const ttns::ket<T> &a, const ttns::sepState<U> &b)
{
    ttns::ket<decltype(T() * U())> ret(a);
    ret.insert(b);
    return ret;
}

template <typename T, typename U>
ttns::ket<decltype(T() * U())> operator+(const ttns::ket<T> &a, const ttns::ket<U> &b)
{
    ttns::ket<decltype(T() * U())> ret(a);
    for (const auto &t : b)
    {
        ret.insert(t);
    }
    return ret;
}

template <typename T>
inline ttns::ket<T> sub(const ttns::stateStr &a, const ttns::stateStr &b)
{
    ttns::ket<T> ret;
    ret.insert(ttns::sepState<T>(a));
    ret.insert(ttns::sepState<T>(T(-1.0), b));
    return ret;
}

template <typename T>
ttns::ket<T> operator-(const ttns::sepState<T> &a, const ttns::stateStr &b)
{
    ttns::ket<T> ret;
    ret.insert(a);
    ret.insert(ttns::sepState<T>(T(-1.0), b));
    return ret;
}

template <typename T>
ttns::ket<T> operator-(const ttns::stateStr &a, const ttns::sepState<T> &b)
{
    ttns::ket<T> ret;
    ret.insert(ttns::sepState<T>(a));
    ret.insert(T(-1.0) * b);
    return ret;
}

template <typename T>
ttns::ket<T> operator-(const ttns::stateStr &a, const ttns::ket<T> &b)
{
    ttns::ket<T> ret;
    ret.insert(a);
    for (const auto &t : b)
    {
        const auto& coeff = std::get<1>(t);
        const auto& term = std::get<0>(t);
        ret.insert(T(-1)*coeff, term);       
    }
    return ret;
}

template <typename T>
ttns::ket<T> operator-(const ttns::ket<T> &a, const ttns::stateStr &b)
{
    ttns::ket<T> ret(a);
    ret.insert(ttns::sepState<T>(T(-1.0), b));

    return ret;
}

template <typename T, typename U>
ttns::ket<decltype(T() * U())> operator-(const ttns::sepState<T> &a, const ttns::sepState<U> &b)
{
    ttns::ket<decltype(T() * U())> ret;
    ret.insert(a);
    ret.insert(U(-1) * b);
    return ret;
}

template <typename T, typename U>
ttns::ket<decltype(T() * U())> operator-(const ttns::sepState<T> &a, const ttns::ket<U> &b)
{
    ttns::ket<decltype(T() * U())> ret;
    ret.insert(a);
    for (const auto &t : b)
    {
        const auto& coeff = std::get<1>(t);
        const auto& term = std::get<0>(t);
        ret.insert(U(-1)*coeff, term);
    }
    return ret;
}

template <typename T, typename U>
ttns::ket<decltype(T() * U())> operator-(const ttns::ket<T> &a, const ttns::sepState<U> &b)
{
    ttns::ket<decltype(T() * U())> ret(a);
    ret.insert(U(-1) * b);
    return ret;
}

template <typename T, typename U>
ttns::ket<decltype(T() * U())> operator-(const ttns::ket<T> &a, const ttns::ket<U> &b)
{
    ttns::ket<decltype(T() * U())> ret(a);
    for (const auto &t : b)
    {
        const auto& coeff = std::get<1>(t);
        const auto& term = std::get<0>(t);
        ret.insert(U(-1)*coeff, term);    
    }
    return ret;
}

#endif // PYTTN_TTNS_LIB_SOP_STATE_HPP_
