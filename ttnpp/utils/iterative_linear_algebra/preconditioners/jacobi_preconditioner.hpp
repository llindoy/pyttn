#ifndef UTILS_JACOBI_PRECONDITIONER_HPP
#define UTILS_JACOBI_PRECONDITIONER_HPP

#include <common/exception_handling.hpp>
#include "preconditioner_base.hpp"

#include <linalg/linalg.hpp>

namespace utils
{

namespace preconditioner
{

template <typename T, typename backend>
class jacobi : public preconditioner<T, backend>
{
public: 
    jacobi(){}
    jacobi(const linalg::diagonal_matrix<T, backend>& mat) : m_mat(mat), m_temp(mat.size(0)){}
    jacobi(const jacobi& o) = default;
    jacobi(jacobi&& o) = default;

    jacobi& operator=(const jacobi& o) = default;
    jacobi& operator=(jacobi&& o) = default;

    template <typename Vin>
    void apply(Vin& x)
    {
        ASSERT(x.size() == m_temp.size(), "Failed to apply preconditioner.  Invalid shape of input array.");
        auto xi = x.reinterpret_shape(m_temp.shape());
        CALL_AND_HANDLE(m_temp = m_mat*xi, "Failed to apply the preconditioner matrix.")
        CALL_AND_HANDLE(x = m_temp, "Failed to copy the preconditioned result into the return location.");
    }

    void clear()
    {
        m_mat.clear();
        m_temp.clear();
    }


    void initialise(const linalg::diagonal_matrix<T, backend>& mat)
    {
        CALL_AND_HANDLE(m_mat = mat, "Failed to copy preconditioning matrix.");
        CALL_AND_HANDLE(m_temp.resize(m_mat.size(0)), "Failed to resize working buffer.");
    }


protected:
    linalg::diagonal_matrix<T, backend> m_mat;
    linalg::vector<T, backend> m_temp;

};

}   //namespace preconditioner
}   //namespace utils

#endif

