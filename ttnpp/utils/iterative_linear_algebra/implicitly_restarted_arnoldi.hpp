#ifndef UTILS_IMPLICITLY_RESTARTED_ARNOLDI_HPP
#define UTILS_IMPLICITLY_RESTARTED_ARNOLDI_HPP

namespace utils
{

/*
 *  A class wrapping the arpack functions for evaluating a few eigenvalues of sparse matrices
 */
template <typename T, typename backend = linalg::blas_backend>
class implicitly_restarted_arnoldi;


template <typename T>
class implicitly_restarted_arnoldi<T, linalg::blas_backend>
{
public:

protected:

};

//Note that we do not currently have a cuda implementation of this algorithm, as a consequence at present we can only evaluate ground state configurations using the CPU
//branch of the code.

}


#endif

