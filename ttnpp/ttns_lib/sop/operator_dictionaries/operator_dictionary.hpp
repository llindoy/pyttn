#ifndef TTNS_OPERATOR_DICTIONARY_HPP
#define TTNS_OPERATOR_DICTIONARY_HPP

#include <unordered_map>
#include <string>

#include "default_operator_dictionaries.hpp"
#include <utils/occupation_number_basis_indexing.hpp>

#include "../../operators/site_operators/site_operator.hpp"
#include "../../operators/site_operators/matrix_operators.hpp"
#include "../system_information.hpp"

namespace ttns
{

template <typename T, typename backend = linalg::blas_backend> 
class site_operator;

template <typename T, typename backend = linalg::blas_backend>
class operator_from_default_dictionaries
{
    using op_type = ops::primitive<T, backend>;
public:
    static std::shared_ptr<op_type> query(const std::string& label, std::shared_ptr<utils::occupation_number_basis> basis, const mode_type& type, bool use_sparse, size_t mode_index = 0)
    {
        size_t hilbert_space_dimension = basis->nstates();
        if(label == "id" || label == "Id" || label == "1")
        {
            return std::make_shared<ops::identity<T, backend>>(hilbert_space_dimension);
        }
        else
        {
            std::shared_ptr<single_site_operator<T>> s_op = query_default_operator_dictionary<T>(type, label);
            if(use_sparse)
            {
                if(s_op->is_diagonal())
                {
                    linalg::diagonal_matrix<T> Mhost;
                    s_op->as_diagonal(basis, mode_index, Mhost);
                    linalg::diagonal_matrix<T, backend> M(Mhost);
                    return std::make_shared<ops::diagonal_matrix_operator<T, backend>>(M);
                }
                else if(s_op->is_sparse())
                {
                    linalg::csr_matrix<T> Mhost;
                    s_op->as_csr(basis, mode_index, Mhost);                    
                    linalg::csr_matrix<T, backend> M(Mhost);

                    return std::make_shared<ops::sparse_matrix_operator<T, backend>>(M);
                }
                else
                {
                    linalg::matrix<T> Mhost;
                    s_op->as_dense(basis, mode_index, Mhost);
                    linalg::matrix<T, backend> M(Mhost);
                    return std::make_shared<ops::dense_matrix_operator<T, backend>>(M);
                }
            }
            else
            {
                linalg::matrix<T> Mhost;
                s_op->as_dense(basis, mode_index, Mhost);
                linalg::matrix<T, backend> M(Mhost);
                return std::make_shared<ops::dense_matrix_operator<T, backend>>(M);
            }
        }
    }
};

template <typename T, typename backend>
class operator_dictionary
{
public:
    using elem_type = std::unordered_map<std::string, site_operator<T, backend>>;
    using dict_type = std::vector<elem_type>;
    using op_type = ops::primitive<T, backend>;

    operator_dictionary(){}
    operator_dictionary(size_t N) : m_dict(N){}
    operator_dictionary(const dict_type& o) : m_dict(o) {}

    operator_dictionary(const operator_dictionary& o) = default;
    operator_dictionary(operator_dictionary&& o) = default;

    operator_dictionary& operator=(const dict_type& o) {m_dict = o; return *this;}
    operator_dictionary& operator=(const operator_dictionary& o) = default;
    operator_dictionary& operator=(operator_dictionary&& o) = default;

    void clear(){m_dict.clear();}
    void resize(size_t N)
    {
        m_dict.resize(N);
    }

    elem_type& operator[](size_t i){return m_dict[i];}
    const elem_type& operator[](size_t i) const{return m_dict[i];}

    const elem_type& site_dictionary(size_t i) const{ASSERT(i < m_dict.size(), "Failed to access site dictionary index out of bounds."); return m_dict[i];}

    void insert(size_t nu, const std::string& label, const site_operator<T, backend>& op)
    {
        ASSERT(nu < m_dict.size(), "Failed to query element from operator dictionary.  Index out of bounds.");
        auto it = m_dict[nu].find(label);

        //if the object is already in the map we don't do anything
        if(it == m_dict[nu].end()){m_dict[nu].insert({label, op});}
        
    }

    const site_operator<T, backend>& operator()(size_t nu, const std::string& label) const
    {
        ASSERT(nu < m_dict.size(), "Failed to query element from operator dictionary.  Index out of bounds.");
        auto elem = m_dict[nu].find(label);
        ASSERT(elem != m_dict[nu].end(), "Failed to access element from operator dictionary.  Index out of bounds.");
        return elem->second;
    }

    std::shared_ptr<op_type> query(size_t nu, const std::string& label) const
    {
        ASSERT(nu < m_dict.size(), "Failed to query element from operator dictionary.  Index out of bounds.");
        auto elem = m_dict[nu].find(label);
        if(elem == m_dict[nu].end()){return std::shared_ptr<op_type>(nullptr);}
        else
        {
            return elem->second.op();
        }
    }

    size_t size() const{return m_dict.size();}
    size_t nmodes() const{return m_dict.size();}
protected:
    dict_type m_dict;
};


}   //namespace ttns

#endif
