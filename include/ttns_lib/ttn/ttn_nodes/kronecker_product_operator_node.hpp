#ifndef TTNS_KRONECKER_PRODUCT_OPERATOR_NODE_HPP
#define TTNS_KRONECKER_PRODUCT_OPERATOR_NODE_HPP

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>
#include <list>

#include <common/exception_handling.hpp>

#include <linalg/linalg.hpp>
#include "hamiltonian_node.hpp"

namespace ttns
{

template <typename T, typename backend> 
class kronecker_product_operator_node_data
{
public:
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type; 
protected:
    using hdata = ttn_node_data<T, backend>;
    using ham_type = hamiltonian_node_data<T, backend>;
    using ham_node_type = typename tree<ham_type>::node_type;

    //objects used to apply the full kronecker product operator
    std::vector<size_type> m_r;
    std::vector<std::vector<std::pair<size_type, size_type> >> m_indices; //for each set of rvalues we store the nu and internal r index for the object
    std::vector<size_type> m_rindices;

    //objects used to apply the kronecker product operator with a single mode missing.  Used in the evaluation of mean fields
    std::vector<std::vector<size_type>> m_r_nu;
    std::vector<std::vector<std::vector<std::pair<size_type, size_type>>>> m_indices_nu;
    std::vector<std::vector<size_type>> m_rindices_nu;

    std::vector<size_type> m_accumulated_indices;

public:
    kronecker_product_operator_node_data() {}
    kronecker_product_operator_node_data(const ham_node_type& ham) 
    {
        CALL_AND_HANDLE(resize(ham), "Failed to construct operator node object.");
    }

    kronecker_product_operator_node_data(const kronecker_product_operator_node_data& o) = default;
    kronecker_product_operator_node_data(kronecker_product_operator_node_data&& o) = default;

    kronecker_product_operator_node_data& operator=(const kronecker_product_operator_node_data& o) = default;    
    kronecker_product_operator_node_data& operator=(kronecker_product_operator_node_data&& o) = default;    

    void resize(const ham_node_type& ham)
    {
        try
        {
            if(!ham.is_leaf())
            {
                size_type nmodes = ham.size();
                m_r_nu.resize(nmodes);
                m_indices_nu.resize(nmodes);

                m_rindices.resize(ham().nterms());  std::fill(m_rindices.begin(), m_rindices.end(), ham().nterms());
                m_rindices_nu.resize(nmodes);
                for(size_type i = 0; i < nmodes; ++i){m_rindices_nu[i].resize(ham().nterms());    std::fill(m_rindices_nu[i].begin(), m_rindices_nu[i].end(), ham().nterms());}
    
                std::vector<std::list<std::pair<size_type, size_type>>> nrs(ham().nterms());
                std::vector<std::list<std::pair<size_type, size_type>>> nrs_nu(ham().nterms());
                size_type nelems = 0;
                size_type nelems_nu = 0;
                for(size_type i = 0; i < nmodes; ++i)
                {
                    for(size_type ri=0; ri < ham[i]().nni(); ++ri)
                    {
                        if(nrs[ham[i]().rindex(ri)].size() == 0){++nelems;}
                        nrs[ham[i]().rindex(ri)].push_back(std::make_pair(i, ri));
                        m_rindices[ham[i]().rindex(ri)] = nrs.size()-1;
                    }
                
                    nrs_nu.clear();
                    nelems_nu = 0;
                    for(size_type j=0; j < nmodes; ++j)
                    {
                        if(i != j)
                        {
                            for(size_type ri=0; ri < ham[j]().nni(); ++ri)
                            {
                                if(nrs_nu[ham[j]().rindex(ri)].size() == 0){++nelems_nu;}
                                nrs_nu[ham[j]().rindex(ri)].push_back(std::make_pair(j, ri));
                                m_rindices_nu[ham[j]().rindex(ri)] = nrs_nu[j].size()-1;
                            }
                        }
                    }
                    m_r_nu[i].resize(nelems_nu);
                    m_indices_nu[i].resize(nelems_nu);

                    size_type count = 0;
                    for(size_type j=0; j < nrs_nu.size(); ++j)
                    {
                        if(nrs_nu[j].size() != 0)
                        {
                            m_r_nu[i][count] = j;   
                            m_indices_nu[i][count].resize(nrs_nu[j].size()); 
                            nrs_nu[j].sort[](const std::pair<size_type, size_type>& i, const std::pair<size_type, size_type>& j){return std::get<0>(i) < std::get<0>(j);}();
                            for(size_type ind  = 0; ind < nrs_nu[j].size(); ++ind)
                            {
                                m_indices[nu][i][count][ind] = nrs_nu[i]
                            }
                            std::copy(nrs_nu[j].begin(), nrs_nu[j].end(), m_indices_nu[i][count].begin());
                            ++count;
                        }
                    }
                }
                indices.resize(nelems);
                m_r.resize(nelems);

                size_type count = 0;
                for(size_type i=0; i < nrs.size(); ++i)
                {
                    if(nrs[i].size() != 0)
                    {
                        m_r[count] = i;   
                        m_indices[count].resize(nrs[i].size()); 
                        nrs[i].sort[](const std::pair<size_type, size_type>& i, const std::pair<size_type, size_type>& j){return std::get<0>(i) < std::get<0>(j);}();
                        std::copy(nrs[i].begin(), nrs[i].end(), m_indices[count].begin());
                        ++count;
                    }
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize the operator node type.");
        }
    }

    void clear()
    {
        try
        {
            m_indices.clear();
            m_indices_nu.clear();
            m_r.clear();
            m_r_nu.clear();
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear operator node type.");
        }
    }

    void operator()(const ham_node_type& ham, size_type r, const hdata& A, mat& temp, mat& res) const
    {
        ASSERT(m_rindices[r] < ham().nterms(), "No operators bound for this r value.");

        size_type i = m_rindices[r];

        bool first_call = true;
        for(size_type j=0; j < m_indices[i].size(); ++j)
        {
            //get the indices required for applying the hamiltonina object.  
            size_type nu = std::get<0>(m_indices[i][j]);    size_type ri = std::get<1>(m_indices[i][j]);

            //now we create the rank 3 representations of the hdata object
            auto _A = A.as_rank_3(nu);
            auto _res = res.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
            auto _temp = temp.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));

            if(first_call)
            {     
                CALL_AND_HANDLE(_res  = contract(op[nu]().hspf(ri), 1, _A, 1), "Failed to compute kronecker product contraction.");      
                res_set = true; first_call = false;
            }
            else if(res_set)
            {   
                CALL_AND_HANDLE(_temp = contract(op[nu]().hspf(ri), 1, _res, 1), "Failed to compute kronecker product contraction.");    
                res_set = false;
            }
            else
            {               
                CALL_AND_HANDLE(_res  = contract(op[nu]().hspf(ri), 1, _temp, 1), "Failed to compute kronecker product contraction.");   
                res_set = true;
            }
        }
    }
    
    void apply_accumulated(const ham_node_type& ham, const  hdata& A, mat& temp, mat& res) const
    {
        bool first_call = true;

    }

    void operator()(const ham_node_type& ham, size_type nuskip, const hdata& A, mat& temp, mat& res) const
    {

    }
};

}   //namespace ttns

#include "node_traits/kronecker_product_operator_node_traits.hpp"

#endif  //TTNS_KRONECKER_PRODUCT_OPERATOR_NODE_HPP//

