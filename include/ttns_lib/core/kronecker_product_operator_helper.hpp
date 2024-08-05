#ifndef TTNS_LIB_KRONECKER_PRODUCT_HELPER_HPP
#define TTNS_LIB_KRONECKER_PRODUCT_HELPER_HPP

#include <linalg/linalg.hpp>
#include "../ttn/ttn.hpp"
#include "../ttn/ms_ttn.hpp"
#include "../operators/sop_operator.hpp"
#include "../operators/multiset_sop_operator.hpp"
#include "observable_node.hpp"
#include "multiset_sop_env_node.hpp"

namespace ttns
{

template <typename T, typename backend>
class kronecker_product_operator_mel
{
private:

    using hdata = ttn_node_data<T, backend>;
    using ms_hdata = multiset_node_data<T, backend>;

    using mat = linalg::matrix<T, backend>;
    using cinftype = sttn_node_data<T>;

    using ob_type = observable_node_data<T, backend>;
    using observable_node = typename tree<ob_type>::node_type;

    using boolnode = typename tree<bool>::node_type;
    using size_type = typename backend::size_type;

public:
    //kronecker product operators for matrix types
    static void apply(const observable_node& op, size_type r, const hdata& A, mat& temp, mat& res)
    {
        try
        {
            bool first_call = true;
            bool res_set = true;

            for(size_type nu=0; nu<op.size(); ++nu)
            {
                auto _A = A.as_rank_3(nu);
                auto _res = res.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                auto _temp = temp.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                if(first_call)
                {     
                    CALL_AND_HANDLE(_res  = contract(op[nu]()[r], 1, _A, 1), "Failed to compute kronecker product contraction.");      
                    res_set = true; first_call = false;
                }
                else if(res_set)
                {   
                    CALL_AND_HANDLE(_temp = contract(op[nu]()[r], 1, _res, 1), "Failed to compute kronecker product contraction.");    
                    res_set = false;
                }
                else
                {               
                    CALL_AND_HANDLE(_res  = contract(op[nu]()[r], 1, _temp, 1), "Failed to compute kronecker product contraction.");   
                    res_set = true;
                }
            }
            if(first_call){res_set = true;  res = A.as_matrix();}
            if(!res_set){res.swap_buffer(temp);}
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("applying kronecker product operator.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply kronecker product operator.");
        }
    }
    
    static void apply(const observable_node& op, size_type r, const boolnode& is_id, const hdata& A, mat& temp, mat& res)
    {
        try
        {
            bool first_call = true;
            bool res_set = true;

            for(size_type nu=0; nu<op.size(); ++nu)
            {
                if(!is_id[nu]())
                {
                    auto _A = A.as_rank_3(nu);
                    auto _res = res.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                    auto _temp = temp.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                    if(first_call)
                    {     
                        CALL_AND_HANDLE(_res  = contract(op[nu]()[r], 1, _A, 1), "Failed to compute kronecker product contraction.");      
                        res_set = true; first_call = false;
                    }
                    else if(res_set)
                    {   
                        CALL_AND_HANDLE(_temp = contract(op[nu]()[r], 1, _res, 1), "Failed to compute kronecker product contraction.");    
                        res_set = false;
                    }
                    else
                    {               
                        CALL_AND_HANDLE(_res  = contract(op[nu]()[r], 1, _temp, 1), "Failed to compute kronecker product contraction.");   
                        res_set = true;
                    }
                }
            }
            if(first_call){res_set = true;  res = A.as_matrix();}
            if(!res_set){res.swap_buffer(temp);}
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("applying kronecker product operator.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply kronecker product operator.");
        }
    }

    //kronecker product operators for matrix types -  currently unsure if this works need to check it in the future.
    static void apply_rectangular(const observable_node& op, size_type r, const hdata& A, mat& temp, mat& res)
    {
        try
        {
            bool first_call = true;
            bool res_set = true;

            size_type Mshape;
            size_type Rshape;

            for(size_type nu=0; nu<op.size(); ++nu)
            {
                size_type m = op[nu]()[r].shape(0);
                auto _A = A.as_rank_3(nu);
                if(first_call)
                {     
                    Mshape = _A.shape(0);
                    Rshape = m;

                    CALL_AND_HANDLE(res.resize(1, Mshape*Rshape*_A.shape(2)), "Failed to resize temporary working buffer.");
                    auto _res = res.reinterpret_shape(Mshape, Rshape, _A.shape(2));


                    CALL_AND_HANDLE(_res  = contract(op[nu]()[r], 1, _A, 1), "Failed to compute kronecker product contraction.");      
                    res_set = true; first_call = false;
                }
                else 
                {
                    Mshape *= Rshape;
                    Rshape = m;
                    if(res_set)
                    {   
                        CALL_AND_HANDLE(temp.resize(1, Mshape*Rshape*_A.shape(2)), "Failed to resize temporary working buffer.");
                        auto _res = res.reinterpret_shape(Mshape, _A.shape(1), _A.shape(2));
                        auto _temp = temp.reinterpret_shape(Mshape, Rshape, _A.shape(2));


                        CALL_AND_HANDLE(_temp = contract(op[nu]()[r], 1, _res, 1), "Failed to compute kronecker product contraction.");    
                        res_set = false;
                    }
                    else
                    {               
                        CALL_AND_HANDLE(res.resize(1, Mshape*Rshape*_A.shape(2)), "Failed to resize temporary working buffer.");
                        auto _res = res.reinterpret_shape(Mshape, Rshape, _A.shape(2));
                        auto _temp = temp.reinterpret_shape(Mshape, _A.shape(1), _A.shape(2));
                        
                        CALL_AND_HANDLE(_res  = contract(op[nu]()[r], 1, _temp, 1), "Failed to compute kronecker product contraction.");   
                        res_set = true;
                    }
                }
            }
            if(first_call){res_set = true;  res = A.as_matrix();}
            if(!res_set){res = temp;}
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("applying kronecker product operator.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply kronecker product operator.");
        }
    }

public:
    template <typename OP, typename A1, typename R1, typename t1>
    static void kron_prod_internal(const OP& op, A1& _A, R1& _res, t1& _temp, bool& first_call, bool& res_set)
    {
        if(first_call)
        {     
            CALL_AND_HANDLE(_res  = contract(op, 1, _A, 1), "Failed to compute kronecker product contraction.");      
            res_set = true; first_call = false;
        }
        else if(res_set)
        {   
            CALL_AND_HANDLE(_temp = contract(op, 1, _res, 1), "Failed to compute kronecker product contraction.");    
            res_set = false;
        }
        else
        {               
            CALL_AND_HANDLE(_res  = contract(op, 1, _temp, 1), "Failed to compute kronecker product contraction.");   
            res_set = true;
        }
    }


    template <typename A1, typename R1, typename t1>
    static void finalise(A1& A, R1& res, t1& temp, bool first_call, bool res_set, bool allow_swap=true)
    {
        if(first_call){res_set = true;  res = A;}
        if(!res_set)
        {
            if(allow_swap)
            {
                res.swap_buffer(temp);
            }
            else
            {
                res = temp;
            }
        }
    }

public:
    //kronecker product operators for the operator ype
    template <typename spfnode, typename spindtype>
    static void kron_prod(spfnode&& op, const spindtype& spinds, const hdata& A, mat& temp, mat& res)
    {
        try
        {
            temp.resize(A.shape(0), A.shape(1));
            res.resize(A.shape(0), A.shape(1));
            bool first_call = true;
            bool res_set = true;

            for(size_type ni=0; ni<spinds.size(); ++ni)
            {
                size_type nu = spinds[ni][0];
                size_type cri = spinds[ni][1];

                auto _A = A.as_rank_3(nu);
                auto _res = res.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                auto _temp = temp.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
        
                kron_prod_internal(op(nu, cri), _A, _res, _temp, first_call, res_set);
            }
            finalise(A.as_matrix(), res, temp, first_call, res_set);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply kronecker product operator.");
        }
    }

public:
    template <typename OP, typename A1, typename R1, typename t1>
    static void kron_prod_rect_internal(const OP& op, A1& _A, R1& res, t1& temp, size_type Mshape, size_type Rshape, bool& first_call, bool& res_set)
    {
        if(first_call)
        {     
            CALL_AND_HANDLE(res.resize(Mshape*Rshape, _A.shape(2)), "Failed to resize temporary working buffer.");
            auto _res = res.reinterpret_shape(Mshape, Rshape, _A.shape(2));

            CALL_AND_HANDLE(_res  = contract(op, 1, _A, 1), "Failed to compute kronecker product contraction.");      
            res_set = true; first_call = false;
        }
        else
        {
            if(res_set)
            {   
                CALL_AND_HANDLE(temp.resize(Mshape*Rshape, _A.shape(2)), "Failed to resize temporary working buffer.");
                auto _res = res.reinterpret_shape(Mshape, _A.shape(1), _A.shape(2));
                auto _temp = temp.reinterpret_shape(Mshape, Rshape, _A.shape(2));

                CALL_AND_HANDLE(_temp = contract(op, 1, _res, 1), "Failed to compute kronecker product contraction.");    
                res_set = false;
            }
            else
            {               
                CALL_AND_HANDLE(res.resize(Mshape*Rshape, _A.shape(2)), "Failed to resize temporary working buffer.");
                auto _res = res.reinterpret_shape(Mshape, Rshape, _A.shape(2));
                auto _temp = temp.reinterpret_shape(Mshape, _A.shape(1), _A.shape(2));
                
                CALL_AND_HANDLE(_res  = contract(op, 1, _temp, 1), "Failed to compute kronecker product contraction.");   
                res_set = true;
            }
        }
    }

    template <typename A1, typename R1, typename t1>
    static void kron_prod_rect_skip(A1& _A, R1& res, t1& temp, size_type Mshape, size_type Rshape, bool& first_call, bool& res_set)
    {
        if(first_call)
        {     
            CALL_AND_HANDLE(res.resize(Mshape*Rshape, _A.shape(2)), "Failed to resize temporary working buffer.");
            auto _res = res.reinterpret_shape(Mshape, Rshape, _A.shape(2));
            CALL_AND_HANDLE(_res  = _A, "Failed to compute kronecker product contraction.");      
            res_set = true; first_call = false;
        }
        else
        {
            if(res_set)
            {   
                CALL_AND_HANDLE(temp.resize(Mshape*Rshape, _A.shape(2)), "Failed to resize temporary working buffer.");
                auto _res = res.reinterpret_shape(Mshape, _A.shape(1), _A.shape(2));
                auto _temp = temp.reinterpret_shape(Mshape, Rshape, _A.shape(2));

                CALL_AND_HANDLE(_temp = _res, "Failed to compute kronecker product contraction.");    
                res_set = false;
            }
            else
            {               
                CALL_AND_HANDLE(res.resize(Mshape*Rshape, _A.shape(2)), "Failed to resize temporary working buffer.");
                auto _res = res.reinterpret_shape(Mshape, Rshape, _A.shape(2));
                auto _temp = temp.reinterpret_shape(Mshape, _A.shape(1), _A.shape(2));
                
                CALL_AND_HANDLE(_res  = _temp, "Failed to compute kronecker product contraction.");   
                res_set = true;
            }
        }
    }

public:
    template <typename spfnode>
    static void kpo_id(const spfnode& op, const hdata& A, mat& temp, mat& res)
    {
        try
        {
            ASSERT(op().has_identity(), "Cannot apply rectangular hamiltonian without having identity matrices bound");
            bool first_call = true;
            bool res_set = true;

            size_type Mshape = 0;
            size_type Rshape = 0;

            for(size_type nu=0; nu<op.size(); ++nu)
            {
                size_type m = op[nu]().spf_id().shape(0);
                auto _A = A.as_rank_3(nu);

                if(first_call){Mshape = _A.shape(0); Rshape = m;}
                else{Mshape *= Rshape;   Rshape = m;}

                kron_prod_rect_internal(op[nu]().spf_id(), _A, res, temp, Mshape, Rshape, first_call, res_set);
            }
            finalise(A.as_matrix(), res, temp, first_call, res_set, false);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply kronecker product operator.");
        }
    }
    template <typename spfnode>
    static void kpo_id(const spfnode& op, const hdata& A, size_t nuskip, mat& temp, mat& res)
    {
        try
        {
            ASSERT(op().has_identity(), "Cannot apply rectangular hamiltonian without having identity matrices bound");
            bool first_call = true;
            bool res_set = true;

            size_type Mshape = 0;
            size_type Rshape = 0;

            for(size_type nu=0; nu<op.size(); ++nu)
            {
                auto _A = A.as_rank_3(nu);
                if(nu != nuskip)
                {
                    size_type m = op[nu]().spf_id().shape(0);

                    if(first_call){Mshape = _A.shape(0); Rshape = m;}
                    else{Mshape *= Rshape;   Rshape = m;}

                    kron_prod_rect_internal(op[nu]().spf_id(), _A, res, temp, Mshape, Rshape, first_call, res_set);
                }
                else
                {
                    if(first_call){Mshape = _A.shape(0); Rshape = _A.shape(1);}
                    else{Mshape *= Rshape;   Rshape = _A.shape(1);}
                    kron_prod_rect_skip(_A, res, temp, Mshape, Rshape, first_call, res_set);
                }
            }
            finalise(A.as_matrix(), res, temp, first_call, res_set, false);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply kronecker product operator.");
        }
    }


    //kronecker product operators for the operator type
    template <typename spfnode, typename spfidnode, typename spindtype>
    static void kron_prod(spfnode&& op, spfidnode&& opid, const spindtype& spinds, const hdata& B, const hdata& A, mat& temp, mat& res)
    {
        try
        {
            bool first_call = true;
            bool res_set = true;

            size_type Mshape = 0;
            size_type Rshape = 0;

            size_t ni = 0;
            for(size_type _nu=0; _nu<A.nmodes(); ++_nu)
            {
                size_type nu = spinds[ni][0];
                size_type cri = spinds[ni][1];
                size_type m = B.dim(_nu);

                auto _A = A.as_rank_3(_nu);

                if(first_call){Mshape = _A.shape(0); Rshape = m;}
                else{Mshape *= Rshape;   Rshape = m;}

                if(_nu == nu)
                {
                    kron_prod_rect_internal(op(nu, cri), _A, res, temp, Mshape, Rshape, first_call, res_set);
                    if(ni+1<spinds.size()){++ni;}
                }
                else
                {
                    kron_prod_rect_internal(opid(_nu), _A, res, temp, Mshape, Rshape, first_call, res_set);
                }
            }
            finalise(A.as_matrix(), res, temp, first_call, res_set, false);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply kronecker product operator.");
        }
    }


    //kronecker product operators for the operator type
    template <typename spfnode, typename spfidnode, typename spindtype>
    static void kron_prod(spfnode&& op, spfidnode&& opid, const spindtype& spinds, const hdata& B, const hdata& A, size_type nuskip, mat& temp, mat& res)
    {
        try
        {
            bool first_call = true;
            bool res_set = true;

            size_type Mshape = 0;
            size_type Rshape = 0;

            size_t ni = 0;
            for(size_type _nu=0; _nu<A.nmodes(); ++_nu)
            {
                size_type nu = spinds[ni][0];
                size_type cri = spinds[ni][1];
                size_type m = B.dim(_nu);

                auto _A = A.as_rank_3(_nu);

                if(_nu != nuskip)
                {
                    if(first_call){Mshape = _A.shape(0); Rshape = m;}
                    else{Mshape *= Rshape;   Rshape = m;}

                    if(_nu == nu)
                    {
                        kron_prod_rect_internal(op(nu, cri), _A, res, temp, Mshape, Rshape, first_call, res_set);
                        if(ni+1<spinds.size()){++ni;}
                    }
                    else
                    {
                        kron_prod_rect_internal(opid(_nu), _A, res, temp, Mshape, Rshape, first_call, res_set);
                    }
                }
                else
                {
                    if(first_call){Mshape = _A.shape(0); Rshape = _A.shape(1);}
                    else{Mshape *= Rshape;   Rshape = _A.shape(1);}
                    kron_prod_rect_skip(_A, res, temp, Mshape, Rshape, first_call, res_set);
                }
            }
            finalise(A.as_matrix(), res, temp, first_call, res_set, false);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply kronecker product operator.");
        }
    }
};  //struct kronecker_product_operator

}   //namespace ttns


#endif  //TTNS_KRONECKER_PRODUCT_HELPER_HPP//

