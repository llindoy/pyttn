#ifndef TTNS_LIB_SWEEPING_ALGORITHM_SUM_OF_PRODUCT_OPERATOR_ENVIRONMENT_TRAITS_HPP
#define TTNS_LIB_SWEEPING_ALGORITHM_SUM_OF_PRODUCT_OPERATOR_ENVIRONMENT_TRAITS_HPP


#include "../../ttn/ttn.hpp"
#include "../../operators/sop_operator.hpp"

#include "../../ttn/ms_ttn.hpp"
#include "../../operators/multiset_sop_operator.hpp"

#include "../../core/matrix_element_buffer.hpp"
#include "../../core/single_particle_operator.hpp"
#include "../../core/sop_env_node.hpp"
#include "../../core/multiset_sop_env_node.hpp"

namespace ttns
{

template <typename ttn_type>
class sop_environment_traits;

class bond_action_helper
{
protected:
    template <typename vtype, typename soptype, typename cinftype, typename mat_type, typename rtype>
    static inline void evaluate_id(const vtype& v, const soptype& h, const cinftype& cinf, mat_type& t1, rtype& res)
    {
        for(size_t ind=0; ind < cinf.nterms(); ++ind)
        {
            if(cinf[ind].is_identity_spf())
            {
                if(cinf[ind].is_identity_mf())
                {
                    CALL_AND_HANDLE(res += cinf[ind].coeff()*v, "Failed to apply identity contribution.");
                }
                else
                {
                    CALL_AND_HANDLE(res += cinf[ind].coeff()*v*trans(h.mf(ind)), "Failed to apply the mean field contribution matrix.");
                }
            }
            else
            {

                if(cinf[ind].is_identity_mf())
                {
                    CALL_AND_HANDLE(res += cinf[ind].coeff()*h.spf(ind)*v, "Failed to apply the single particle contribution.");
                }
                else
                {
                    CALL_AND_HANDLE(t1 = h.spf(ind)*v, "Failed to apply the single particle contribution.");
                    CALL_AND_HANDLE(res += cinf[ind].coeff()*t1*trans(h.mf(ind)), "Failed to apply the mean field contribution.");
                }
            } 
        }
    }
    template <typename vtype, typename soptype, typename cinftype, typename mat_type, typename rtype>
    static inline void evaluate_olap(const vtype& v, const soptype& h, const cinftype& cinf, mat_type& t1, rtype& res)
    {
        for(size_t ind=0; ind < cinf.nterms(); ++ind)
        {
            if(cinf[ind].is_identity_spf())
            {
                CALL_AND_HANDLE(t1 = h.spf_id()*v, "Failed to apply the single particle contribution.");
                if(cinf[ind].is_identity_mf())
                {
                    CALL_AND_HANDLE(res += cinf[ind].coeff()*t1*trans(h.mf_id()), "Failed to apply the mean field contribution.");
                }
                else
                {
                    CALL_AND_HANDLE(res += cinf[ind].coeff()*t1*trans(h.mf(ind)), "Failed to apply the mean field contribution matrix.");
                }
            }
            else
            {
                CALL_AND_HANDLE(t1 = h.spf(ind)*v, "Failed to apply the single particle contribution.");
                if(cinf[ind].is_identity_mf())
                {
                    CALL_AND_HANDLE(res += cinf[ind].coeff()*t1*trans(h.mf_id()), "Failed to apply the mean field contribution.");
                }
                else
                {
                    CALL_AND_HANDLE(res += cinf[ind].coeff()*t1*trans(h.mf(ind)), "Failed to apply the mean field contribution.");
                }
            } 
        }
    }
public:
    template <typename vtype, typename soptype, typename cinftype, typename mat_type, typename rtype>
    static inline void evaluate(const vtype& v, const soptype& h, const cinftype& cinf, mat_type& t1, rtype& res, bool use_identity = true )
    {
        size_t n1 = t1.shape(0);  size_t n2 = t1.shape(1);

        if(use_identity)
        {
            CALL_AND_RETHROW(evaluate_id(v, h, cinf, t1, res));
        }
        else
        {
            CALL_AND_RETHROW(evaluate_olap(v, h, cinf, t1, res));
        }
        CALL_AND_RETHROW(t1.resize(n1, n2));
    }
};

class site_action_leaf_helper
{
protected:
    template <typename vtype, typename soptype, typename env_type, typename cinftype, typename mat_type, typename rtype>
    static inline void evaluate_id(const vtype& v, const soptype& h, const cinftype& cinf, const env_type& hprim, mat_type& t1, mat_type& t2, rtype& res)
    {   
        try
        {
            using T = typename mat_type::value_type;
            for(size_t ind=0; ind < cinf.nterms(); ++ind)
            {
                if(cinf[ind].is_identity_spf())
                {
                    T coeff(0);

                    for(size_t i = 0; i < cinf[ind].spf_coeff().size(); ++i){coeff += cinf[ind].spf_coeff(i);}
                    coeff *= cinf[ind].coeff();
                    if(cinf[ind].is_identity_mf())
                    {
                        CALL_AND_HANDLE(res += coeff*v, "Failed to apply identity contribution.");
                    }
                    else
                    {
                        CALL_AND_HANDLE(res += coeff*v*trans(h.mf(ind)), "Failed to apply the mean field contribution matrix.");
                    }
                }
                else
                {
                    if(cinf[ind].is_identity_mf())
                    {
                        for(size_t i = 0; i < cinf[ind].nspf_terms(); ++i)
                        {
                            T coeff =  cinf[ind].spf_coeff(i) * cinf[ind].coeff();
                            auto& indices = cinf[ind].spf_indexing()[i][0];
                            CALL_AND_HANDLE(hprim[indices[0]][indices[1]].apply(v, t2), "Failed to apply leaf operator.");
                            res += coeff*t2;
                        }
                    }
                    else
                    {
                        {
                            T coeff = cinf[ind].spf_coeff(0) * cinf[ind].coeff();
                            auto& indices = cinf[ind].spf_indexing()[0][0];
                            CALL_AND_HANDLE(hprim[indices[0]][indices[1]].apply(v, t1), "Failed to apply leaf operator.");
                            t2 = coeff*t1;
                        }
                        for(size_t i = 1; i < cinf[ind].nspf_terms(); ++i)
                        {
                            T coeff = cinf[ind].spf_coeff(i)* cinf[ind].coeff();
                            auto& indices = cinf[ind].spf_indexing()[i][0];
                            CALL_AND_HANDLE(hprim[indices[0]][indices[1]].apply(v, t1), "Failed to apply leaf operator.");
                            t2 += coeff*t1;
                        }

                       CALL_AND_HANDLE(res += t2*trans(h.mf(ind)), "Failed to apply the mean field contribution.");
                    }
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply the leaf coefficient evolution operator at a node.");
        }
    }
    template <typename vtype, typename soptype, typename env_type, typename cinftype, typename mat_type, typename rtype>
    static inline void evaluate_olap(const vtype& v, const soptype& h, const cinftype& cinf, const env_type& hprim, mat_type& t1, mat_type& t2, rtype& res)
    {   
        try
        {
            using T = typename mat_type::value_type;
            for(size_t ind=0; ind < cinf.nterms(); ++ind)
            {
                if(cinf[ind].is_identity_spf())
                {
                    T coeff(0);

                    for(size_t i = 0; i < cinf[ind].spf_coeff().size(); ++i){coeff += cinf[ind].spf_coeff(i);}
                    coeff *= cinf[ind].coeff();
                    if(cinf[ind].is_identity_mf())
                    {
                        CALL_AND_HANDLE(res += coeff*v*trans(h.mf_id()), "Failed to apply identity contribution.");
                    }
                    else
                    {
                        CALL_AND_HANDLE(res += coeff*v*trans(h.mf(ind)), "Failed to apply the mean field contribution matrix.");
                    }
                }
                else
                {                        {
                    T coeff = cinf[ind].spf_coeff(0) * cinf[ind].coeff();
                    auto& indices = cinf[ind].spf_indexing()[0][0];
                    CALL_AND_HANDLE(hprim[indices[0]][indices[1]].apply(v, t1), "Failed to apply leaf operator.");
                    t2 = coeff*t1;
                    }
                    for(size_t i = 1; i < cinf[ind].nspf_terms(); ++i)
                    {
                        T coeff = cinf[ind].spf_coeff(i)* cinf[ind].coeff();
                        auto& indices = cinf[ind].spf_indexing()[i][0];
                        CALL_AND_HANDLE(hprim[indices[0]][indices[1]].apply(v, t1), "Failed to apply leaf operator.");
                        t2 += coeff*t1;
                    }
                    if(cinf[ind].is_identity_mf())
                    {
                       CALL_AND_HANDLE(res += t2*trans(h.mf_id()), "Failed to apply the mean field contribution.");
                    }
                    else
                    {
                       CALL_AND_HANDLE(res += t2*trans(h.mf(ind)), "Failed to apply the mean field contribution.");
                    }
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply the leaf coefficient evolution operator at a node.");
        }
    }

public:
    template <typename vtype, typename soptype, typename env_type, typename cinftype, typename mat_type, typename rtype>
    static inline void evaluate(const vtype& v, const soptype& h, const cinftype& cinf, const env_type& hprim, mat_type& t1, mat_type& t2, rtype& res, bool use_identity = true)
    {   
        size_t n1 = t1.shape(0);  size_t n2 = t1.shape(1);
        if(use_identity)
        {
            CALL_AND_RETHROW(evaluate_id(v, h, cinf, hprim, t1, t2, res));
        }
        else
        {
            CALL_AND_RETHROW(evaluate_olap(v, h, cinf, hprim, t1, t2, res));
        }
        CALL_AND_RETHROW(t1.resize(n1, n2));
        CALL_AND_RETHROW(t2.resize(n1, n2));
    }
};

class site_action_branch_helper
{
public:
    template <typename T, typename backend, typename soptype, typename cinftype, typename mat_type, typename rtype>
    static inline void evaluate(const ttn_node_data<T, backend>& v, const soptype& h, const cinftype& cinf, mat_type& t1, mat_type& t2, mat_type& t3, rtype& res)
    {   
        using spo_core = single_particle_operator_engine<T, backend>;

        size_t n1 = t1.shape(0);  size_t n2 = t1.shape(1);
        for(size_t ind=0; ind < cinf.nterms(); ++ind)
        {
            if(cinf[ind].is_identity_spf())
            {
                T coeff(0);
                for(size_t i = 0; i < cinf[ind].spf_coeff().size(); ++i){coeff += cinf[ind].spf_coeff(i);}
                coeff *= cinf[ind].coeff();

                if(cinf[ind].is_identity_mf())
                {
                    CALL_AND_HANDLE(res += coeff*v.as_matrix(), "Failed to apply identity contribution.");
                }
                else
                {
                    CALL_AND_HANDLE(res += coeff*v.as_matrix()*trans(h().mf(ind)), "Failed to apply the mean field contribution matrix.");
                }
            }
            else
            {
                if(cinf[ind].is_identity_mf())
                {
                    for(size_t i = 0; i < cinf[ind].nspf_terms(); ++i)
                    {
                        T coeff = cinf[ind].spf_coeff(i) * cinf[ind].coeff();
                        CALL_AND_HANDLE(spo_core::kron_prod(h, cinf, ind, i, v, t1, t2), "Failed to apply kronecker product operator.");
                        res += coeff*t2;
                    }
                }
                else
                {
                    T coeff = cinf[ind].spf_coeff(0) * cinf[ind].coeff();
                    CALL_AND_HANDLE(spo_core::kron_prod(h, cinf, ind, 0, v, t1, t2), "Failed to apply kronecker product operator.");
                    t3 = coeff*t2;
                    for(size_t i = 1; i < cinf[ind].nspf_terms(); ++i)
                    {
                        coeff = cinf[ind].spf_coeff(i)*cinf[ind].coeff();
                        CALL_AND_HANDLE(spo_core::kron_prod(h, cinf, ind, i, v, t1, t2), "Failed to apply kronecker product operator.");
                        t3 += coeff*t2;
                    }

                    CALL_AND_HANDLE(res += t3*trans(h().mf(ind)), "Failed to apply the mean field contribution.");
                }
            }
        }
        t1.resize(n1, n2);
        t2.resize(n1, n2);
        t3.resize(n1, n2);
    }
public:
    template <typename T, typename backend, typename soptype, typename cinftype, typename mat_type, typename rtype>
    static inline void evaluate(const ttn_node_data<T, backend>& v, const ttn_node_data<T, backend>& vb, const soptype& h, const cinftype& cinf, mat_type& t1, mat_type& t2, mat_type& t3, rtype& res)
    {   
        using kpo = kronecker_product_operator_mel<T, backend>;
        using spo_core = single_particle_operator_engine<T, backend>;

        size_t n1 = t1.shape(0);  size_t n2 = t1.shape(1);
        for(size_t ind=0; ind < cinf.nterms(); ++ind)
        {
            if(cinf[ind].is_identity_spf())
            {
                T coeff = cinf[ind].spf_coeff(0) * cinf[ind].coeff();
                CALL_AND_HANDLE(kpo::kpo_id(h, v, t1, t2), "Failed to apply kronecker product operator.");
                t3 = coeff*t2;
                if(cinf[ind].is_identity_mf())
                {
                    CALL_AND_HANDLE(res += t3*trans(h().mf_id()), "Failed to apply the mean field contribution.");
                }
                else
                {
                    CALL_AND_HANDLE(res += t3*trans(h().mf(ind)), "Failed to apply the mean field contribution.");
                }
            }
            else
            {   
                T coeff = cinf[ind].spf_coeff(0) * cinf[ind].coeff();
                CALL_AND_HANDLE(spo_core::kron_prod(h, cinf, ind, 0, vb, v, t1, t2), "Failed to apply kronecker product operator.");
                t3 = coeff*t2;
                for(size_t i = 1; i < cinf[ind].nspf_terms(); ++i)
                {
                    coeff = cinf[ind].spf_coeff(i)*cinf[ind].coeff();
                    CALL_AND_HANDLE(spo_core::kron_prod(h, cinf, ind, i, vb, v, t1, t2), "Failed to apply kronecker product operator.");
                    t3 += coeff*t2;
                }
                if(cinf[ind].is_identity_mf())
                {
                    CALL_AND_HANDLE(res += t3*trans(h().mf_id()), "Failed to apply the mean field contribution.");
                }
                else
                {
                    CALL_AND_HANDLE(res += t3*trans(h().mf(ind)), "Failed to apply the mean field contribution.");
                }
            }
        }
        t1.resize(n1, n2);
        t2.resize(n1, n2);
        t3.resize(n1, n2);
    }
};

template <typename T, typename backend>
class sop_environment_traits<ttn<T, backend>>
{
public:
    using container_type = tree<sop_env_node_data<T, backend>>;
    using environment_type = sop_operator<T, backend>;

    using hnode = typename ttn<T,backend>::node_type;
    using hdata = typename hnode::value_type;
    using bond_matrix_type = typename hnode::bond_matrix_type;

    using node_type = typename container_type::node_type;
    using size_type = typename backend::size_type;

    class bond_action
    {
    public:
        template <typename vtype, typename mat_type, typename rtype>
        inline void operator()(const vtype& v, const node_type& h, const environment_type& hprim, mat_type& t1, rtype& res) const
        {
            try
            {
                const auto& cinf = hprim.contraction_info()[h.id()]();
                CALL_AND_HANDLE(res = hprim.Eshift()*v, "Failed to apply shift contribution.");
                CALL_AND_RETHROW(bond_action_helper::evaluate(v, h(), cinf, t1, res));
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply the action of the full Hamiltonian at a node.");
            }
        }

        template <typename mat_type, typename rtype>
        inline void operator()(const bond_matrix_type& v, const node_type& h, const environment_type& hprim, mat_type& t1, rtype& res) const
        {
            try
            {
                const auto& cinf = hprim.contraction_info()[h.id()]();
                CALL_AND_HANDLE(res = hprim.Eshift()*v, "Failed to apply shift contribution.");
                CALL_AND_RETHROW(bond_action_helper::evaluate(v, h(), cinf, t1, res));
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply the action of the full Hamiltonian at a node.");
            }
        }
    };  //class bond_action

    class site_action_leaf
    {
    public:
        template <typename vtype, typename mat_type, typename rtype>
        inline void operator()(const vtype& v, const node_type& h, const environment_type& hprim, mat_type& t1, mat_type& t2, rtype& res) const
        {   
            try
            {
                const auto& cinf = hprim.contraction_info()[h.id()]();
                CALL_AND_HANDLE(res = hprim.Eshift()*v, "Failed to apply shift contribution");
                CALL_AND_RETHROW(site_action_leaf_helper::evaluate(v, h(), cinf, hprim.mode_operators(), t1, t2, res));
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply the leaf coefficient evolution operator at a node.");
            }
        }
        template <typename mat_type, typename rtype>
        inline void operator()(const hdata& v, const node_type& h, const environment_type& hprim, mat_type& t1, mat_type& t2, rtype& res) const
        {   
            try
            {
                const auto& cinf = hprim.contraction_info()[h.id()]();
                CALL_AND_HANDLE(res = hprim.Eshift()*v, "Failed to apply shift contribution");
                CALL_AND_RETHROW(site_action_leaf_helper::evaluate(v.as_matrix(), h(), cinf, hprim.mode_operators(), t1, t2, res));
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply the leaf coefficient evolution operator at a node.");
            }
        }
    };  //class site_action_leaf


    class site_action_branch
    {
    protected:
        mutable hdata* node_inf;

    public:
        void set_pointer(hdata* npointer) const{node_inf = npointer;}
        void unset_pointer() const{node_inf = nullptr;}

        site_action_branch() : node_inf(nullptr){}
        ~site_action_branch(){node_inf = nullptr;}

    public:
        template <typename vtype, typename mat_type, typename rtype>
        inline void operator()(const vtype& v, const node_type& h, const environment_type& hprim, mat_type& t1, mat_type& t2, mat_type& t3, rtype& res) const
        {   
            ASSERT(node_inf != nullptr, "Cannot apply site action branch without first binding a node_type object to this.");
            (*node_inf).as_matrix() = v;
            CALL_AND_RETHROW(this->operator()(*node_inf, h, hprim, t1, t2, t3, res));
        }

        template <typename mat_type, typename rtype>
        inline void operator()(const hdata& v, const node_type& h, const environment_type& hprim, mat_type& t1, mat_type& t2, mat_type& t3, rtype& res) const
        {   
            try
            {
                const auto& vmat = v.as_matrix();
                const auto& cinf = hprim.contraction_info()[h.id()]();
                CALL_AND_HANDLE(res = hprim.Eshift()*vmat, "Failed to apply shift contribution.");
                CALL_AND_RETHROW(site_action_branch_helper::evaluate(v, h, cinf, t1, t2, t3, res));
            }        
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply the branch coefficient evolution operator at a node.");
            }
        }
    };  //class site_action_branch
};

template <typename T, typename backend>
class sop_environment_traits<ms_ttn<T, backend>>
{
public:
    using environment_type = multiset_sop_operator<T, backend>;
    using container_type = tree<ms_sop_env_node_data<T, backend>>;

    using hnode = typename ttn<T,backend>::node_type;
    using hdata = typename hnode::value_type;

    using ms_hnode = typename ms_ttn<T,backend>::node_type;
    using ms_hdata = typename ms_hnode::value_type;

    using bond_matrix_type = typename ms_hnode::bond_matrix_type;

    using node_type = typename container_type::node_type;
    using size_type = typename backend::size_type;

    class bond_action
    {
    public:
        template <typename vtype, typename mat_type, typename rtype>
        inline void operator()(const vtype& v, const node_type& h, const environment_type& hprim, mat_type& t1, rtype& res) const
        {
        }

        template <typename mat_type, typename rtype>
        inline void operator()(const bond_matrix_type& v, const node_type& h, const environment_type& hprim, mat_type& t1, std::vector<rtype>& res) const
        {
            try
            { 
                const auto& cinf = hprim.contraction_info()[h.id()]();
                for(size_t row = 0; row < cinf.size(); ++row)
                {
                    res[row] *= 0.0;
                    for(size_t ci = 0; ci < cinf[row].size(); ++ci)
                    {
                        size_t col = cinf[row][ci].col();
                        CALL_AND_RETHROW(bond_action_helper::evaluate(v[col], h()[row][ci], cinf[row][ci], t1, res[row], row==col));
                    }
                }
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply the action of the full Hamiltonian at a node.");
            }
        }

    };  //class bond_action

    class site_action_leaf
    {
    public:
        template <typename vtype, typename mat_type, typename rtype>
        inline void operator()(const vtype& v, const node_type& h,  const environment_type& hprim, mat_type& t1, mat_type& t2, rtype& res) const
        {   
        }

        template <typename mat_type, typename rtype>
        inline void operator()(const ms_hdata& v, const node_type& h,  const environment_type& hprim, mat_type& t1, mat_type& t2, rtype& res) const
        {   
            try
            { 
                const auto& cinf = hprim.contraction_info()[h.id()]();
                for(size_t row = 0; row < cinf.size(); ++row)
                {
                    for(size_t ci = 0; ci < cinf[row].size(); ++ci)
                    {
                        size_t col = cinf[row][ci].col();
                        CALL_AND_RETHROW(site_action_leaf_helper::evaluate(v[col], h()[row][ci], cinf[row][ci], hprim.mode_operators(row, ci), t1, t2, res[row], row == col));
                    }
                }
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply the action of the full Hamiltonian at a node.");
            }
        }
    };  //class site_action_leaf

    class site_action_branch
    {
    public:
        template <typename vtype, typename mat_type, typename rtype>
        inline void operator()(const vtype& v, node_type& h,  const environment_type& hprim, mat_type& t1, mat_type& t2, mat_type& t3, rtype& res) const
        {   
        }

        template <typename mat_type, typename rtype>
        inline void operator()(const ms_hdata& v, node_type& h,  const environment_type& hprim, mat_type& t1, mat_type& t2, mat_type& t3, rtype& res) const
        {   
            try
            { 
                const auto& cinf = hprim.contraction_info()[h.id()]();
                for(size_t row = 0; row < cinf.size(); ++row)
                {
                    for(size_t ci = 0; ci < cinf[row].size(); ++ci)
                    {
                        size_t col = cinf[row][ci].col();

                        ms_sop_env_slice<T, backend> hslice(h, row, ci);
                        if(row == col)
                        {
                            CALL_AND_RETHROW(site_action_branch_helper::evaluate(v[col], hslice, cinf[row][ci], t1, t2, t3, res[row]));
                        }
                        else
                        {
                            CALL_AND_RETHROW(site_action_branch_helper::evaluate(v[col], v[row], hslice, cinf[row][ci], t1, t2, t3, res[row]));
                        }
                    }
                }
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply the action of the full Hamiltonian at a node.");
            }
        }

    };  //class site_action_branch
};


}

#endif

