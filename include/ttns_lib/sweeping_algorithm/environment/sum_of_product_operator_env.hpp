#ifndef TTNS_LIB_SWEEPING_ALGORITHM_SUM_OF_PRODUCT_OPERATOR_ENVIRONMENT_HPP
#define TTNS_LIB_SWEEPING_ALGORITHM_SUM_OF_PRODUCT_OPERATOR_ENVIRONMENT_HPP


#include "../../ttn/ttn.hpp"
#include "../../operators/sop_operator.hpp"

#include "../../ttn/ms_ttn.hpp"
#include "../../operators/multiset_sop_operator.hpp"

#include "../../core/matrix_element_buffer.hpp"
#include "../../core/single_particle_operator.hpp"
#include "../../core/mean_field_operator.hpp"

#include "sop_environment_traits.hpp"

namespace ttns
{

//TODO: Ensure that the matrices are correctly resized for each evaluation.
template <typename T, typename backend = linalg::blas_backend, template <typename, typename> class ttn_class = ttn>
class sop_environment
{
public:
    using ttn_type = ttn_class<T, backend>;
    using container_type = typename sop_environment_traits<ttn_type>::container_type;
    using environment_type = typename sop_environment_traits<ttn_type>::environment_type;

    using bond_action = typename sop_environment_traits<ttn_type>::bond_action;
    using site_action_leaf = typename sop_environment_traits<ttn_type>::site_action_leaf;
    using site_action_branch = typename sop_environment_traits<ttn_type>::site_action_branch;

    using spo_core = single_particle_operator_engine<T, backend>;
    using mfo_core = mean_field_operator_engine<T, backend>;

    using node_type = typename container_type::node_type;
    using data_type = typename container_type::value_type;

    using sop_node_type = typename environment_type::node_type;
    
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;

    using hnode = typename ttn_type::node_type;
    using hdata = typename hnode::value_type;

    struct parameter_list{};

    using buffer_type = matrix_element_buffer<T, backend>;

public:
    sop_environment(){}
    sop_environment(const sop_environment& o) = default;
    sop_environment(sop_environment&& o) = default;

    sop_environment& operator=(const sop_environment& o) = default;
    sop_environment& operator=(sop_environment&& o) = default;

    void initialise(const ttn_type& A, const environment_type& sop, container_type& ham)
    {
        using common::zip;   using common::rzip;
        try
        {
            ASSERT(has_same_structure(A, sop.contraction_info()), "The input state and sop operator do not have the same topology.");
            size_type maxsize = matrix_element_buffer<T, backend>::get_maximum_size(A);
            size_type maxcapacity = matrix_element_buffer<T, backend>::get_maximum_capacity(A);

            size_type maxmfo = maximum_dimension(A);
            if(maxmfo > maxcapacity){ maxcapacity = maxmfo;}

            //resize the working arrays.  We will resize these to the maximum possible array sizes
            try
            {
                m_buf.reallocate(maxcapacity, m_num_buffers);
                m_buf.resize(1, maxsize);
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to reallocate internal buffers required for the sop environment");
            }

            CALL_AND_HANDLE(ham.construct_topology(A), "Failed to construct the topology of the matrix element buffer tree.");

            for(auto z : zip(ham, A, sop.contraction_info()))
            {
                auto& mel = std::get<0>(z); const auto& a = std::get<1>(z);   const auto& h = std::get<2>(z);
                mel().initialise(h(), a);
            }
            ham.root()().initialise_root();
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise the sop environment");
        }
    }

    inline void initialise(const ttn_type& A, const environment_type& h, container_type& ham, const parameter_list&){CALL_AND_RETHROW(initialise(A, h, ham));}
    inline void initialise(const ttn_type& A, const environment_type& h, container_type& ham, parameter_list&&){CALL_AND_RETHROW(initialise(A, h, ham));}

    const size_type& num_buffers() const{return m_num_buffers;}
    size_type& num_buffers(){return m_num_buffers;}

    void clear()
    {
        CALL_AND_HANDLE(m_buf.clear(), "Failed to clear sop object.");
    }

    inline void update_env_down(const environment_type& op, const hnode& _a1, node_type& _h)
    {
        CALL_AND_HANDLE(mfo_core::evaluate(op.contraction_info()[_h.id()], _a1, m_buf.HA, m_buf.temp, _h), "Failed to evaluate the mean field operator.");
    }

    inline void update_env_up(const environment_type& op, const hnode& _a1, node_type& _h, bool force_update = true)
    {
        CALL_AND_HANDLE(spo_core::evaluate(op, op.contraction_info()[_h.id()], _a1, _h, m_buf.HA, m_buf.temp, false, force_update), "Failed to evaluate the single particle operator.");
    }

    const buffer_type& buffer() const{return m_buf;}
    buffer_type& buffer(){return m_buf;}

public:
#ifdef PYTTN_BUILD_CUDA
    static size_type maximum_dimension(const ttn_type& A)
    {
        size_type maxmfo = 0;
        //we need to fix this code, it currently doesn't work
        if(std::is_same<backend, linalg::cuda_backend>::value)
        {
            for(const auto& a : A)
            {
                size_type size = mfo_core::contraction_buffer_size(a, true);   
                if(size > maxmfo){maxmfo = size;}
            }
        }
        return maxmfo;
    }
#else
    static size_type maximum_dimension(const ttn_type& /*A*/){return 0;}
#endif

public:
    size_type m_num_buffers = 1;
    buffer_type m_buf;
    bond_action fha;
    site_action_leaf cel;
    site_action_branch ceb;
};

}

#endif

