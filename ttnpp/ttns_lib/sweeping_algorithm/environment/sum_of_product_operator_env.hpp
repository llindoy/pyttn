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

#ifndef PYTTN_TTNS_LIB_SWEEPING_ALGORITHM_ENVIRONMENT_SUM_OF_PRODUCT_OPERATOR_ENV_HPP_
#define PYTTN_TTNS_LIB_SWEEPING_ALGORITHM_ENVIRONMENT_SUM_OF_PRODUCT_OPERATOR_ENV_HPP_

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

    // TODO: Ensure that the matrices are correctly resized for each evaluation.
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

        using size_type = typename linalg::traits<backend>::size_type;
        using real_type = typename linalg::get_real_type<T>::type;

        using hnode = typename ttn_type::node_type;
        using hdata = typename hnode::value_type;
        static constexpr std::string_view class_info{"sopenv:"};

        struct parameter_list
        {
            parameter_list() : opsum_nthreads(1) {}
            parameter_list(size_t nop) : opsum_nthreads(nop) {}
            parameter_list(const parameter_list &o) = default;
            parameter_list(parameter_list &&o) = default;
            parameter_list &operator=(const parameter_list &o) = default;
            parameter_list &operator=(parameter_list &&o) = default;

            size_type opsum_nthreads;
        };

        using buffer_type = matrix_element_buffer<T, backend>;

    public:
        buffer_type m_buf;
        bond_action fha;
        site_action_leaf cel;
        site_action_branch ceb;
        size_type m_operator_sum_nthreads = 1;
        size_type m_set_var_nthreads = 1;
        size_type m_maxcapacity = 0;
        size_type m_maxsize = 0;

    public:
        sop_environment() {}
        sop_environment(const sop_environment &o) = default;
        sop_environment(sop_environment &&o) = default;

        sop_environment &operator=(const sop_environment &o) = default;
        sop_environment &operator=(sop_environment &&o) = default;

        inline void initialise(const ttn_type &A, const environment_type &h, container_type &ham, size_type set_var_nthreads = 1) 
        {
            parameter_list params; 
            CALL_AND_RETHROW(initialise(A, h, ham, params, set_var_nthreads)); 
        }

        void initialise(const ttn_type &A, const environment_type &sop, container_type &ham, const parameter_list & params, size_type set_var_nthreads = 1)
        {
#ifdef TRACE_LOG
            logging::trace("initialising sum of product operator environment.");
#endif
            using common::rzip;
            using common::zip;
            try
            {
                ASSERT(has_same_structure(A, sop.contraction_info()), "The input state and sop operator do not have the same topology.");
                size_type maxsize = matrix_element_buffer<T, backend>::get_maximum_size(A);
                size_type maxcapacity = matrix_element_buffer<T, backend>::get_maximum_capacity(A);

                size_type maxmfo = maximum_dimension(A);
                if (maxmfo > maxcapacity)
                {
                    maxcapacity = maxmfo;
                }
                size_type os_nt = params.opsum_nthreads < 1 ? 1 : params.opsum_nthreads;
                size_type sv_nt = set_var_nthreads < 1 ? 1 : set_var_nthreads;

                m_operator_sum_nthreads = os_nt;
                m_set_var_nthreads = sv_nt;

                m_maxcapacity = maxcapacity;
                m_maxsize = maxsize;
                // resize the working arrays.  We will resize these to the maximum possible array sizes
                try
                {

                    m_buf.reallocate(maxcapacity, os_nt*sv_nt);
                    m_buf.resize(1, maxsize);
                    fha = bond_action(os_nt, sv_nt);
                    cel = site_action_leaf(os_nt, sv_nt);
                    ceb = site_action_branch(os_nt, sv_nt);
                }
                catch (const std::exception &ex)
                {
                    logging::error(ex.what());
                    RAISE_EXCEPTION("Failed to reallocate internal buffers required for the sop environment");
                }


                CALL_AND_HANDLE(ham.construct_topology(A), "Failed to construct the topology of the matrix element buffer tree.");

                for (auto z : zip(ham, A, sop.contraction_info()))
                {
                    auto &mel = std::get<0>(z);
                    const auto &a = std::get<1>(z);
                    const auto &h = std::get<2>(z);
                    mel().initialise(h(), a);
                }
                ham.root()().initialise_root();
            }
            catch (const std::exception &ex)
            {
                logging::error(ex.what());
                RAISE_EXCEPTION("Failed to initialise the sop environment");
            }
        }

        inline void initialise(const ttn_type &A, const environment_type &h, container_type &ham, parameter_list && _params, size_type set_var_nthreads = 1) 
        {
            parameter_list params(_params);
            CALL_AND_RETHROW(initialise(A, h, ham, params, set_var_nthreads)); 
        }


        void clear()
        {
#ifdef TRACE_LOG
            logging::trace("clearing sum of product operator environment.");
#endif
            CALL_AND_HANDLE(m_buf.clear(), "Failed to clear sop object.");
            m_maxsize = 0;
            m_maxcapacity = 0;
        }

        inline void update_env_down(const environment_type &op, const hnode &_a1, node_type &_h)
        {
#ifdef TRACE_LOG
            logging::trace("updating all mean field operators in the sum of product operator environment.");
#endif
            CALL_AND_HANDLE(mfo_core::evaluate(op.contraction_info()[_h.id()], _a1, m_buf, _h, m_operator_sum_nthreads, m_set_var_nthreads), "Failed to evaluate the mean field operator.");
        }

        inline void update_env_up(const environment_type &op, const hnode &_a1, node_type &_h, bool force_update = true)
        {
#ifdef TRACE_LOG
            logging::trace("updating all single particle operators in the sum of product operator environment.");
#endif
            CALL_AND_HANDLE(spo_core::evaluate(op, op.contraction_info()[_h.id()], _a1, _h, m_buf, false, force_update, m_operator_sum_nthreads, m_set_var_nthreads), "Failed to evaluate the single particle operator.");
        }

        const buffer_type &buffer() const { return m_buf; }
        buffer_type &buffer() { return m_buf; }

    public:
#ifdef PYTTN_BUILD_CUDA
        static size_type maximum_dimension(const ttn_type &A)
        {
            size_type maxmfo = 0;
            // we need to fix this code, it currently doesn't work
            if (std::is_same<backend, linalg::cuda_backend>::value)
            {
                for (const auto &a : A)
                {
                    size_type size = mfo_core::contraction_buffer_size(a, true);
                    if (size > maxmfo)
                    {
                        maxmfo = size;
                    }
                }
            }
            return maxmfo;
        }
#else
        static size_type maximum_dimension(const ttn_type & /*A*/) { return 0; }
#endif

#ifdef CEREAL_LIBRARY_FOUND
    template <typename archive>
    void save(archive& ar) const
    {
#ifdef TRACE_LOG
        logging::trace("saving the sum of product operator environment.");
#endif
        CALL_AND_HANDLE(ar(cereal::make_nvp("opsum_nthreads", m_operator_sum_nthreads)), "Failed to serialise sum of product operator env.  Failed to serialise opsum nthreads.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("setvar_nthreads", m_set_var_nthreads)), "Failed to serialise sum of product operator env.  Failed to serialise set var nthreads.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("maxcapacity", m_maxcapacity)), "Failed to serialise sum of product operator env.  Failed to serialise max capacity.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("maxsize", m_maxsize)), "Failed to serialise sum of product operator env.  Failed to serialise max size.");
       
    }

    template <typename archive>
    void load(archive& ar)
    {
#ifdef TRACE_LOG
        logging::trace("loading the sum of product operator environment.");
#endif
        CALL_AND_HANDLE(ar(cereal::make_nvp("opsum_nthreads", m_operator_sum_nthreads)), "Failed to serialise sum of product operator env.  Failed to serialise opsum nthreads.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("setvar_nthreads", m_set_var_nthreads)), "Failed to serialise sum of product operator env.  Failed to serialise set var nthreads.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("maxcapacity", m_maxcapacity)), "Failed to serialise sum of product operator env.  Failed to serialise max capacity.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("maxsize", m_maxsize)), "Failed to serialise sum of product operator env.  Failed to serialise max size.");

        // resize the working arrays.  We will resize these to the maximum possible array sizes
        try
        {
            m_buf.reallocate(m_maxcapacity, m_operator_sum_nthreads*m_set_var_nthreads);
            m_buf.resize(1, m_maxsize);
            fha = bond_action( m_operator_sum_nthreads, m_set_var_nthreads);
            cel = site_action_leaf( m_operator_sum_nthreads, m_set_var_nthreads);
            ceb = site_action_branch( m_operator_sum_nthreads, m_set_var_nthreads);
        }
        catch (const std::exception &ex)
        {
            logging::error(ex.what());
            RAISE_EXCEPTION("Failed to reallocate internal buffers required for the sop environment");
        }
    }
#endif

    };

}

#endif // PYTTN_TTNS_LIB_SWEEPING_ALGORITHM_ENVIRONMENT_SUM_OF_PRODUCT_OPERATOR_ENV_HPP_
