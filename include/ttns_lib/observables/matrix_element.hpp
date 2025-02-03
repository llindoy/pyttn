#ifndef TTNS_MATRIX_ELEMENT_HPP
#define TTNS_MATRIX_ELEMENT_HPP

#include "../core/observable_node.hpp"
#include "../core/matrix_element_core.hpp"
#include "../core/matrix_element_buffer.hpp"
#include "../core/single_particle_operator.hpp"

#include "../operators/site_operators/site_operator.hpp"
#include "../operators/product_operator.hpp"
#include "../operators/sop_operator.hpp"
#include "../ttn/ttn_nodes/node_traits/bool_node_traits.hpp"

namespace ttns
{


template <typename ttn_type>
struct ttn_sop_type;

template <typename T, typename backend>
struct ttn_sop_type<ttn<T, backend>>{using sop_type = sop_operator<T, backend>;};

template <typename T, typename backend>
struct ttn_sop_type<ms_ttn<T, backend>>{using sop_type = multiset_sop_operator<T, backend>;};

template <typename T, typename backend, bool CONST>
struct ttn_sop_type<multiset_ttn_slice<T, backend, CONST> >{using sop_type = sop_operator<T, backend>;};

//TODO: Need to add support for multiset ttn slices.
//TODO: Extend each function taking two vectors to allow for different left and right types assuming they have
//      the same backend and dtype.
template <typename T, typename backend=linalg::blas_backend>
class matrix_element
{
protected:
    using matrix_type = linalg::matrix<T, backend>;
    using observable_node = typename tree<observable_node_data<T, backend>>::node_type;
    using boolnode = typename tree<bool>::node_type;

    using real_type = typename tmp::get_real_type<T>::type;
    using size_type = typename backend::size_type;

    using me_core = matrix_element_engine<T, backend>;
    using op_base = typename me_core::op_base;

    using ancestor_index = typename ttn<T, backend>::ancestor_index;

protected:
    tree<bool> m_is_identity;
    tree<observable_node_data<T, backend>> m_matel;         //swap this out to work with the multiset object instead of the single set observable node object.

    matrix_element_buffer<T, backend> m_buf;
    linalg::matrix<T> m_hmat;

public:
    matrix_element() {}
    template <typename state_type>
    matrix_element(const state_type& A, size_type numbuff = 1, bool use_capacity = false) {CALL_AND_HANDLE(resize(A, numbuff, use_capacity), "Failed to construct matrix_element object.  Failed to allocate internal buffers.");}
    template <typename state_type>
    matrix_element(const state_type& A, const state_type& B, size_type numbuff = 1, bool use_capacity = false) {CALL_AND_HANDLE(resize(A, B, numbuff, use_capacity), "Failed to construct matrix_element object.  Failed to allocate internal buffers.");}

    template <typename state_type>
    matrix_element(const state_type& A, const typename ttn_sop_type<state_type>::sop_type& sop, size_type numbuff = 1, bool use_capacity = false) {CALL_AND_HANDLE(resize(A, sop, numbuff, use_capacity), "Failed to construct matrix_element object.  Failed to allocate internal buffers.");}
    template <typename state_type>
    matrix_element(const state_type& A, const state_type& B, const typename ttn_sop_type<state_type>::sop_type& sop, size_type numbuff = 1, bool use_capacity = false) {CALL_AND_HANDLE(resize(A, B, sop, numbuff, use_capacity), "Failed to construct matrix_element object.  Failed to allocate internal buffers.");}

    matrix_element(const matrix_element& o) = default;
    matrix_element(matrix_element&& o) = default;

    matrix_element& operator=(const matrix_element& o) = default;
    matrix_element& operator=(matrix_element&& o) = default;

    void clear()
    {
        try
        {
            m_matel.clear();
            m_is_identity.clear();
            m_buf.clear();
            m_hmat.clear();
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear matrix element object.");
        }
    }

    template <typename state_type>
    void resize(const state_type& A, size_type numbuff = 1, bool use_capacity = false)
    {
        CALL_AND_RETHROW(resize(A, A, numbuff, use_capacity));
    }

    template <typename state_type>
    void resize(const state_type& A, const state_type& B, size_type numbuff = 1, bool use_capacity = false)
    {
        try
        {
            using common::zip;   using common::rzip;
            ASSERT(has_same_structure(A, B), "The input hierarchical tucker tensors do not have the same topology.");

            CALL_AND_RETHROW(resize_to_fit(A, B, numbuff, use_capacity));
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize internal buffers of the matrix_element object.");
        }
    }

    template <typename state_type>
    void resize(const state_type& A, const typename ttn_sop_type<state_type>::sop_type& sop, size_type numbuff = 1, bool use_capacity = false)
    {
        CALL_AND_RETHROW(resize(A, A, sop, numbuff, use_capacity));
    }

    template <typename state_type>
    void resize(const state_type& A, const state_type& B,  const typename ttn_sop_type<state_type>::sop_type& sop, size_type numbuff = 1, bool use_capacity = false)
    {
        try
        {
            using common::zip;   using common::rzip;
            ASSERT(has_same_structure(A, B), "The input hierarchical tucker tensors do not have the same topology.");
            ASSERT(has_same_structure(A, sop.contraction_info()), "The input state and sop operator do not have the same topology.");

            CALL_AND_RETHROW(resize_to_fit(A, B, sop, numbuff, use_capacity));
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize internal buffers of the matrix_element object.");
        }
    }


protected:
    template <typename state_type>
    void compute_norm_internal(const state_type& psi, bool use_sparsity, size_t set_index, size_t r = 0)
    {
        try
        {
            using common::zip;   using common::rzip;
            CALL_AND_RETHROW(resize_to_fit(psi, psi));

            if(use_sparsity && psi.has_orthogonality_centre())
            {
                //if we have an orthognality centre and have elected to use sparsity then we construct the indexing objec starting at the orthognality centre
                ancestor_index inds;
                psi.ancestor_indexing(psi.orthogonality_centre(), inds);

                //first iterate through the nodes we plan on traversing and flag that the nodes are identity
                for(const auto& pair : reverse(inds))
                {
                    size_type ind = std::get<0>(pair);
                    if(!psi[ind].is_leaf()){for(auto& c : m_is_identity[ind]){c() = true;}}
                    m_is_identity[ind]() = false;
                }

                for(const auto& pair : inds)
                {
                    size_type ind = std::get<0>(pair);
                    const auto& p = psi[ind]; auto& mel = m_matel[ind]; auto& is_id = m_is_identity[ind];       
                    update_expectation_value(p, set_index, mel, r, is_id, m_buf);
                }
            }
            else
            {
                for(auto z : rzip(psi, m_matel, m_is_identity))
                {
                    const auto& p = std::get<0>(z); auto& mel = std::get<1>(z); auto& is_id = std::get<2>(z); 
                    update_expectation_value(p, set_index, mel, r, is_id, m_buf);
                }
            }
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing inner product of hierarchical tucker tensor with itself.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute inner product of hierarchical tucker tensor with itself.");
        }
    }
public:
    template <typename state_type>
    real_type operator()(const state_type& psi){ CALL_AND_RETHROW(return this->operator()(psi, true));}

    template <typename state_type>
    real_type operator()(const state_type& psi, bool use_sparsity)
    {
        try
        {
            using common::zip;   using common::rzip;
            CALL_AND_RETHROW(resize_to_fit(psi, psi));

            real_type retval = 0.0;
            size_type r = 0;

            if(use_sparsity && psi.has_orthogonality_centre())
            {
                //if we have an orthognality centre and have elected to use sparsity then we construct the indexing objec starting at the orthognality centre
                ancestor_index inds;
                psi.ancestor_indexing(psi.orthogonality_centre(), inds);

                //first iterate through the nodes we plan on traversing and flag that the nodes are identity
                for(const auto& pair : reverse(inds))
                {
                    size_type ind = std::get<0>(pair);
                    if(!psi[ind].is_leaf()){for(auto& c : m_is_identity[ind]){c() = true;}}
                    m_is_identity[ind]() = false;
                }

                for(size_type set_index = 0; set_index < psi.nset(); ++set_index)
                {
                    for(const auto& pair : inds)
                    {
                        size_type ind = std::get<0>(pair);
                        const auto& p = psi[ind]; auto& mel = m_matel[ind]; auto& is_id = m_is_identity[ind];       
                        update_expectation_value(p, set_index, mel, r, is_id, m_buf);
                    }
                    CALL_AND_HANDLE(retval += linalg::real(gather_result(m_matel[0]()[r])), "Failed to return result.");
                }
            }
            else
            {
                for(size_type set_index = 0; set_index < psi.nset(); ++set_index)
                {
                    for(auto z : rzip(psi, m_matel, m_is_identity))
                    {
                        const auto& p = std::get<0>(z); auto& mel = std::get<1>(z); auto& is_id = std::get<2>(z); 
                        update_expectation_value(p, set_index, mel, r, is_id, m_buf);
                    }
                    CALL_AND_HANDLE(retval += linalg::real(gather_result(m_matel[0]()[r])), "Failed to return result.");
                }
            }
            
            return retval;
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing inner product of hierarchical tucker tensor with itself.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute inner product of hierarchical tucker tensor with itself.");
        }
    }

protected:
    template <typename state_type, typename optype>
    bool validate_operator(optype&& /* op */, size_type mode, const state_type& psi)
    {
        return mode < psi.nmodes();
    }

    template <typename state_type, typename optype>
    bool validate_operator(const std::vector<optype>& op, const std::vector<size_type> modes, const state_type& psi)
    {
        if(modes.size() != op.size()){return false;}
        for(const auto& mode : modes)
        {
            if (mode >= psi.nmodes()){return false;}
        }
        return true;
    }

    template <typename state_type, typename optype, typename mode_type>
    inline T expectation_value_internal(optype&& op, const mode_type& mode, const state_type& psi, bool use_sparsity = true)
    {
        using common::zip;   using common::rzip;
        ASSERT(validate_operator(std::forward<optype>(op), mode, psi), "The mode that the input operator acts on is out of bounds.");
        CALL_AND_RETHROW(resize_to_fit(psi, psi));

        T retval = T(0.0)*0.0;

        size_type r = 0;
        if(use_sparsity && psi.has_orthogonality_centre())
        {
            //if we have an orthognality centre and have elected to use sparsity then we construct the indexing objec starting at the orthognality centre
            ancestor_index inds;
            psi.ancestor_indexing_leaf(mode, inds);
            psi.ancestor_indexing(psi.orthogonality_centre(), inds);

            //first iterate through the nodes we plan on traversing and flag that only some of the nodes associated with 
            //it are not the identity
            for(const auto& pair : reverse(inds))
            {
                size_type ind = std::get<0>(pair);
                if(!psi[ind].is_leaf()){for(auto& c : m_is_identity[ind]){c() = true;}}
                m_is_identity[ind]() = false;
            }

            for(size_type set_index = 0; set_index < psi.nset(); ++set_index)
            {
                for(const auto& pair : inds)
                {
                    size_type ind = std::get<0>(pair);
                    const auto& p = psi[ind]; auto& mel = m_matel[ind]; auto& is_id = m_is_identity[ind];       
                    CALL_AND_HANDLE(update_expectation_value(p, set_index, mel, r, is_id, op, mode, m_buf), "Failed to update expectation value.");
                }
                CALL_AND_HANDLE(retval += gather_result(m_matel[0]()[r]), "Failed to return result.");
            }
        }
        else
        {
            for(size_type set_index = 0; set_index < psi.nset(); ++set_index)
            {
                for(auto z : rzip(psi, m_matel, m_is_identity))
                {
                    const auto& p = std::get<0>(z); auto& mel = std::get<1>(z); auto& is_id = std::get<2>(z); 
                    CALL_AND_HANDLE(update_expectation_value(p, set_index, mel, r, is_id, std::forward<optype>(op), mode, m_buf), "Failed to update expectation value.");
                }
                CALL_AND_HANDLE(retval += gather_result(m_matel[0]()[r]), "Failed to return result.");
            }
        }

        return retval;
    }

public:
    template <typename state_type>
    inline T operator()(site_operator<T, backend>& op, const state_type& psi)
    {
        CALL_AND_HANDLE(return this->expectation_value_internal(op.op(), op.mode(), psi, true), "Failed to compute expectation value.");
    }

    template <typename state_type>
    inline T operator()(site_operator<T, backend>& op, size_type mode, const state_type& psi)
    {
        CALL_AND_HANDLE(return this->expectation_value_internal(op.op(), mode, psi, true), "Failed to compute expectation value.");
    }

    template <typename state_type>
    inline T operator()(site_operator<T, backend>& op, const state_type& psi, bool use_sparsity)
    {
        CALL_AND_HANDLE(return this->expectation_value_internal(op.op(), op.mode(), psi, use_sparsity), "Failed to compute expectation value.");
    }

    template <typename state_type>
    inline T operator()(site_operator<T, backend>& op, size_type mode, const state_type& psi, bool use_sparsity)
    {
        CALL_AND_HANDLE(return this->expectation_value_internal(op.op(), mode, psi, use_sparsity), "Failed to compute expectation value.");
    }

    //a function that calculates the expectation value of an operator acting on a single mode of the ttns.  
    template <typename state_type, typename op_type>
    inline typename std::enable_if<std::is_base_of<op_base, op_type>::value, T>::type operator()(op_type& op, size_type mode, const state_type& psi)
    {
        CALL_AND_HANDLE(return this->expectation_value_internal(op, mode, psi, true), "Failed to compute expectation value.");
    }

    template <typename state_type, typename op_type>
    inline typename std::enable_if<std::is_base_of<op_base, op_type>::value, T>::type operator()(op_type& op, size_type mode, const state_type& psi, bool use_sparsity)
    {
        CALL_AND_HANDLE(return this->expectation_value_internal(op, mode, psi, use_sparsity), "Failed to compute expectation value.");
    }

    template <typename state_type>
    inline T operator()(std::vector<site_operator<T, backend>>& ops, const state_type& psi)
    {
        std::vector<size_type> modes(ops.size());
        for(size_type i = 0; i < ops.size(); ++i){modes[i] = ops[i].mode();}
        CALL_AND_HANDLE(return this->expectation_value_internal(ops, modes, psi, true), "Failed to compute expectation value.");
    }

    template <typename state_type>
    inline T operator()(std::vector<site_operator<T, backend>>& ops, const std::vector<size_type>& modes, const state_type& psi)
    {
        CALL_AND_HANDLE(return this->expectation_value_internal(ops, modes, psi, true), "Failed to compute expectation value.");
    }

    template <typename state_type>
    inline T operator()(std::vector<site_operator<T, backend>>& ops, const state_type& psi, bool use_sparsity)
    {
        std::vector<size_type> modes(ops.size());
        for(size_type i = 0; i < ops.size(); ++i){modes[i] = ops[i].mode();}
        CALL_AND_HANDLE(return this->expectation_value_internal(ops, modes, psi, use_sparsity), "Failed to compute expectation value.");
    }

    template <typename state_type>
    inline T operator()(std::vector<site_operator<T, backend>>& ops, const std::vector<size_type>& modes, const state_type& psi, bool use_sparsity)
    {
        CALL_AND_RETHROW(resize_to_fit(psi, psi));
        CALL_AND_HANDLE(return this->expectation_value_internal(ops, modes, psi, use_sparsity), "Failed to compute expectation value.");
    }


    template <typename state_type>
    inline T operator()(product_operator<T, backend>& ops, const state_type& psi)
    {
        CALL_AND_RETHROW(return ops.coeff()*this->operator()(ops.mode_operators(), psi));
    }

    template <typename state_type>
    inline T operator()(product_operator<T, backend>& ops, const state_type& psi, bool use_sparsity)
    {
        CALL_AND_RETHROW(return ops.coeff()*this->operator()(ops.mode_operators(), psi, use_sparsity));
    }
public:
    template <typename state_type>
    T operator()(const state_type& bra, const state_type& ket)
    {
        try
        {
            using common::zip;   using common::rzip;
            if(&bra == &ket){CALL_AND_RETHROW(return operator()(bra));}
            ASSERT(bra.nset() == ket.nset(), "Cannot compute matrix element between tensor networks with different set sizes.");
            CALL_AND_RETHROW(resize_to_fit(bra, ket));

            size_type r = 0;
            T retval = T(0.0)*0.0;
            for(size_type set_index = 0; set_index < bra.nset(); ++set_index)
            {
                for(auto z : rzip(bra, ket, m_matel))
                {
                    const auto& b = std::get<0>(z); const auto& k = std::get<1>(z); auto& mel = std::get<2>(z);
                    CALL_AND_RETHROW(m_buf.resize(k(set_index).shape(0), k(set_index).shape(1)));
                    CALL_AND_HANDLE(update_matrix_element(b, set_index, k, set_index, mel, r, m_buf), "Failed to update matrix element.");
                }

                CALL_AND_HANDLE(retval += gather_result(m_matel[0]()[r]), "Failed to return result.");
            }
            return retval;
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing inner product of two hierarchical tucker tensors.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute inner product of two hierarchical tucker tensors.");
        }
    }

protected:
    template <typename state_type, typename mode_type, typename optype>
    inline T matrix_element_internal(optype&& op, const mode_type& mode, const state_type& bra, const state_type& ket)
    {
        using common::zip;   using common::rzip;
        if(&bra == &ket){CALL_AND_RETHROW(return expectation_value_internal(op, mode, bra));}
        ASSERT(bra.nset() == ket.nset(), "Cannot compute matrix element between tensor networks with different set sizes.");

        ASSERT(validate_operator(std::forward<optype>(op), mode, ket), "The mode that the input operator acts on is out of bounds.");
        CALL_AND_RETHROW(resize_to_fit(bra, ket));

        size_type r = 0;
        T retval = T(0.0)*0.0;
        for(size_type set_index = 0; set_index < bra.nset(); ++set_index)
        {
            for(auto z : rzip(bra, ket, m_matel))
            {
                const auto& b = std::get<0>(z); const auto& k = std::get<1>(z); auto& mel = std::get<2>(z);
                update_matrix_element(b, set_index, k, set_index, mel, r, std::forward<optype>(op), mode, m_buf);
            }

            CALL_AND_HANDLE(retval += gather_result(m_matel[0]()[r]), "Failed to return result.");
        }
        return retval;
    }

public:
    //a function that calculates the matrix element of an operator that acts on single mode of the ttns.      
    template <typename state_type, typename op_type>
    inline typename std::enable_if<std::is_base_of<op_base, op_type>::value, T>::type operator()(op_type& op, size_type mode, const state_type& bra, const state_type& ket)
    {
        CALL_AND_HANDLE(return matrix_element_internal(op, mode, bra, ket), "Failed to compute matrix element.");
    }


    template <typename state_type>
    inline T operator()(site_operator<T, backend>& op, const state_type& bra, const state_type& ket)
    {
        CALL_AND_HANDLE(return matrix_element_internal(op, op.mode(), bra, ket), "Failed to compute matrix element.");
    }

    template <typename state_type>
    inline T operator()(site_operator<T, backend>& op, size_type mode, const state_type& bra, const state_type& ket)
    {
        CALL_AND_HANDLE(return matrix_element_internal(op, mode, bra, ket), "Failed to compute matrix element.");
    }

    template <typename state_type>
    inline T operator()(std::vector<site_operator<T, backend>>& ops, const state_type& bra, const state_type& ket)
    {
        std::vector<size_type> modes(ops.size());
        for(size_type i = 0; i < ops.size(); ++i){modes[i] = ops[i].mode();}
        CALL_AND_HANDLE(return this->matrix_element_internal(ops, modes, bra, ket), "Failed to compute matrix element.");
    }

    template <typename state_type>
    inline T operator()(std::vector<site_operator<T, backend>>& ops, const std::vector<size_type>& modes, const state_type& bra, const state_type& ket)
    {
        CALL_AND_HANDLE(return this->matrix_element_internal(ops, modes, bra, ket), "Failed to compute matrix element.");
    }    

    template <typename state_type>
    inline T operator()(product_operator<T, backend>& ops, const state_type& bra, const state_type& ket)
    {
        CALL_AND_RETHROW(return ops.coeff()*this->operator()(ops.mode_operators(), bra, ket));
    }

protected:
    inline T accum_root(const sttn_node_data<T>& hr, T Eshift = T(0), bool use_identity = false)
    {
        T retval = T(0.0)*0.0;
        if(!use_identity)
        {
            retval = Eshift;
        }
        for(size_t j = 0; j < hr.nterms(); ++j)
        {
            CALL_AND_HANDLE(retval += hr[j].coeff()*gather_result(m_matel[0]()[j]), "Failed to return result.");
            if(use_identity && std::abs(Eshift) > 1e-14)
            {
                CALL_AND_HANDLE(retval += Eshift*gather_result(m_matel[0]().id()), "Failed to return result.");
            }
        }
        return retval;
    }

    inline T accum_root(const sttn_node_data<T>& hr, size_t i, size_t c, T Eshift = T(0), bool use_identity = false)
    {
        ASSERT(i == 0 && c == 0, "Index out of bounds.");
        return accum_root(hr, Eshift, use_identity = false);
    }

    inline T accum_root(const multiset_sttn_node_data<T>& hr, size_t i, size_t c, T Eshift = T(0), bool use_identity = false)
    {
        return accum_root(hr[i][c], Eshift, use_identity = false);
    }

public:
    template <typename state_type>
    inline T operator()(typename ttn_sop_type<state_type>::sop_type& sop, const state_type& psi)
    {
        try
        {
            using spo = single_particle_operator_engine<T, backend>;
            using common::zip;   using common::rzip;
            ASSERT(has_same_structure(psi, sop.contraction_info()), "The input hiearchical tucker tensors do not both have the same topology as the matrix_element object.");
            ASSERT(psi.nset() == sop.nset(), "Cannot compute matrix element between tensor networks with different set sizes.");
            CALL_AND_RETHROW(resize_to_fit(psi, psi, sop));

            bool compute_identity = !(psi.is_orthogonalised());
            T retval = T(0.0)*0.0;

            //to do modify this code to handle multiset sops
            for(size_type set_index = 0; set_index <psi.nset(); ++set_index)
            {
                for(size_t nr = 0; nr < sop.nrow(set_index); ++nr)
                {
                    bool use_identity = false;
                    T Eshift = sop.Eshift_val(set_index, nr);
                    size_t col = sop.column_index(set_index, nr);
                    if(col == set_index && sop.is_scalar(set_index, nr) && std::abs(Eshift) > 1e-14)
                    {
                        CALL_AND_HANDLE(this->compute_norm_internal(psi, true, set_index, 0), "Failed to compute norm of diagonal term.");
                        T val(0);
                        CALL_AND_HANDLE(val += linalg::real(gather_result(m_matel[0]()[0])), "Failed to return result.");
                        retval += Eshift*val;
                    }
                    else
                    {
                        for(auto z : rzip(psi, m_matel, sop.contraction_info()))
                        {
                            const auto& p = std::get<0>(z); auto& mel = std::get<1>(z);  const auto& h = std::get<2>(z);
                            CALL_AND_RETHROW(m_buf.resize(p(set_index).shape(0), p(set_index).shape(1)));

                            CALL_AND_HANDLE(spo::evaluate(sop, h, p, p, set_index, nr, mel, m_buf.HA, m_buf.temp, compute_identity), "Failed to compute expectation value.");
                        }
                        const auto& hr = sop.contraction_info().root()();
                        CALL_AND_HANDLE(retval += accum_root(hr, set_index, nr, Eshift, use_identity), "Failed to return result.");
                    }

                }
            }
            return retval;
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing inner product of two hierarchical tucker tensors.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute inner product of two hierarchical tucker tensors.");
        }
    }

    //TODO: optimise the evaluation of multiset sops where there are no non-trivial terms in the index
    template <typename state_type>
    inline T operator()(typename ttn_sop_type<state_type>::sop_type& sop, const state_type& bra, const state_type& ket)
    {
        if(&bra == &ket){CALL_AND_RETHROW(return this->operator()(sop, ket););}
        try
        {
            using spo = single_particle_operator_engine<T, backend>;
            using common::zip;   using common::rzip;
            ASSERT(has_same_structure(bra, sop.contraction_info()) && has_same_structure(ket, bra), "The input hiearchical tucker tensors do not both have the same topology as the matrix_element object.");
            ASSERT(ket.nset() == sop.nset() && bra.nset() == ket.nset(), "Cannot compute matrix element between tensor networks with different set sizes.");
            CALL_AND_RETHROW(resize_to_fit(bra, ket, sop));

            bool compute_identity = !(bra.is_orthogonalised() && ket.is_orthogonalised());
            T retval = T(0.0)*0.0;
            //to do modify this code to handle multiset sops
            for(size_type set_index = 0; set_index <ket.nset(); ++set_index)
            {
                for(size_t nr = 0; nr < sop.nrow(set_index); ++nr)
                {
                    for(auto z : rzip(bra, ket, m_matel, sop.contraction_info()))
                    {
                        const auto& b = std::get<0>(z); const auto& k = std::get<1>(z); auto& mel = std::get<2>(z);  const auto& h = std::get<3>(z);
                        CALL_AND_RETHROW(m_buf.resize(k(set_index).shape(0), k(set_index).shape(1)));
                        CALL_AND_HANDLE(spo::evaluate(sop, h, b, k, set_index, nr, mel, m_buf.HA, m_buf.temp, compute_identity), "Failed to compute expectation value.");
                    }

                    const auto& hr = sop.contraction_info().root()();
                    T Eshift = sop.Eshift_val(set_index, nr);
                    CALL_AND_HANDLE(retval += accum_root(hr, set_index, nr, Eshift, true), "Failed to return result.");
                }
            }
            return retval;
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing inner product of two hierarchical tucker tensors.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute inner product of two hierarchical tucker tensors.");
        }
    }

    /* 
protected:
    template <typename state_type, typename mode_type, typename optype>
    inline std::vector<T>& matrix_element_repeated(std::vector<optype>& op, const mode_type& mode, const state_type& bra, const state_type& ket, std::vector<T>& res)
    {

        using common::zip;   using common::rzip;
        if(&bra == &ket)
        {
            CALL_AND_HANDLE(res.resize(op.size()), "Failed to resize result buffer");   
            for(size_type i = 0; i < res.size(); ++i)
            {
                CALL_AND_RETHROW(res[i] = expectation_value_internal(op[i], mode, bra));
            }
            return res;
        }
        else
        {
            ASSERT(bra.nset() == ket.nset(), "Cannot compute matrix element between tensor networks with different set sizes.");

            ASSERT(has_same_structure(bra, m_matel) && has_same_structure(ket, m_matel), "The two input hiearchical tucker tensor do not both have the same topology as the matrix_element object.");

            //first go through and validate the operator object
            for(size_type i = 0; i < op.size(); ++i)
            {
                ASSERT(validate_operator(op[i], mode, ket), "The mode that the input operator acts on is out of bounds.");
            }

            size_type  r = 0;
            //set up the result buffer
            CALL_AND_HANDLE(res.resize(op.size()), "Failed to resize result buffer");   
            for(size_type i = 0; i < res.size(); ++i){res[i] = T(0.0)*0.0;}

            //now we iterate over the set indices
            for(size_type set_index = 0; set_index < bra.nset(); ++set_index)
            {
                //now get the nodes that will be different in the evaluation of each of the ops
                ancestor_index inds;
                ket.ancestor_indexing_leaf(mode, inds);

                //now we iterate over the observable index
                for(size_type op_index = 0; op_index < op.size(); ++op_index)
                {
                    //in the first iteration of the loop we do the full calculation ensuring all of the mel tensors have been correctly set.
                    if(op_index == 0)
                    {
                        for(auto z : rzip(bra, ket, m_matel, m_is_identity))
                        {
                            const auto& b = std::get<0>(z); const auto& k = std::get<1>(z); auto& mel = std::get<2>(z); auto& is_id = std::get<3>(z);
                            update_matrix_element(b, set_index, k, set_index, mel, r, is_id, op[op_index], mode, m_buf);
                        }
                        CALL_AND_HANDLE(res[op_index] += gather_result(m_matel[0]()[r]), "Failed to return result.");
                    }
                    //in all subsequent iterations we only need to update the nodes that will change.  In constrast to the expectation value case we don't set the 
                    //is_identity variables to true as they in general won't be
                    else
                    {
                        for(const auto& pair : inds)
                        {
                            size_type ind = std::get<0>(pair);
                            const auto& b = bra[ind]; const auto& k = ket[ind]; auto& mel = m_matel[ind]; auto& is_id = m_is_identity[ind];       
                            update_matrix_element(b, set_index, k, set_index, mel, r, is_id, op[op_index], mode, m_buf);
                        }
                        CALL_AND_HANDLE(res[op_index] += gather_result(m_matel[0]()[r]), "Failed to return result.");
                    }
                }
            }

            return res;
        }
    }

public:
    template <typename state_type>
    inline std::vector<T>& operator()(std::vector<site_operator<T, backend>>& op, size_type mode, const state_type& bra, const state_type& ket, std::vector<T>& res)
    {
        CALL_AND_HANDLE(return matrix_element_internal(op, mode, bra, ket, res), "Failed to compute matrix element.");
    }

    //a function that calculates the matrix element of an operator acting on a set of modes of the ttns.  
    template <typename state_type>
    inline std::vector<T>& operator()(std::vector<std::vector<site_operator<T, backend>>>& ops, const std::vector<size_type>& modes, const state_type& bra, const state_type& ket, std::vector<T>& res)
    {
        CALL_AND_HANDLE(return this->matrix_element_repeated(ops, modes, bra, ket, res), "Failed to compute matrix element.");
    }

 */
protected:
#ifdef __NVCC__
    T gather_result(const linalg::matrix<T, linalg::cuda_backend>& o)
    {
        CALL_AND_HANDLE(m_hmat = o, "Failed to copy device result back to host.");
        CALL_AND_HANDLE(return m_hmat(0,0), "Failed to return result.");
    }
#endif

    T gather_result(const linalg::matrix<T, linalg::blas_backend>& o)
    {
        CALL_AND_HANDLE(return o(0,0), "Failed to return result.");
    }


protected:
    /*
     * Functions for updating expectation values for the node.
     */
    template <typename state_node>
    static void update_expectation_value(const state_node& p, size_type i1, observable_node& mel, size_type r, boolnode& is_id, matrix_element_buffer<T, backend>& buf)
    {
        CALL_AND_RETHROW(buf.resize(p(i1).shape(0), p(i1).shape(1)));
        if(!p.is_leaf())
        {
            CALL_AND_HANDLE(me_core::compute_branch(p(i1), buf.HA, buf.temp, mel, r, is_id), "Failed to compute root node contraction.");
        }
        else
        {
            CALL_AND_HANDLE(me_core::compute_leaf(p(i1), mel, r, is_id), "Failed to compute leaf node contraction.");
        }
    }

    template <typename state_node, typename optype>
    static void update_expectation_value(const state_node& p, size_type i1, observable_node& mel, size_type r, boolnode& is_id, optype&& op, size_type mode, matrix_element_buffer<T, backend>& buf)
    {
        CALL_AND_RETHROW(buf.resize(p(i1).shape(0), p(i1).shape(1)));
        if(!p.is_leaf())
        {
            CALL_AND_HANDLE(me_core::compute_branch(p(i1), buf.HA, buf.temp, mel, r, is_id), "Failed to compute root node contraction.");
        }
        else
        {
            if(p.leaf_index() != mode)
            {
                CALL_AND_HANDLE(me_core::compute_leaf(p(i1), mel, r, is_id), "Failed to compute leaf node contraction.");
            }
            else
            {
                CALL_AND_HANDLE(me_core::compute_leaf(std::forward<optype>(op), p(i1), buf.HA, mel, r, is_id), "Failed to compute leaf node contraction.");
            }
        }
    }

    template <typename state_node>
    static void update_expectation_value(const state_node& p, size_type i1, observable_node& mel, size_type r, boolnode& is_id, std::vector<site_operator<T, backend>>& ops, const std::vector<size_type>& modes, matrix_element_buffer<T, backend>& buf)
    {
        CALL_AND_RETHROW(buf.resize(p(i1).shape(0), p(i1).shape(1)));
        if(!p.is_leaf())
        {
            CALL_AND_HANDLE(me_core::compute_branch(p(i1), buf.HA, buf.temp, mel, r, is_id), "Failed to compute root node contraction.");
        }
        else
        {
            auto it = std::find(modes.begin(), modes.end(), p.leaf_index());
            if(it == modes.end())
            {
                CALL_AND_HANDLE(me_core::compute_leaf(p(i1), mel, r, is_id), "Failed to compute leaf node contraction.");
            }
            else
            {
                CALL_AND_HANDLE(me_core::compute_leaf(ops[it-modes.begin()], p(i1), buf.HA, mel, r, is_id), "Failed to compute leaf node contraction.");
            }
        }
    }

    template <typename state_node>
    static void update_matrix_element(const state_node& b, size_type i1, const state_node& k, size_type i2, observable_node& mel, size_type r, matrix_element_buffer<T, backend>& buf)
    {
        CALL_AND_RETHROW(buf.resize(k(i2).shape(0), k(i2).shape(1)));
        if(!b.is_leaf())
        {
            CALL_AND_HANDLE(me_core::compute_branch(b(i1), k(i2), buf.HA, buf.temp, mel, r), "Failed to compute root node contraction.");
        }
        else
        {
            CALL_AND_HANDLE(me_core::compute_leaf(b(i1), k(i2), mel, r), "Failed to compute leaf node contraction.");
        }
    }

    template <typename state_node, typename optype>
    static void update_matrix_element(const state_node& b, size_type i1, const state_node& k, size_type i2, observable_node& mel, size_type r, optype&& op, size_type mode, matrix_element_buffer<T, backend>& buf)
    {
        CALL_AND_RETHROW(buf.resize(k(i2).shape(0), k(i2).shape(1)));
        if(!b.is_leaf())
        {
            CALL_AND_HANDLE(me_core::compute_branch(b(i1), k(i2), buf.HA, buf.temp, mel, r), "Failed to compute root node contraction.");
        }
        else
        {
            if(b.leaf_index() != mode)
            {
                CALL_AND_HANDLE(me_core::compute_leaf(b(i1), k(i2), mel, r), "Failed to compute leaf node contraction.");
            }
            else
            {
                CALL_AND_HANDLE(me_core::compute_leaf(op, b(i1), k(i2), buf.HA, mel, r), "Failed to compute leaf node contraction.");
            }
        }
    }

    template <typename state_node>
    static void update_matrix_element(const state_node& b, size_type i1, const state_node& k, size_type i2, observable_node& mel, size_type r, std::vector<site_operator<T, backend>>& ops, const std::vector<size_type>& modes, matrix_element_buffer<T, backend>& buf)
    {
        CALL_AND_RETHROW(buf.resize(k(i2).shape(0), k(i2).shape(1)));
        if(!b.is_leaf())
        {
            CALL_AND_HANDLE(me_core::compute_branch(b(i1), k(i2), buf.HA, buf.temp, mel, r), "Failed to compute root node contraction.");
        }
        else
        {
            auto it = std::find(modes.begin(), modes.end(), b.leaf_index());
            if(it == modes.end())
            {
                CALL_AND_HANDLE(me_core::compute_leaf(b(i1), k(i2), mel, r), "Failed to compute leaf node contraction.");
            }
            else
            {
                CALL_AND_HANDLE(me_core::compute_leaf(ops[it-modes.begin()], b(i1), k(i2), buf.HA, mel, r), "Failed to compute leaf node contraction.");
            }
        }
    }

protected:
    inline size_t get_nterms(const sttn_node_data<T>& hr, size_t i, size_t c)
    {
        ASSERT(i == 0 && c == 0, "Index out of bounds.");
        return hr.nterms();
    }

    inline size_t get_nterms(const multiset_sttn_node_data<T>& hr, size_t i, size_t c)
    {
        return hr[i][c].nterms();
    }

    template <typename state_type>
    void resize_to_fit(const state_type& A, const state_type& B,  const typename ttn_sop_type<state_type>::sop_type& sop, size_type numbuff = 1, bool use_capacity = false)
    {
        try
        {
            using common::zip;   using common::rzip;
            ASSERT(has_same_structure(A, B), "The input hierarchical tucker tensors do not have the same topology.");
            ASSERT(has_same_structure(A, sop.contraction_info()), "The input state and sop operator do not have the same topology.");

            if(!has_same_structure(A, m_matel))
            {
                m_matel.clear();
                m_is_identity.clear();
                CALL_AND_HANDLE(m_matel.construct_topology(B), "Failed to construct the topology of the matrix element buffer tree.");
                CALL_AND_HANDLE(m_is_identity.construct_topology(B), "Failed to construct the topology of the is_identity matrix tree.");
            }

            size_t maxcapacity = 0;
            for(auto z : zip(m_matel, m_is_identity, A, B, sop.contraction_info()))
            {
                auto& mel = std::get<0>(z); auto& is_id = std::get<1>(z); const auto& a = std::get<2>(z); const auto& b = std::get<3>(z);   const auto& h = std::get<4>(z);
                is_id() = false;

                mel().has_identity() = (&A != &B) && A.is_orthogonalised() && B.is_orthogonalised();

                size_t nterms = 0;

                for(size_t iset = 0; iset < sop.nset(); ++iset)
                {   
                    for(size_t j = 0; j < sop.nrow(iset); ++j)
                    {
                        size_t nt = get_nterms(h(), iset, j);
                        if(nt == 0){nt = 1;}
                        if(nt > nterms){nterms = nt;}

                        size_t jset = sop.column_index(iset, j);
                        if(iset != jset){mel().has_identity() = true;}
                    }
                }
                if(nterms > mel().size())
                {
                    if(mel().size() == 0)
                    {
                        CALL_AND_HANDLE(mel().resize(nterms), "Failed to resize matrix element container.");
                    }
                    else
                    {
                        CALL_AND_HANDLE(mel().expand_buffer(nterms), "Failed to resize matrix element container.");
                    }
                }
                size_t capacity = b.maxhrank(use_capacity)*a.maxhrank(use_capacity);
                if(capacity > mel().matrix_capacity())
                {
                    CALL_AND_HANDLE(mel().reallocate_matrices(capacity), "Failed to reallocate mel tensors.")
                }
                mel().store_identity();

                capacity = matrix_element_buffer<T, backend>::get_capacity(a, b, use_capacity);
                if(capacity > maxcapacity){maxcapacity = capacity;}
            }

            if(numbuff > m_buf.buf || maxcapacity > m_buf.cap)
            {
                m_buf.reallocate(maxcapacity, numbuff);
                m_buf.resize(1, maxcapacity);
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize internal buffers of the matrix_element object.");
        }
    }

    template <typename state_type>
    void resize_to_fit(const state_type& A, const state_type& B, size_type numbuff = 1, bool use_capacity = false)
    {
        try
        {
            using common::zip;   using common::rzip;
            ASSERT(has_same_structure(A, B), "The input hierarchical tucker tensors do not have the same topology.");
            if(!has_same_structure(A, m_matel))
            {
                m_matel.clear();
                m_is_identity.clear();
                CALL_AND_HANDLE(m_matel.construct_topology(B), "Failed to construct the topology of the matrix element buffer tree.");
                CALL_AND_HANDLE(m_is_identity.construct_topology(B), "Failed to construct the topology of the is_identity matrix tree.");
            }

            size_t maxcapacity = 0;
            for(auto z : zip(m_matel, m_is_identity, A, B))
            {
                auto& mel = std::get<0>(z); auto& is_id = std::get<1>(z); const auto& a = std::get<2>(z); const auto& b = std::get<3>(z);
                is_id() = false;

                if(numbuff > mel().size())
                {
                    if(mel().size() == 0)
                    {
                        CALL_AND_HANDLE(mel().resize(numbuff), "Failed to resize matrix element container.");
                    }
                    else
                    {
                        CALL_AND_HANDLE(mel().expand_buffer(numbuff), "Failed to resize matrix element container.");
                    }
                }

                size_t ncap = a.maxhrank(use_capacity)*b.maxhrank(use_capacity);
                if(mel().matrix_capacity() < ncap)
                {
                    CALL_AND_HANDLE(mel().reallocate_matrices(ncap), "Failed to reallocate mel tensors.")
                }

                auto melsize = mel().matrix_size();
                if(melsize[0] < a.maxhrank() && melsize[1] < b.maxhrank())
                {
                    CALL_AND_HANDLE(mel().resize_matrices(a.maxhrank(), b.maxhrank()), "Failed to reallocate mel tensors.");
                }

                mel().store_identity();
                size_t capacity = matrix_element_buffer<T, backend>::get_capacity(a, b, use_capacity);
                if(capacity > maxcapacity){maxcapacity = capacity;}
            }

            if(numbuff > m_buf.buf || maxcapacity > m_buf.cap)
            {
                m_buf.reallocate(maxcapacity, numbuff);
                m_buf.resize(1, maxcapacity);
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize internal buffers of the matrix_element object.");
        }
    }

};

}   //namespace ttns

#endif  //TTNS_MATRIX_ELEMENT_HPP//

