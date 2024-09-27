#ifndef TTNS_OPERATORS_SITE_OPERATORS_HPP
#define TTNS_OPERATORS_SITE_OPERATORS_HPP

#include <complex>
#include <string>
#include <memory>

#include <linalg/linalg.hpp>
#include <common/tmp_funcs.hpp>

#include "primitive_operator.hpp"

#include "../../sop/sSOP.hpp"
#include "../../sop/system_information.hpp"
#include "../../sop/operator_dictionaries/operator_dictionary.hpp"

namespace ttns
{

/* 
 * A class for handling individual site operators. Here we use type erasure to construct a type that is easier to work with for the 
 * python side of the code. 
 * */
template <typename T, typename backend> 
class site_operator
{
public:
    using vector_type = linalg::vector<T, backend>;
    using matrix_type = linalg::matrix<T, backend>;
    using size_type = typename backend::size_type;
    using matrix_ref = matrix_type&;
    using const_matrix_ref = const matrix_type&;
    using vector_ref = vector_type&;
    using const_vector_ref = const vector_type&;
    using real_type = typename tmp::get_real_type<T>::type;

protected:
    mutable std::shared_ptr<ops::primitive<T, backend>> m_op;
    size_type m_mode;

public:
    site_operator() : m_op(nullptr) {}

    site_operator(std::shared_ptr<ops::primitive<T, backend>> op) : m_op(op->clone()), m_mode(0) {}
    site_operator(std::shared_ptr<ops::primitive<T, backend>> op, size_type mode) : m_op(op->clone()), m_mode(mode) {}

    template <typename OpType, typename = typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, void>::type>
    site_operator(const OpType& op) : m_op(std::make_shared<OpType>(op)), m_mode(0) {}

    template <typename OpType, typename = typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, void>::type>
    site_operator(OpType&& op) : m_op(std::make_shared<OpType>(std::move(op))), m_mode(0) {}

    template <typename OpType, typename = typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, void>::type>
    site_operator(const OpType& op, size_type mode) : m_op(std::make_shared<OpType>(op)), m_mode(mode) {}

    template <typename OpType, typename = typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, void>::type>
    site_operator(OpType&& op, size_type mode) : m_op(std::make_shared<OpType>(std::move(op))), m_mode(mode) {}

    site_operator(sOP& sop, const system_modes& sys, bool use_sparse = true)
    {
        CALL_AND_HANDLE(initialise(sop, sys, use_sparse), "Failed to construct sop operator.");
    }
    site_operator(sOP& sop, const system_modes& sys, const operator_dictionary<T, backend>& opdict, bool use_sparse = true)
    {
        CALL_AND_HANDLE(initialise(sop, sys, opdict, use_sparse), "Failed to construct sop operator.");
    }

    site_operator(const site_operator& o) = default;
    site_operator(site_operator&& o) = default;
    ~site_operator() {}

    site_operator& operator=(const site_operator& o) = default;
    site_operator& operator=(site_operator&& o) = default;

    //resize this object from a tree structure, a SOP object, a system info class and an optional operator dictionary.
    //This implementation does not support composite modes currently.  To do add mode combination
    void initialise(sOP& sop, const system_modes& sys, bool use_sparse = true)
    {
        //get the primitive mode index associated with this sop term
        size_type nu = sop.mode();
        std::pair<size_t, size_t> mode_info = sys.primitive_mode_index(nu);
        size_t mode = std::get<0>(mode_info);
        size_t lmode = std::get<1>(mode_info);

        //now get the composite mode index associated with this primitive mode
        m_mode = sys.mode_index(mode);

        std::vector<size_t> hilbert_space_dimension(sys[mode].nmodes());
        for(size_t lmi = 0; lmi < sys[mode].nmodes(); ++lmi)
        {
            hilbert_space_dimension[lmi] = sys[mode][lmi].lhd();
        }
        std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(hilbert_space_dimension);

        std::string label = sop.op();
        using opdictype = operator_from_default_dictionaries<T, backend>;

        CALL_AND_HANDLE(m_op = opdictype::query(label, basis, sys.primitive_mode(nu).type(), use_sparse, lmode), "Failed to insert new element in mode operator.");
        ASSERT(m_op != nullptr, "Failed to construct site operator object.");
    }

    void initialise(sOP& sop, const system_modes& sys, const operator_dictionary<T, backend>& opdict, bool use_sparse = true)
    {
        //get the primitive mode index associated with this sop term
        size_type nu = sop.mode();
        std::pair<size_t, size_t> mode_info = sys.primitive_mode_index(nu);
        size_t mode = std::get<0>(mode_info);
        size_t lmode = std::get<1>(mode_info);

        //now get the composite mode index associated with this primitive mode
        m_mode = sys.mode_index(mode);

        std::vector<size_t> hilbert_space_dimension(sys[mode].nmodes());
        for(size_t lmi = 0; lmi < sys[mode].nmodes(); ++lmi)
        {
            hilbert_space_dimension[lmi] = sys[mode][lmi].lhd();
        }
        std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(hilbert_space_dimension);

        using opdictype = operator_from_default_dictionaries<T, backend>;
        std::string label = sop.op();

        bool opbound = false;
        if(nu < opdict.nmodes())
        {
            //first start to access element from opdict
            std::shared_ptr<ops::primitive<T, backend>> op = opdict.query(nu, label);

            if(op != nullptr)
            {
                ASSERT(op->size() == sys[mode].lhd(), "Invalid operator size in default operator dictionary.");
                m_op = op;
                opbound = true;
            }
        }

        if(!opbound)
        {
            CALL_AND_HANDLE(m_op = opdictype::query(label, basis, sys.primitive_mode(nu).type(), use_sparse, lmode), "Failed to insert new element in mode operator.");
            ASSERT(m_op != nullptr, "Failed to construct site operator object.");
        }

    }



    template <typename OpType>
    typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, site_operator&>::type 
    operator=(const OpType& op)
    {
        m_op = std::make_shared<OpType>(op);
        return *this;
    }

    template <typename OpType>
    typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, site_operator&>::type 
    operator=(OpType&& op) 
    {
        m_op = std::make_shared<OpType>(std::move(op));
        return *this;
    }

    site_operator& operator=(std::shared_ptr<ops::primitive<T, backend>> op)
    {
        m_op = op->clone();
        return *this;
    }

    template <typename OpType>
    typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, site_operator&>::type 
    bind(const OpType& op)
    {
        m_op = std::make_shared<OpType>(op);
        return *this;
    }

    template <typename OpType>
    typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, site_operator&>::type 
    bind(OpType&& op) 
    {
        m_op = std::make_shared<OpType>(std::move(op));
        return *this;
    }

    site_operator& bind(std::shared_ptr<ops::primitive<T, backend>> op)
    {
        m_op = op->clone();
        return *this;
    }

    template <typename Atype, typename HAtype>
    void apply(const Atype& A, HAtype& HA)
    {
        ASSERT(m_op != nullptr, "Cannot apply empty site operator.");
        CALL_AND_RETHROW(m_op->apply(A, HA));
    }

    template <typename Atype, typename HAtype> 
    void apply(const Atype& A, HAtype& HA, real_type t, real_type dt)
    {
        ASSERT(m_op != nullptr, "Cannot apply empty site operator.");
        CALL_AND_RETHROW(m_op->apply(A, HA, t, dt));
    }
    template <typename Atype, typename HAtype> 
    void apply(const Atype& A, HAtype& HA) const
    {
        ASSERT(m_op != nullptr, "Cannot apply empty site operator.");
        CALL_AND_RETHROW(m_op->apply(A, HA));
    }

    template <typename Atype, typename HAtype> 
    void apply(const Atype& A, HAtype& HA, real_type t, real_type dt) const
    {
        ASSERT(m_op != nullptr, "Cannot apply empty site operator.");
        CALL_AND_RETHROW(m_op->apply(A, HA, t, dt));
    }

    //function for allowing you to update time-dependent Hamiltonians
    void update(real_type t, real_type dt)
    {
        ASSERT(m_op != nullptr, "Cannot apply empty site operator.");
        CALL_AND_RETHROW(m_op->update(t, dt));
    }

    bool is_resizable() const
    {
        ASSERT(m_op != nullptr, "Cannot check if empty site operator is resizable.");
        return m_op->is_resizable();
    }
    void resize(size_type n)
    {
        ASSERT(m_op != nullptr, "Cannot resize empty site operator.");
        CALL_AND_RETHROW(m_op->resize(n));
    }
    std::string to_string() const
    {
        ASSERT(m_op != nullptr, "Cannot convert empty site operator to string.");
        return m_op->to_string();
    };

    size_type mode_dimension() const{return m_op->size();}
    size_type size() const{return m_op->size();}
    bool is_identity() const{return m_op->is_identity();}

    std::shared_ptr<ops::primitive<T, backend>> op(){return m_op;}
    std::shared_ptr<ops::primitive<T, backend>> op() const{return m_op->clone();}

    size_t mode() const {return m_mode;}
    size_t& mode() {return m_mode;}

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("op", m_op)), "Failed to primitive operator.  Failed to serialise its size.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("mode", m_mode)), "Failed to primitive operator.  Failed to serialise its size.");
    }

#endif
};


}   //namespace ttns




#endif  //TTNS_OPERATORS_PRIMITIVE_OPERATORS_HPP//

