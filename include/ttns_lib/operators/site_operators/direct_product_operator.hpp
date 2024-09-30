#ifndef TTNS_OPERATORS_DIRECT_PRODUCT_OPERATORS_HPP
#define TTNS_OPERATORS_DIRECT_PRODUCT_OPERATORS_HPP

#include <linalg/linalg.hpp>
#include "primitive_operator.hpp"

namespace ttns
{
namespace ops
{

//need to implement the kronecker product operator object
template <typename T, typename backend = linalg::blas_backend> 
class direct_product_operator : public primitive<T, backend>
{
public:
    using base_type = primitive<T, backend>;

    //use the parent class type aliases
    using typename base_type::matrix_type;
    using typename base_type::size_type;
    using typename base_type::matrix_ref;
    using typename base_type::const_matrix_ref;
    using typename base_type::real_type;
    using typename base_type::vector_type;
    using typename base_type::vector_ref;
    using typename base_type::const_vector_ref;
    using typename base_type::matview;
    using typename base_type::resview;
    using typename base_type::tensview;
    using typename base_type::restensview;

public:
    direct_product_operator()  : base_type() {}

    direct_product_operator(const std::vector<std::shared_ptr<base_type>>& ops) try : base_type()   
    {
        CALL_AND_RETHROW(initialise(ops));
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct dense matrix operator object.");
    }

    template <typename ... Args>
    direct_product_operator(const std::shared_ptr<base_type>& m, Args&& ... args) try : base_type()
    {
        m_operators.resize(sizeof...(args)+1);
        set_operators(0, m, std::forward<Args>(args)...);
        size_type size = 1;
        for(size_type i=0; i < m_operators.size(); ++i)
        {
            size *= m_operators[i]->size();
        }
        base_type::m_size = size;
        ASSERT(m_operators.size() > 0, "Invliad operator object.");
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct dense matrix operator object.");
    }

    void initialise(const std::vector<std::shared_ptr<base_type>>& ops)
    {
        m_operators=ops;        
        size_type size = 1;
        for(size_type i=0; i < m_operators.size(); ++i)
        {
            size *= m_operators[i]->size();
        }
        base_type::m_size = size;
        ASSERT(m_operators.size() > 0, "Invliad operator object.");
    }

    direct_product_operator(const direct_product_operator& o) = default;
    direct_product_operator(direct_product_operator&& o) = default;

    direct_product_operator& operator=(const direct_product_operator& o) = default;
    direct_product_operator& operator=(direct_product_operator&& o) = default;

    bool is_resizable() const final{return false;}
    void resize(size_type /*n*/){ASSERT(false, "This shouldn't be called.");}

    void update(real_type /*t*/, real_type /*dt*/) final{}  
    std::shared_ptr<base_type> clone() const{return std::make_shared<direct_product_operator>(m_operators);}

    std::shared_ptr<base_type> transpose() const 
    {
        std::vector<std::shared_ptr<base_type>> operators;  operators.resize(m_operators.size());
        for(size_t i = 0; i < m_operators.size(); ++i)
        {
            operators.push_back(m_operators[i]->transpose());
        }
        return std::make_shared<direct_product_operator>(operators);
    }

    void apply(const resview& A, resview& HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}  
    void apply(const resview& A, resview& HA, real_type /* t */, real_type /* dt */) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}  
    void apply(const matview& A, resview& HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));} 
    void apply(const matview& A, resview& HA, real_type /* t */, real_type /* dt */) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
  
    void apply(const_matrix_ref A, matrix_ref HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
    void apply(const_matrix_ref A, matrix_ref HA, real_type /*t*/, real_type /*dt*/) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
    void apply(const_vector_ref A, vector_ref HA) final{CALL_AND_RETHROW(apply_rank_1(A, HA));}
    void apply(const_vector_ref A, vector_ref HA, real_type /*t*/, real_type /*dt*/) final{CALL_AND_RETHROW(apply_rank_1(A, HA));}

    //functions for applying the operator to rank 3 tensor views
    void apply(const restensview& A, restensview& HA) final {CALL_AND_RETHROW(apply_rank_3(A, HA));}
    void apply(const restensview& A, restensview& HA, real_type /* t */, real_type /* dt */) final  {CALL_AND_RETHROW(apply_rank_3(A, HA));}
    void apply(const tensview& A, restensview& HA) final {CALL_AND_RETHROW(apply_rank_3(A, HA));}
    void apply(const tensview& A, restensview& HA, real_type /* t */, real_type /* dt */) final {CALL_AND_RETHROW(apply_rank_3(A, HA));}

protected:
    template <typename T1, typename T2>
    void apply_rank_1(const T1& A, T2& HA)
    {
        CALL_AND_HANDLE(m_temp.resize(A.size()), "Failed to resize temporary array object.");
        ASSERT(m_operators.size() > 0, "Invliad operator object.");
        
        bool HA_set = apply_internal(A, HA);
        if(!HA_set){CALL_AND_HANDLE(HA = m_temp, "Failed to copy temp array.");}
    }

    template <typename T1, typename T2>
    void apply_rank_2(const T1& A, T2& HA)
    {
        CALL_AND_HANDLE(m_temp.resize(A.size()), "Failed to resize temporary array object.");
        ASSERT(m_operators.size() > 0, "Invliad operator object.");
        
        bool HA_set = apply_internal(A, HA);
        if(!HA_set){CALL_AND_HANDLE(HA = m_temp.reinterpret_shape(HA.shape(0), HA.shape(1)), "Failed to copy temp array.");}
    }

    template <typename T1, typename T2>
    void apply_rank_3(const T1& A, T2& HA)
    {
        CALL_AND_HANDLE(m_temp.resize(A.size()), "Failed to resize temporary array object.");
        ASSERT(m_operators.size() > 0, "Invliad operator object.");
        
        bool HA_set = apply_internal(A, HA, A.shape(0));
        if(!HA_set){CALL_AND_HANDLE(HA = m_temp.reinterpret_shape(HA.shape(0), HA.shape(1), HA.shape(2)), "Failed to copy temp array.");}

    }

    template <typename T1, typename T2>
    bool apply_internal(const T1& A, T2& HA, size_t d1=1)
    {
        std::array<size_type, 3> mdims = {{d1,1,A.size()/d1}};
        
        bool HA_set = false;
        bool applied = false;
        for(size_type i = 0; i < m_operators.size(); ++i)
        {
            mdims[0] *= mdims[1];
            mdims[1] = m_operators[i]->size();
            mdims[2] /= mdims[1];

            auto At = A.reinterpret_shape(mdims[0], mdims[1], mdims[2]);
            auto HAt = HA.reinterpret_shape(mdims[0], mdims[1], mdims[2]);
            auto Tt = m_temp.reinterpret_shape(mdims[0], mdims[1], mdims[2]);

            if(!m_operators[i]->is_identity())
            {
                if(!applied)
                {
                    CALL_AND_HANDLE(m_operators[i]->apply(At, HAt), "Failed to compute kronecker product contraction."); 
                    HA_set = true;
                    applied = true;
                }
                else
                {
                    if(HA_set)
                    {
                        CALL_AND_HANDLE(m_operators[i]->apply(HAt, Tt), "Failed to compute kronecker product contraction."); 
                        HA_set = false;
                    }
                    else
                    {
                        CALL_AND_HANDLE(m_operators[i]->apply(Tt, HAt), "Failed to compute kronecker product contraction."); 
                        HA_set = true;
                    }
                }
            }
        }
        if(!applied)
        {
            HA = A;
            HA_set = true;
        }
        return HA_set;
    }


public:    
    std::string to_string() const final
    {
        std::stringstream oss;
        oss << "direct product operator: " << std::endl;
        for(size_t i = 0; i < m_operators.size(); ++i)
        {
            oss << m_operators[i]->to_string() << std::endl;
        }
        return oss.str();
    }



protected:
    std::vector<std::shared_ptr<base_type>> m_operators;
    vector_type m_temp;

protected:

    void set_operators(size_t i, const std::shared_ptr<base_type>& m)
    {
        m_operators[i] = m;
    }

    template <typename ... Args>
    void set_operators(size_t i, const std::shared_ptr<base_type>& m, Args&& ... args)
    {
        m_operators[i] = m;
        set_operators(i+1, std::forward<Args>(args)...);
    }

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::base_class<primitive<T, backend> >(this)), "Failed to serialise direct_product operator object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("matrix", m_operators)), "Failed to serialise direct_product operator object.  Error when serialising the matrix.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("temp", m_temp)), "Failed to serialise direct_product operator object.  Error when serialising the matrix.");
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::base_class<primitive<T, backend> >(this)), "Failed to serialise direct_product operator object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("matrix", m_operators)), "Failed to serialise direct_product operator object.  Error when serialising the matrix.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("temp", m_temp)), "Failed to serialise direct_product operator object.  Error when serialising the matrix.");
    }
#endif
};


}
}

#ifdef CEREAL_LIBRARY_FOUND
TTNS_REGISTER_SERIALIZATION(ttns::ops::direct_product_operator, ttns::ops::primitive)
#endif


#endif  //TTNS_OPERATORS_DIRECT_PRODUCT_OPERATOR_HPP
