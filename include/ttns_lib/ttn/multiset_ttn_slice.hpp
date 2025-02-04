#ifndef MULTISET_TTN_SLICE_HPP
#define MULTISET_TTN_SLICE_HPP

#include <common/tmp_funcs.hpp>
#include <common/exception_handling.hpp>

namespace ttns
{

template <typename T, typename backend, bool CONST>
struct multiset_slice_storage_type;

template <typename T, typename backend>
struct multiset_slice_storage_type<T, backend, true>
{
public:
    using size_type = typename backend::size_type;
    using type = const ms_ttn<T, backend>&;
    multiset_slice_storage_type(type obj) : m_obj(obj){}

    template <typename U, typename be>
    void set_slice(const ttn<U, be>& o, size_type sind)
    {
        RAISE_EXCEPTION("Cannot assign const multiset storage slice.");
    }

    template <typename U, typename be, bool C2>
    void set_slice(const multiset_ttn_slice<U, be, C2>& o, size_type sind, size_type sind2)
    {
        RAISE_EXCEPTION("Cannot assign const multiset storage slice.");
    }

    type obj() {return m_obj;}
    type obj() const {return m_obj;}
protected:
    type m_obj;
};

template <typename T, typename backend>
struct multiset_slice_storage_type<T, backend, false>
{
public:
    using size_type = typename backend::size_type;
    using type = ms_ttn<T, backend>&;
    multiset_slice_storage_type(type obj) : m_obj(obj){}

    template <typename U, typename be>
    void set_slice(size_type sind, const ttn<U, be>& o)
    {
        CALL_AND_RETHROW(m_obj.set_slice(sind, o));
    }

    template <typename U, typename be, bool C2>
    void set_slice(size_type sind, const multiset_ttn_slice<U, be, C2>& o)
    {
        CALL_AND_RETHROW(m_obj.set_slice(sind, o));
    }

    type obj(){return m_obj;}
    const ms_ttn<T, backend>& obj() const {return m_obj;}
protected:
    type m_obj;
};


template <typename T, typename backend, bool CONST>
class multiset_ttn_slice : multiset_slice_storage_type<T, backend, CONST>
{
public:
    using size_type = typename backend::size_type;
    using base_type = multiset_slice_storage_type<T, backend, CONST>;
    using obj_type = typename base_type::type;

    using base_type::obj;

public:
    multiset_ttn_slice() = delete;
    multiset_ttn_slice(obj_type _obj, size_type slice_index) : base_type(_obj), m_slice_index(slice_index){}
    multiset_ttn_slice(const multiset_ttn_slice& o) = default;
    multiset_ttn_slice(multiset_ttn_slice&& o) = default;

    multiset_ttn_slice& operator=(const multiset_ttn_slice& o)
    {
        base_type::set_slice(m_slice_index, o);
        return *this;
    }

    template <typename U, typename be, bool enabled = CONST>
    multiset_ttn_slice& operator=(const ttn<U, be>& o)
    {
        base_type::set_slice(m_slice_index, o);
        return *this;
    }

    template <typename U, typename be, bool const2>
    typename std::enable_if<not std::is_same<be, backend>::value or not std::is_same<U, T>::value or const2 == CONST, multiset_ttn_slice&>::type operator=(const multiset_ttn_slice<U, be, const2>& o)
    {
        base_type::set_slice(m_slice_index, o);
        return *this;
    }

    size_type nset() const{return 1;}

    size_type& slice_index(){return m_slice_index;}
    const size_type& slice_index() const{return m_slice_index;}


protected:
    size_type m_slice_index;

};

template <typename U, typename T, typename backend, bool CONST, typename = typename std::enable_if<is_tree<U>::value, void>::type>
static inline bool 
has_same_structure(const T& t, const multiset_ttn_slice<T, backend, CONST>& u)
{
    return has_same_structure(t, u.obj());
}

template <typename U, typename T, typename backend, bool CONST, typename = typename std::enable_if<is_tree<U>::value, void>::type>
static inline bool 
has_same_structure(const multiset_ttn_slice<T, backend, CONST>& u, const T& t)
{
    return has_same_structure(t, u.obj());
}
}   //namespace ttns

#endif


