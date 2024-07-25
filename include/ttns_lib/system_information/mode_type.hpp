#ifndef TTNS_LIB_MODE_TYPE_HPP
#define TTNS_LIB_MODE_TYPE_HPP

#include <common/exception_handling.hpp>

namespace ttns
{

namespace system
{

class mode
{
public:
    mode() : m_dimen(1), m_is_fermionic(false) {}
    mode(size_t dimen) : m_dimen(dimen), m_is_fermionic(false){}

    virtual ~mode(){}

    size_t dimen() const{return m_dimen;}
    size_t& dimen(){return m_dimen;}

    bool is_fermionic() const{return m_is_fermionic;}

protected:
    size_t m_dimen;
    bool m_is_fermionic;
};





}   //namespace system
}   //namespace ttns

#endif

