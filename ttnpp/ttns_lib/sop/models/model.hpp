#ifndef TTNS_SOP_MODELS_HPP
#define TTNS_SOP_MODELS_HPP

#include "../sSOP.hpp"
#include "../SOP.hpp"
#include "../system_information.hpp"

namespace ttns
{

//a interface class for handling generic models.  The model classes are designed for offloading
//the work associated with setting up the Hamiltonian and its parameters from the user. 
template <typename T> 
class model
{
public:
    using real_type = typename linalg::get_real_type<T>::type;
public:
    model(){}
    virtual ~model(){}

    virtual SOP<T> hamiltonian(real_type tol = 1e-14)
    {
        SOP<T> sop;
        this->hamiltonian(sop, tol);
        return sop;
    }
    virtual system_modes system_info()
    {
        system_modes inf;
        this->system_info(inf);
        return inf;
    }

    virtual void hamiltonian(sSOP<T>& H, real_type tol = 1e-14) = 0;
    virtual void hamiltonian(SOP<T>& H, real_type tol = 1e-14) = 0;
    virtual void system_info(system_modes& sysinf) = 0;
};
}

#endif

