#ifndef UTILS_ITERATIVE_METHODS_PRECONDITIONERS_HPP
#define UTILS_ITERATIVE_METHODS_PRECONDITIONERS_HPP


namespace utils
{

namespace preconditioner
{

template <typename T, typename backend>
class preconditioner{};

template <typename T, typename backend>
class identity : public preconditioner<T, backend>
{
public: 
    identity(){}
    identity(const identity& o) = default;
    identity(identity&& o) = default;

    identity& operator=(const identity& o) = default;
    identity& operator=(identity&& o) = default;

    template <typename Vin>
    void apply(Vin&){}
    void clear(){}
    void initialise(){}
};

}   //namespace preconditioner
}   //namespace utils

#endif

