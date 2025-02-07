#ifndef UTILS_OCCUPATION_NUMBER_INDEXING_HPP_
#define UTILS_OCCUPATION_NUMBER_INDEXING_HPP_

#include <vector>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <complex>
#include <unordered_set>
#include <tuple>

#include <linalg/linalg.hpp>
#include <common/exception_handling.hpp>

namespace utils
{
class occupation_number_basis
{
public:
    occupation_number_basis() : m_D(0), m_K(0), m_nm1(0), m_np1(0) {}
    virtual ~occupation_number_basis() {};

    virtual void clear() = 0;

    virtual void get_state(size_t i, std::vector<size_t>& res) const = 0;
    virtual size_t get_occupation(size_t i, size_t k) const = 0;
    virtual bool contains_state(const std::vector<size_t>& ind) const = 0;
    virtual size_t get_index(const std::vector<size_t>& ind) const = 0;


    virtual bool contains_raised_state(size_t i, size_t k) const = 0;
    virtual bool contains_lowered_state(size_t i, size_t k) const = 0;

    //we want to see if it has an element where k1 is lowered and k2 is raised
    virtual bool contains_adjacent_state(size_t i, size_t k1, size_t k2) const = 0;

    virtual size_t get_raised_index(size_t i, size_t k) const = 0;

    virtual size_t get_lowered_index(size_t i, size_t k) const = 0;

    virtual size_t get_adjacent_index(size_t i, size_t k1, size_t k2) const = 0;

    size_t nstates() const{return m_D;}
    size_t nmodes() const{return m_K;}
    size_t Nconnections_lower_states() const{return m_nm1;}
    size_t Nconnections_higher_states() const{return m_np1;}

    virtual size_t Nprods() const{return 1;}
    virtual bool is_direct_product() const{return false;}

    size_t dim(size_t i) const
    {
        ASSERT(i < m_K, "Index out of bounds.");
        return m_localdims[i];
    }
protected:
    std::vector<size_t> m_localdims;
    size_t m_D;
    size_t m_K;
    size_t m_nm1;
    size_t m_np1;
};
/*
 * A class for indexing a direct product occupation number basis
 * Similarly for a direct product occupation number basis all of the indexing is completely trivial.
 */

class direct_product_occupation_number_basis : public occupation_number_basis
{
    using bt = occupation_number_basis;
protected:
    std::vector<size_t> m_stride;        //an array storing the occupation numbers of a 

public:
    direct_product_occupation_number_basis() : bt() {}
    direct_product_occupation_number_basis(size_t L, size_t K) : bt()
    {
        initialise(L, K);
    }

    template < typename vect_type>
    direct_product_occupation_number_basis(const vect_type& N) : bt()
    {
        CALL_AND_HANDLE(initialise(N), "Failed to construct occupation number indexing object.");
    }   

    template <typename iter_type, typename = typename std::enable_if<!std::is_integral<iter_type>::value, void>::type>
    direct_product_occupation_number_basis(iter_type i1, iter_type i2) : bt()
    {
        CALL_AND_HANDLE(initialise(i1, i2), "Failed to construct occupation number indexing object.");
    }

    direct_product_occupation_number_basis(const direct_product_occupation_number_basis& o) = default;
    direct_product_occupation_number_basis(direct_product_occupation_number_basis&& o) = default;

    direct_product_occupation_number_basis& operator=(const direct_product_occupation_number_basis& o) = default;
    direct_product_occupation_number_basis& operator=(direct_product_occupation_number_basis&& o) = default;
    ~direct_product_occupation_number_basis(){}
    
    virtual void clear()
    {
        bt::m_localdims.clear();
        bt::m_D = 0;
        bt::m_K = 0;
    }

    virtual size_t Nprods() const{return m_K;}
    virtual bool is_direct_product() const{return true;}

    void initialise(size_t L, size_t K)
    {
        try
        {
            bt::m_K = K;
            bt::m_localdims.resize(K);
            m_stride.resize(K);

            for(size_t i = 0; i < K; ++i)
            {
                bt::m_localdims[i] = L;
            }

            setup_dim_and_stride();
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise occupation number indexing object.");
        }
    }

    template <typename vect_type>
    void initialise(const vect_type& N) 
    {
        try
        {
            ASSERT(N.size() != 0, "wk must have at least 1 element.");
            bt::m_K = N.size();
            bt::m_localdims.resize(bt::m_K);
            m_stride.resize(bt::m_K);

            for(size_t i = 0; i < bt::m_K; ++i)
            {
                bt::m_localdims[i] = N[i];
            }
            setup_dim_and_stride();
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise occupation number indexing object.");
        }
    }

    template <typename iter_type, typename = typename std::enable_if<!std::is_integral<iter_type>::value, void>::type>
    void initialise(iter_type i1, iter_type i2) 
    {
        try
        {
            std::vector<int64_t> wk(i1, i2);
            CALL_AND_RETHROW(initialise(wk));
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise occupation number indexing object.");
        }
    }


    virtual void get_state(size_t i, std::vector<size_t>& res) const
    {
        ASSERT(i < bt::m_D, "Unable to occupation occupation number by index.  Invalid index.");
        res.resize(bt::m_K);
        size_t counter = i;
        for(size_t k=0; k < bt::m_K; ++k)
        {
            res[k] = counter/m_stride[k];
            counter = counter - res[k]*m_stride[k];
        }
    }

    virtual size_t get_occupation(size_t i, size_t kind) const
    {
        ASSERT(i < bt::m_D, "Unable to occupation occupation number by index.  Invalid index.");
        ASSERT(kind < bt::m_K, "Unable to occupation occupation number by index.  Invalid index.");
        size_t resk=0;
        size_t counter = i;
        for(size_t k=0; k < bt::m_K; ++k)
        {
            resk = counter/m_stride[k];
            if(k == kind){return resk;}
            counter = counter - resk*m_stride[k];
        }
        RAISE_EXCEPTION("Something went exceptionally wrong in the get occupation call.");
    }

    virtual bool contains_state(const std::vector<size_t>& ind) const
    {
        for(size_t k=0; k < bt::m_K; ++k)
        {
            if(ind[k] >= bt::m_localdims[k]){return false;}
        }
        return true;
    }

    virtual size_t get_index(const std::vector<size_t>& ind) const
    {
        size_t res = 0;
        for(size_t k=0; k < bt::m_K; ++k)
        {
            res += m_stride[k]*ind[k];
        }
        return res;
    }

    virtual bool contains_raised_state(size_t i, size_t kind) const
    {
        ASSERT(i < bt::m_D, "Unable to occupation occupation number by index.  Invalid index.");
        ASSERT(kind < bt::m_K, "Unable to occupation occupation number by index.  Invalid index.");
        size_t resk=0;
        size_t counter = i;
        for(size_t k=0; k < bt::m_K; ++k)
        {
            resk = counter/m_stride[k];
            if(k == kind)
            {
                return (resk+1 != bt::m_localdims[kind]);
            }
            counter = counter - resk*m_stride[k];
        }
        return false;
    }

    virtual bool contains_lowered_state(size_t i, size_t kind) const
    {
        ASSERT(i < bt::m_D, "Unable to occupation occupation number by index.  Invalid index.");
        ASSERT(kind < bt::m_K, "Unable to occupation occupation number by index.  Invalid index.");
        size_t resk=0;
        size_t counter = i;
        for(size_t k=0; k < bt::m_K; ++k)
        {
            resk = counter/m_stride[k];
            if(k == kind)
            {
                return (resk != 0);
            }
            counter = counter - resk*m_stride[k];
        }
        return false;
    }

    //we want to see if it has an element where k1 is lowered and k2 is raised
    virtual bool contains_adjacent_state(size_t i, size_t k1, size_t k2) const
    {
        ASSERT(i < bt::m_D && k1 < bt::m_K && k2  < bt::m_K, "Unable to get index of raised term.  Index out of bounds.");

        //if k1 == k2 then the elements are the same and so this isn't an adjacent term
        if(k1 == k2){return false;}

        size_t resk=0;
        size_t counter = i;
        bool valid_state = true;
        for(size_t k=0; k < bt::m_K; ++k)
        {
            resk = counter/m_stride[k];
            if(k == k1)
            {
                valid_state = valid_state && (resk != 0);
            }
            else if (k == k2)
            {
                valid_state = valid_state && (resk+1 != bt::m_localdims[k]);
            }
            counter = counter - resk*m_stride[k];
        }
        return valid_state;
    }

    virtual size_t get_raised_index(size_t i, size_t kind) const
    {
        ASSERT(i < bt::m_D && kind < bt::m_K, "Unable to get index of raised term.  Index out of bounds.");

        size_t resk=0;
        size_t counter = i;
        size_t index = 0;
        for(size_t k=0; k < bt::m_K; ++k)
        {
            resk = counter/m_stride[k];
            if(k == kind)
            {
                ASSERT(resk + 1 != bt::m_localdims[k], "Raised index will be out of bounds.");
                index += (resk+1)*m_stride[k];
            }
            else
            {
                index += resk*m_stride[k];
            }
            counter = counter - resk*m_stride[k];
        }
        return index;
    }
    
    virtual size_t get_lowered_index(size_t i, size_t kind) const
    {
        ASSERT(i < bt::m_D && kind < bt::m_K, "Unable to get index of raised term.  Index out of bounds.");

        size_t resk=0;
        size_t counter = i;
        size_t index = 0;
        for(size_t k=0; k < bt::m_K; ++k)
        {
            resk = counter/m_stride[k];
            if(k == kind)
            {
                ASSERT(resk != 0, "Lowered index will be out of bounds.");
                index += (resk-1)*m_stride[k];
            }
            else
            {
                index += resk*m_stride[k];
            }
            counter = counter - resk*m_stride[k];
        }
        return index;
    }

    virtual size_t get_adjacent_index(size_t i, size_t k1, size_t k2) const
    {
        ASSERT(i < bt::m_D && k1 < bt::m_K && k2  < bt::m_K, "Unable to get index of adjacent term.  Index out of bounds.");

        ASSERT(k1 != k2, "Unable to get index of adjacent term.  If k1 == k2 then this is not an adjacent term.");

        size_t resk=0;
        size_t counter = i;
        size_t index = 0;
        for(size_t k=0; k < bt::m_K; ++k)
        {
            resk = counter/m_stride[k];
            if(k == k1)
            {
                ASSERT(resk != 0, "Failed to get index of adjacent term.  Index out of bounds.");
                index += (resk-1)*m_stride[k];
            }
            else if (k == k2)
            {
                ASSERT(resk+1 != bt::m_localdims[k], "Failed to get index of adjacent term.  Index out of bounds.");
                index += (resk+1)*m_stride[k];
            }
            else
            {
                index += resk*m_stride[k];
            }
            counter = counter - resk*m_stride[k];
        }
        return index;
    }

protected:
    void setup_dim_and_stride()
    {   
        size_t counter = bt::m_K;
        m_stride[counter-1] = 1;

        for(size_t k = 1; k < bt::m_K; ++k)
        {
            --counter;
            m_stride[counter-1] = bt::m_localdims[counter]*m_stride[counter];
            bt::m_D *= bt::m_localdims[counter-1];
        }

        bt::m_D = 1;
        for(size_t k = 0; k < bt::m_K; ++k)
        {
            bt::m_D *= bt::m_localdims[k];
        }
    }
};


/*
 * A class for indexing a basis of occupation number states truncated at either a maximum energy scale or maximum 
 * number of excitation.    This class contains a set of function uses a set of objects to explicitly store the number
 * of excitations.  This is the only occupation_number_basis object that supports energy based truncation as in general
 * the energy based truncation leads to non-trivial indexing.  For level based truncation it is possible to use the 
 * indexing based on pascal simplices to evaluate the indexing without storing any intermediate object.
 *
 * Similarly for a direct product occupation number basis all of the indexing is completely trivial.
 */
class truncated_occupation_number_basis : public occupation_number_basis
{
    using bt = occupation_number_basis;
protected:
    std::vector<std::vector<size_t>> m_nk;        //an array storing the occupation numbers of a 
    std::vector<std::vector<int64_t>> m_nkp1;     //stores 
    std::vector<std::vector<int64_t>> m_nkm1;

public:
    truncated_occupation_number_basis() : bt() {}
    truncated_occupation_number_basis(size_t L, size_t K) : truncated_occupation_number_basis()
    {
        initialise(L, K);
    }

    template <typename T, typename vect_type, typename ... SF>
    truncated_occupation_number_basis(const vect_type& wk, T wmax, SF&&... sf) : truncated_occupation_number_basis()
    {
        CALL_AND_HANDLE(initialise(wk, wmax, std::forward<SF>(sf)...), "Failed to construct occupation number indexing object.");
    }   

    template <typename T, typename iter_type, typename ... SF>
    truncated_occupation_number_basis(iter_type i1, iter_type i2, T wmax, SF&&... sf) : truncated_occupation_number_basis()
    {
        CALL_AND_HANDLE(initialise(i1, i2, wmax, std::forward<SF>(sf)...), "Failed to construct occupation number indexing object.");
    }

    truncated_occupation_number_basis(const truncated_occupation_number_basis& o) = default;
    truncated_occupation_number_basis(truncated_occupation_number_basis&& o) = default;

    truncated_occupation_number_basis& operator=(const truncated_occupation_number_basis& o) = default;
    truncated_occupation_number_basis& operator=(truncated_occupation_number_basis&& o) = default;

    ~truncated_occupation_number_basis(){}
    
    void clear()
    {
        m_nk.clear();
        m_nkp1.clear();
        m_nkm1.clear();
        bt::m_localdims.clear();
        bt::m_D = 0;
        bt::m_K = 0;
        bt::m_np1 = 0;
        bt::m_nm1 = 0;
    }

    void initialise(size_t L, size_t K)
    {
        try
        {
            bt::m_K = K;
            std::vector<size_t> nk(K, 0);
            bt::m_localdims.resize(K);  std::fill(bt::m_localdims.begin(), bt::m_localdims.end(), L);
            bt::m_D = pascals_simplex(L, K);
            m_nk.resize(bt::m_D, nk);

            //now we set up the indices
            size_t j = 0;
            size_t k = K - (j+1);
            
            for(size_t i = 1; i < bt::m_D; ++i)
            {
                bool state_found = false;
                while(!state_found)
                {
                    k = K - (j+1);
                    ++nk[k];

                    size_t Ltot = 0;
                    for(size_t k2=0; k2 < K; ++k2)
                    {
                        Ltot += nk[k2];   
                    }

                    if(Ltot <= L)
                    {
                        j = 0;
                        state_found = true;
                    }
                    else
                    {
                        ASSERT(k!=0, "Something went wrong when setting up indices.");
                        nk[k] = 0;
                        ++j;
                    }
                }
                m_nk[i] = nk;
            }

            //and now set up the raising and lowering connections
            CALL_AND_HANDLE(setup_connections(), "Failed to setup connections between states.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise occupation number indexing object.");
        }
    }

    template <typename T, typename vect_type, typename ... SF>
    typename std::enable_if<!std::is_integral<vect_type>::value, void>::type initialise(const vect_type& wk, T wmax, SF&& ... sf) 
    {
        try
        {
            ASSERT(wk.size() != 0, "wk must have at least 1 element.");
            bt::m_K = wk.size();
            bt::m_localdims.resize(bt::m_K);

            //create an object with sf(wk) sorted in ascending order
            std::vector<T> wk_p(bt::m_K);
            for(size_t i=0; i < bt::m_K; ++i)
            {
                wk_p[i] = std::abs(apply_sf(wk[i], std::forward<SF>(sf)...));
                bt::m_localdims[i] = 0; 
                while(wk_p[i]*bt::m_localdims[i] <= wmax)
                {
                    ++bt::m_localdims[i];
                }
            }   
            
            //determine the number of states that have energy equal to or less than wmax
            size_t counter = 0;     //we always have the zero occupancy state
            bool fully_indexed = false;
            std::vector<size_t> nk(bt::m_K, 0);

            size_t j = 0;
            size_t k = bt::m_K - (j+1);
            while(!fully_indexed)
            {
                bool state_found = false;
                while(!state_found)
                {
                    k = bt::m_K - (j+1);
                    ++nk[k];

                    T wktot = 0;
                    for(size_t j2=0; j2 < bt::m_K; ++j2)
                    {
                        wktot += nk[j2] * wk_p[j2];
                    }

                    if(wktot <= wmax)
                    {
                        state_found = true;
                        j=0;
                    }
                    else if(k!=0)
                    {
                        nk[k] = 0;
                        ++j;
                    }
                    else
                    {
                        state_found = true; 
                        fully_indexed = true;
                    }
                }
                ++counter;
            }


            //now we initialise the m_nk array
            bt::m_D = counter;
            std::fill(nk.begin(), nk.end(), 0);
            m_nk.resize(bt::m_D, nk);

            j = 0;
            k = bt::m_K - (j+1);

            for(size_t i = 1; i < bt::m_D; ++i)
            {
                bool state_found = false;
                while(!state_found)
                {
                    k = bt::m_K - (j+1);
                    ++nk[k];

                    T wktot = 0;
                    for(size_t j2=0; j2 < bt::m_K; ++j2)
                    {
                        wktot += nk[j2] * wk_p[j2];
                    }

                    if(wktot <= wmax)
                    {
                        state_found = true;
                        j=0;
                    }
                    else
                    {
                        ASSERT(k!=0, "Something went wrong.");
                        nk[k] = 0;
                        ++j;
                    }
                }
                m_nk[i] = nk;
            }

            //and now set up the raising and lowering connections
            CALL_AND_HANDLE(setup_connections(), "Failed to setup connections between states.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise occupation number indexing object.");
        }
    }

    template <typename T, typename iter_type, typename ... SF>
    typename std::enable_if<!std::is_integral<iter_type>::value, void>::type initialise(iter_type i1, iter_type i2, T wmax, SF&&... sf) 
    {
        try
        {
            std::vector<T> wk(i1, i2);
            CALL_AND_RETHROW(initialise(wk, wmax, std::forward<SF>(sf)...));
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise occupation number indexing object.");
        }
    }

    virtual void get_state(size_t i, std::vector<size_t>& res) const
    {
        ASSERT(i < m_nk.size(), "Unable to occupation occupation number by index.  Invalid index.");
        res = m_nk[i];
    }

    virtual size_t get_occupation(size_t i, size_t k) const
    {
        ASSERT(i < m_nk.size(), "Unable to occupation occupation number by index.  Invalid index.");
        ASSERT(k < bt::m_K, "Unable to occupation occupation number by index.  Invalid index.");
        return m_nk[i][k];
    }

    virtual bool contains_state(const std::vector<size_t>& ind) const
    {
        ASSERT(ind.size() == bt::m_K, "Unable to get index of element.  Index out of bounds.");
        auto itr = std::lower_bound(m_nk.begin(), m_nk.end(), ind, 
                       [](const std::vector<size_t>& a, const std::vector<size_t>& b)
                       {
                           for(size_t ii=0; ii<a.size(); ++ii)
                           {
                               if(a[ii] < b[ii]){return true;}
                               else if(a[ii] > b[ii]){return false;}
                           }
                           return false;
                       }
                   );
        if(itr == m_nk.end()){return false;}
        return true;
    }

    virtual size_t get_index(const std::vector<size_t>& ind) const
    {
        ASSERT(ind.size() == bt::m_K, "Unable to get index of element.  Index out of bounds.");
        auto itr = std::lower_bound(m_nk.begin(), m_nk.end(), ind, 
                       [](const std::vector<size_t>& a, const std::vector<size_t>& b)
                       {
                           for(size_t ii=0; ii<a.size(); ++ii)
                           {
                               if(a[ii] < b[ii]){return true;}
                               else if(a[ii] > b[ii]){return false;}
                           }
                           return false;
                       }
                   );
        if(itr == m_nk.end()){RAISE_EXCEPTION("Occupation number sequence not found.");}
        else if(*itr != ind){RAISE_EXCEPTION("Occupation number sequence not found.");}
        else{return std::distance(m_nk.begin(), itr);}       
    }

    virtual bool contains_raised_state(size_t i, size_t k) const
    {
        ASSERT(i < m_nk.size() && k < bt::m_K, "Unable to get index of raised term.  Index out of bounds.");
        return (m_nkp1[i][k] != -1);
    }

    virtual bool contains_lowered_state(size_t i, size_t k) const
    {
        ASSERT(i < m_nk.size() && k < bt::m_K, "Unable to get index of raised term.  Index out of bounds.");
        return (m_nkm1[i][k] != -1);
    }

    //we want to see if it has an element where k1 is lowered and k2 is raised
    virtual bool contains_adjacent_state(size_t i, size_t k1, size_t k2) const
    {
        ASSERT(i < m_nk.size() && k1 < bt::m_K && k2  < bt::m_K, "Unable to get index of raised term.  Index out of bounds.");

        //if k1 == k2 then the elements are the same and so this isn't an adjacent term
        if(k1 == k2){return false;}

        //if the term where we lower k1 isn't in the hierarchy then the adjacent term also isn't 
        if(!contains_lowered_state(i, k1)){return false;}

        size_t lowered_index = get_lowered_index(i, k1);
    
        return contains_raised_state(lowered_index, k2);
    }

    virtual size_t get_raised_index(size_t i, size_t k) const
    {
        ASSERT(i < m_nk.size() && k < bt::m_K, "Unable to get index of raised term.  Index out of bounds.");
        ASSERT(m_nkp1[i][k] != -1, "Unable to get index of raised term.  Term leaves the space represented by this object.");
        return static_cast<size_t>(m_nkp1[i][k]);
    }
    
    virtual size_t get_lowered_index(size_t i, size_t k) const
    {
        ASSERT(i < m_nk.size() && k < bt::m_K, "Unable to get index of lowered term.  Index out of bounds.");
        ASSERT(m_nkm1[i][k] != -1, "Unable to get index of lowered term.  Term leaves the space represented by this object.");
        return static_cast<size_t>(m_nkm1[i][k]);
    }

    virtual size_t get_adjacent_index(size_t i, size_t k1, size_t k2) const
    {
        ASSERT(i < m_nk.size() && k1 < bt::m_K && k2  < bt::m_K, "Unable to get index of adjacent term.  Index out of bounds.");

        ASSERT(k1 != k2, "Unable to get index of adjacent term.  If k1 == k2 then this is not an adjacent term.");
        ASSERT(contains_lowered_state(i, k1), "Unable to get index of adjacent term.  Element is not in the hierarchy failed to lower k1.");
        size_t lowered_index = get_lowered_index(i, k1);

        ASSERT(contains_raised_state(lowered_index, k2), "Unable to get index of adjacent term.  The adjacent term is not a member of the hierarchy.");
        return get_raised_index(lowered_index, k2);
    }


public:
    static inline size_t pascals_simplex(size_t L, size_t D)
    {
        size_t N = 1;
        for(size_t i = 1; i <= D; ++i)
        {
            N = (N*(L+i))/i;
        }
        return N;
    }

protected:
    void setup_connections()
    {
        {
            std::vector<int64_t> initialisev(bt::m_K, -1); 
            //now we resize the objects used to store the indices of lowered and raised objects.  
            m_nkm1.resize(bt::m_D, initialisev);
            m_nkp1.resize(bt::m_D, initialisev);
        }

        std::vector<size_t> nt(bt::m_K);  
        bt::m_np1 = 0;
        bt::m_nm1 = 0;

        for(size_t i=0; i < bt::m_D; ++i)
        {
            nt = m_nk[i];
            for(size_t j=0; j<bt::m_K; ++j)
            {
                if(nt[j] == 0){m_nkm1[i][j] = -1;}
                else
                {
                    nt[j] = m_nk[i][j] - 1;
                    auto itr = std::lower_bound(m_nk.begin(), m_nk.end(), nt, 
                                    [](const std::vector<size_t>& a, const std::vector<size_t>& b)
                                    {
                                        for(size_t ii=0; ii<a.size(); ++ii)
                                        {
                                            if(a[ii] < b[ii]){return true;}
                                            else if(a[ii] > b[ii]){return false;}
                                        }
                                        return false;
                                    }
                                );
                    if(itr == m_nk.end()){m_nkm1[i][j] = -1;}
                    else if(*itr != nt){m_nkm1[i][j] = -1;}
                    else{m_nkm1[i][j] = std::distance(m_nk.begin(), itr); ++bt::m_nm1;}
                }

                //now we attempt to find the +1
                nt[j] = m_nk[i][j] + 1;
                auto itr = std::lower_bound(m_nk.begin(), m_nk.end(), nt, 
                                [](const std::vector<size_t>& a, const std::vector<size_t>& b)
                                {
                                    for(size_t ii=0; ii<a.size(); ++ii)
                                    {
                                        if(a[ii] < b[ii]){return true;}
                                        else if(a[ii] > b[ii]){return false;}
                                    }
                                    return false;
                                }
                            );
                if(itr == m_nk.end()){m_nkp1[i][j] = -1;}
                else if(*itr != nt){m_nkp1[i][j] = -1;}
                else{m_nkp1[i][j] = std::distance(m_nk.begin(), itr); ++bt::m_np1;}

                nt[j] = m_nk[i][j];
            }
        }
    }

protected:
    template <typename U>
    U apply_sf(const U& w){return w;}

    template <typename U, typename F>
    auto apply_sf(const U& w, F&& f) -> decltype(f(w)){return f(w);}
};


//a class for constructing a direct product of occupation number basis objects.
class composite_occupation_number_basis : public occupation_number_basis
{
    using bt = occupation_number_basis;
protected:
    std::vector<size_t> m_dim;        //an array storing the occupation numbers of a 
    std::vector<size_t> m_stride;        //an array storing the occupation numbers of a 
    std::vector<std::shared_ptr<occupation_number_basis>> m_bases;

public:
    composite_occupation_number_basis() : bt() {}

    template <typename ... Args>
    composite_occupation_number_basis(Args&& ... args) : bt()
    {
        initialise(std::forward<Args>(args)...);
    }

    composite_occupation_number_basis(const std::vector<std::shared_ptr<occupation_number_basis>>& N) : bt()
    {
        CALL_AND_HANDLE(initialise(N), "Failed to construct occupation number indexing object.");
    }   

    composite_occupation_number_basis(const composite_occupation_number_basis& o) = default;
    composite_occupation_number_basis(composite_occupation_number_basis&& o) = default;

    composite_occupation_number_basis& operator=(const composite_occupation_number_basis& o) = default;
    composite_occupation_number_basis& operator=(composite_occupation_number_basis&& o) = default;
    ~composite_occupation_number_basis(){}
    
    virtual void clear()
    {
        m_dim.clear();
        bt::m_D = 0;
        bt::m_K = 0;
    }

    virtual size_t Nprods() const{return m_bases.size();}
    virtual bool is_direct_product() const
    {
        return true;
    }

    template <typename occt1, typename ... Args>
    typename std::enable_if<std::is_base_of<occupation_number_basis, occt1>::value, void>::type 
    initialise(occt1&& t1, Args&&... args)
    {
        try
        {
            bt::m_K += t1.nmodes();
            m_bases.push_back(std::make_shared<occt1>(std::move(t1)));
            initialise(std::forward<Args>(args)...);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise occupation number indexing object.");
        }
    }


    template <typename occt1, typename ... Args>
    typename std::enable_if<std::is_base_of<occupation_number_basis, occt1>::value, void>::type 
    initialise(const occt1& t1, Args&&... args)
    {
        try
        {
            bt::m_K += t1.nmodes();
            m_bases.push_back(std::make_shared<occt1>(t1));
            initialise(std::forward<Args>(args)...);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise occupation number indexing object.");
        }
    }

    //the final iteration step of the basis construction
    void initialise()
    {
        try
        {
            setup_dim_and_stride();
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise occupation number indexing object.");
        }
    }

    void initialise(const std::vector<std::shared_ptr<occupation_number_basis>>& N) 
    {
        try
        {  
             bt::m_K = 0;
             m_bases = N;
             for(size_t i = 0; i < N.size(); ++i)
             {
                 bt::m_K += N[i]->nmodes();
             }
             setup_dim_and_stride();
        }  
        catch(const std::exception& ex)
        {  
             std::cerr << ex.what() << std::endl;
             RAISE_EXCEPTION("Failed to initialise occupation number indexing object.");
        }  
    }      

    virtual void get_state(size_t i, std::vector<size_t>& res) const
    {
        ASSERT(i < bt::m_D, "Unable to occupation occupation number by index.  Invalid index.");
        res.resize(bt::m_K);
        size_t counter = i;

        size_t icounter = 0;
        for(size_t k=0; k < m_bases.size(); ++k)
        {
            std::vector<size_t> rloc(m_bases[k]->nmodes());
            size_t iloc = counter / m_stride[k];
            m_bases[k]->get_state(iloc, rloc);
            
            for(size_t j = 0; j < m_bases[k]->nmodes(); ++j)
            {
                res[icounter] = rloc[j];
                ++icounter;
            }

            counter = counter - iloc*m_stride[k];
        }
    }

    virtual size_t get_occupation(size_t i, size_t kind) const
    {
        ASSERT(i < bt::m_D, "Unable to occupation occupation number by index.  Invalid index.");
        ASSERT(kind < bt::m_K, "Unable to occupation occupation number by index.  Invalid index.");

        size_t resk=0;
        size_t counter = i;
        size_t kcounter = 0;

        for(size_t k=0; k < m_bases.size(); ++k)
        {
            resk = counter/m_stride[k];

            if(kind < kcounter + m_bases[k]->nmodes())
            {
                return m_bases[k]->get_occupation(resk, kind-kcounter);
            }
            kcounter += m_bases[k]->nmodes();
            counter = counter - resk*m_stride[k];
        }
        RAISE_EXCEPTION("Something went exceptionally wrong in the get occupation call.");
    }

    virtual bool contains_state(const std::vector<size_t>& ind) const
    {
        size_t icounter = 0;
    
        for(size_t k=0; k < m_bases.size(); ++k)
        {
            std::vector<size_t> rloc(m_bases[k]->nmodes());
            for(size_t i = 0; i < m_bases[k]->nmodes(); ++i)
            {
                rloc[i] = ind[icounter];
                ++icounter;
            }
            if(!m_bases[k]->contains_state(rloc)){return false;}

        }
        return true;
    }

    virtual size_t get_index(const std::vector<size_t>& ind) const
    {
        size_t res = 0;
        size_t icounter = 0;
    
        for(size_t k=0; k < m_bases.size(); ++k)
        {
            std::vector<size_t> rloc(m_bases[k]->nmodes());
            for(size_t i = 0; i < m_bases[k]->nmodes(); ++i)
            {
                rloc[i] = ind[icounter];
                ++icounter;
            }
            size_t iloc = m_bases[k]->get_index(rloc);

            res += m_stride[k]*iloc;
        }
        return res;
    }

    virtual bool contains_raised_state(size_t i, size_t kind) const
    {
        ASSERT(i < bt::m_D, "Unable to occupation occupation number by index.  Invalid index.");
        ASSERT(kind < bt::m_K, "Unable to occupation occupation number by index.  Invalid index.");
        size_t resk=0;
        size_t counter = i;
        size_t kcounter = 0;

        for(size_t k=0; k < m_bases.size(); ++k)
        {
            resk = counter/m_stride[k];

            if(kind < kcounter + m_bases[k]->nmodes())
            {
                return m_bases[k]->contains_raised_state(resk, kind-kcounter);
            }
            kcounter += m_bases[k]->nmodes();
            counter = counter - resk*m_stride[k];
        }
        return false;
    }

    virtual bool contains_lowered_state(size_t i, size_t kind) const
    {
        ASSERT(i < bt::m_D, "Unable to occupation occupation number by index.  Invalid index.");
        ASSERT(kind < bt::m_K, "Unable to occupation occupation number by index.  Invalid index.");
        size_t resk=0;
        size_t counter = i;
        size_t kcounter = 0;

        for(size_t k=0; k < m_bases.size(); ++k)
        {
            resk = counter/m_stride[k];

            if(kind < kcounter + m_bases[k]->nmodes())
            {
                return m_bases[k]->contains_lowered_state(resk, kind-kcounter);
            }
            kcounter += m_bases[k]->nmodes();
            counter = counter - resk*m_stride[k];
        }

        return false;
    }

    //we want to see if it has an element where k1 is lowered and k2 is raised
    virtual bool contains_adjacent_state(size_t i, size_t k1, size_t k2) const
    {
        ASSERT(i < bt::m_D && k1 < bt::m_K && k2  < bt::m_K, "Unable to get index of raised term.  Index out of bounds.");

        //if k1 == k2 then the elements are the same and so this isn't an adjacent term
        if(k1 == k2){return false;}

        size_t resk=0;
        size_t counter = i;
        bool valid_state = true;
        size_t kcounter = 0;

        for(size_t k=0; k < m_bases.size(); ++k)
        {
            resk = counter/m_stride[k];

            if(k1 < kcounter + m_bases[k]->nmodes() && k1 >= kcounter)
            {
                valid_state = valid_state && m_bases[k]->contains_lowered_state(resk, k1-kcounter);
            }
            else if(k2 < kcounter + m_bases[k]->nmodes() && k2 >= kcounter)
            {
                valid_state = valid_state && m_bases[k]->contains_raised_state(resk, k2-kcounter);
            }          
            kcounter += m_bases[k]->nmodes();
            counter = counter - resk*m_stride[k];
        }
        return valid_state;
    }

    virtual size_t get_raised_index(size_t i, size_t kind) const
    {
        ASSERT(i < bt::m_D && kind < bt::m_K, "Unable to get index of raised term.  Index out of bounds.");
        size_t resk=0;
        size_t counter = i;
        size_t index = 0;
        size_t kcounter = 0;

        try
        {  
            for(size_t k=0; k < m_bases.size(); ++k)
            {
                resk = counter/m_stride[k];

                if(kind < kcounter + m_bases[k]->nmodes() && kind >= kcounter)
                {
                    size_t rind = m_bases[k]->get_raised_index(resk, kind-kcounter);
                    index += rind*m_stride[k];
                }
                else
                {
                    index += resk*m_stride[k];
                }
                kcounter += m_bases[k]->nmodes();
                counter = counter - resk*m_stride[k];
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct raised index for composite basis.");
        }
        return index;
    }
    
    virtual size_t get_lowered_index(size_t i, size_t kind) const
    {
        ASSERT(i < bt::m_D && kind < bt::m_K, "Unable to get index of raised term.  Index out of bounds.");
        size_t resk=0;
        size_t counter = i;
        size_t index = 0;
        size_t kcounter = 0;

        try
        {
            for(size_t k=0; k < m_bases.size(); ++k)
            {
                resk = counter/m_stride[k];

                if(kind < kcounter + m_bases[k]->nmodes() && kind >= kcounter)
                {
                    size_t rind = m_bases[k]->get_lowered_index(resk, kind-kcounter);
                    index += rind*m_stride[k];
                }
                else
                {
                    index += resk*m_stride[k];
                }
                kcounter += m_bases[k]->nmodes();
                counter = counter - resk*m_stride[k];
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct lowered index for composite basis.");
        }
        return index;
    }

    virtual size_t get_adjacent_index(size_t i, size_t k1, size_t k2) const
    {
        ASSERT(i < bt::m_D && k1 < bt::m_K && k2  < bt::m_K, "Unable to get index of adjacent term.  Index out of bounds.");
        ASSERT(k1 != k2, "Unable to get index of adjacent term.  If k1 == k2 then this is not an adjacent term.");

        size_t resk=0;
        size_t counter = i;
        size_t index = 0;
        size_t kcounter = 0;

        try
        {
            for(size_t k=0; k < m_bases.size(); ++k)
            {
                resk = counter/m_stride[k];

                if(k1 < kcounter + m_bases[k]->nmodes() && k1 >= kcounter)
                {
                    size_t rind = m_bases[k]->get_lowered_index(resk, k1-kcounter);
                    index += rind*m_stride[k];
                }
                else if( (k2 < kcounter + m_bases[k]->nmodes()) && (k2 >= kcounter))
                {
                    size_t rind = m_bases[k]->get_raised_index(resk, k2-kcounter);
                    index += rind*m_stride[k];
                }          
                else
                {
                    index += resk*m_stride[k];
                }
                kcounter += m_bases[k]->nmodes();
                counter = counter - resk*m_stride[k];
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct adjacent index for composite basis.");
        }
        return index;
    }

protected:
    void setup_dim_and_stride()
    {   
        m_stride.resize(m_bases.size());
        m_dim.resize(m_bases.size());
        m_localdims.resize(bt::m_K);
        bt::m_D = 1;
        size_t icounter = 0;
        for(size_t i = 0; i < m_bases.size(); ++i)
        {
            m_dim[i] = m_bases[i]->nstates();
            bt::m_D *= m_dim[i];
            for(size_t k = 0; k < m_bases[i]->nmodes(); ++k)
            {
                bt::m_localdims[icounter] = m_bases[i]->dim(k); 
                ++icounter;
            }
        }
    
        size_t counter = m_bases.size();
        m_stride[counter-1] = 1;

        for(size_t k = 1; k < m_bases.size(); ++k)
        {
            --counter;
            m_stride[counter-1] = m_dim[counter]*m_stride[counter];
        }
    }
};
}   //namespace utils


#endif  //UTILS_OCCUPATION_NUMBER_INDEXING_HPP_//


