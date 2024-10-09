#ifndef TTNS_SOP_TTN_CONTRACTION_HPP
#define TTNS_SOP_TTN_CONTRACTION_HPP

#include <linalg/linalg.hpp>

#include "ttn.hpp"
#include "../operators/sop_operator.hpp"

namespace ttns
{

template <typename T, typename backend>
class sop_ttn_contraction_engine;


//implementation of sum of product operator ttn contraction for blas types.  This also supports action of two-mode operators on the sop
template <typename T>
class sop_ttn_contraction_engine<T, linalg::blas_backend>
{
public:
    using backend = linalg::blas_backend;
    using sop_type = sop_operator<T, backend>;
    using ttn_type = ttn<T, backend>;
    using size_type = typename ttn_type::size_type;
    using real_type = typename ttn_type::real_type;


    template <typename ntype>
    static inline bool node_has_identity(const ntype& op)
    {
        bool has_identity = false;
        for(size_t ind = 0; ind < op.nterms(); ++ind)
        {
            if(op[ind].is_identity_spf())
            {
                has_identity = true;
            }
        }
        return has_identity;
    }

    //function for resizing the output tensor so that it is large enough to store the result of the contraction of op and A.
    //This also validates the sizes of op and A to ensure that all of the operators are valid sizes.  
    static inline bool resize_output_network(const sop_type& Op, const ttn_type& A, ttn_type& B, real_type cutoff = real_type(1e-12))
    {
        const auto & cinf = Op.contraction_info();
        ASSERT(has_same_structure(cinf, A), "Cannot perform sop ttn contraction.  The input sop and ttn do not have the same tensor network topologies.");
        
        //if B does not have the same topology as A.  Then we will allocate an empty ttn using the construct topology function.
        //This sets up the node structure but doesn't ever allocate the node sizes.
        if(!has_same_structure(A, B))
        {
            B.reallocate(A);
        }

        bool include_constant_contribution = (std::abs(Op.Eshift() ) > cutoff);

        //now we iterate over every node in the tensor networks and we use the sizes of op and A to determine the sizes of B.
        for(size_type i=0; i < A.size(); ++i)
        {
            bool is_leaf = A[i].is_leaf();
            const auto& op = cinf[i];
            const auto& a = A[i]();
            auto& b = B[i]();

            //determine the bond dimension pointing up the node and optionally add on the identity term if it isn't present
            //and the sop includes a constant contribution
            size_type ophrank = op().nterms(); 

            if (include_constant_contribution)
            {
                ophrank += node_has_identity(op()) ? 1 : 0;
            }
            //if this is an interior node.  Then B is just the tensor kronecker product of the two nodes
            if(!is_leaf)
            {
                try
                {
                    size_type maxhrank = a.max_hrank();

                    size_type bhrank = a.hrank()*ophrank;

                    if(bhrank > maxhrank){maxhrank = bhrank;}

                    std::vector<size_type> dimen(A[i].size());
                    std::vector<size_type> maxdimen(A[i].size());
                    for(size_type cind = 0; cind < A[i].size(); ++cind)
                    {
                        //determine the bond dimension pointing downwards to the child node and optionally add on the identity term if it isn't present
                        //and the sop includes a constant contribution
                        size_type opmode_dim = op[cind]().nterms();
                        if(include_constant_contribution)
                        {
                            opmode_dim += node_has_identity(op[cind]()) ? 1 : 0;
                        }

                        //work out the bond dimensions pointing to each of the children nodes
                        dimen[cind] = a.dim(cind)*(opmode_dim);
                        maxdimen[cind] = a.dim(cind, true);
                        if(dimen[cind] > maxdimen[cind]){maxdimen[cind] = dimen[cind];}
                    }

                    //allocate the new b array object
                    b.reallocate(maxhrank, maxdimen);
                    b.resize(bhrank, dimen);
                }
                catch(const std::exception& ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to determine size of output TN.  Error at branch node.");
                }

            }
            //if we are at a leaf node.  First we check to make sure we can perform the requested truncation then
            else
            {
                //TODO: figure out how to check compatibility of mode dimensions
                //ASSERT(a().dimen() == 

                try
                {
                    size_type maxhrank = a.max_hrank();
                    size_type bhrank = a.hrank()*ophrank;

                    if(bhrank > maxhrank){maxhrank = bhrank;}

                    std::vector<size_type> dimen(1);
                    dimen[0] = a.dimen();

                    b.reallocate(maxhrank, dimen);
                    b.resize(bhrank, dimen);
                }
                catch(const std::exception& ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to determine size of output TN.  Error at branch node.");
                }
            }
        }

        return include_constant_contribution;
    }

    static inline void sop_ttn_contraction(const sop_type& Op, const ttn_type& A, ttn_type& B, T coeff = T(1.0), real_type cutoff = real_type(1e-12))
    {
        bool include_constant = false;
        CALL_AND_HANDLE(include_constant = resize_output_network(Op, A, B, cutoff), "Failed to perform sop ttn contraction.  Failed to resize output TTN.");

        const auto & cinf = Op.contraction_info();
        //now we know that B is the correct size we perform the exact contractions.
        //For all interior nodes.  This just corresponds to a kronecker product.  
        //For all exterior nodes.  We actually need to perform the required contractions.
#ifdef USE_OPENMP
        #pragma omp parallel for default(shared)
#endif
        for(size_type i = 0; i < A.size(); ++i)
        {
            const auto& op = cinf[i];
            const auto& a = A[i]();
            auto& b = B[i]();
            //start by filling the output matrix with zeros
            b.as_matrix().fill_zeros();

            if(A[i].is_leaf())
            {
                CALL_AND_HANDLE(sop_ttn_contraction_leaf(Op, op(), a, b, include_constant), "Failed to perform leaf contraction.")
            }
            else
            {
                CALL_AND_HANDLE(sop_ttn_contraction_branch(op, a, b, include_constant, Op.Eshift(), coeff), "Failed to perform branch contraction.")
            }
        }
    }


    static inline void sop_ttn_contraction_zip_up(const sop_type& Op, const ttn_type& A, ttn_type& B, T coeff = T(1.0), real_type tol = real_type(0), size_type nchi = 0, real_type cutoff = real_type(1e-12))
    {
        bool include_constant = false;
        CALL_AND_HANDLE(include_constant = resize_output_network(Op, A, B, cutoff), "Failed to perform sop ttn contraction.  Failed to resize output TTN.");

        const auto & cinf = Op.contraction_info();
        //now we know that B is the correct size we perform the exact contractions.
        //For all interior nodes.  This just corresponds to a kronecker product.  
        //For all exterior nodes.  We actually need to perform the required contractions.
        
        for(size_type i = 0; i < A.size(); ++i)
        {
            const auto& op = cinf[i];
            const auto& a = A[i]();
            auto& b = B[i]();
            //start by filling the output matrix with zeros
            b.as_matrix().fill_zeros();

            if(A[i].is_leaf())
            {
                CALL_AND_HANDLE(sop_ttn_contraction_leaf(Op, op(), a, b, include_constant), "Failed to perform leaf contraction.")
            }
            else
            {
                CALL_AND_HANDLE(sop_ttn_contraction_branch(op, a, b, include_constant, Op.Eshift(), coeff), "Failed to perform branch contraction.")
            }
        }
    }

protected:
    template <typename Optype, typename Atype, typename Btype>
    static inline void sop_ttn_contraction_leaf(const sop_type& Op, const Optype& op, const Atype& a, Btype& b, bool include_constant_contribution)
    {
        //allocate the HA matrix so that it is large enough to store contraction results
        linalg::matrix<T, backend> temp(a.as_matrix().shape());

        bool identity_added = false;
        size_type ophrank = op.nterms(); 
        if (include_constant_contribution)
        {
            identity_added = node_has_identity(op);
            ophrank += identity_added ? 1 : 0;
        }


        linalg::tensor<T, 3, backend> HA(ophrank, a.as_matrix().shape(0), a.as_matrix().shape(1));
        HA.fill_zeros();

        //compute the action of each of the primitive operators on each term.
        for(size_type ind = 0; ind < op.nterms(); ++ind)
        {
            //if the mode is the identity operator then we just add on the matrix value of our tensor
            if(op[ind].is_identity_spf()){HA[ind] += a.as_matrix();}
            else
            {
                //otherwise we iterate over all of the terms in this term apply the action of the operator 
                //onto the site tensor and add that to the result
                for(size_type i = 0; i < op[ind].nspf_terms(); ++i)
                {
                    auto& indices = op[ind].spf_indexing()[i][0];
                    CALL_AND_HANDLE(Op[indices[0]][indices[1]].apply(a.as_matrix(), temp), "Failed to apply leaf operator.");
                    HA[ind] += op[ind].spf_coeff(i)*temp;
                }
            }
        }
        //if we have had to augment the operator with an additional identity term we add this at the end
        if(identity_added)
        {
            HA[op.nterms()] = a.as_matrix();
        }

        //and transpose the result so that we have the correct structure
        auto bv = b.as_matrix().reinterpret_shape(a.as_matrix().shape(0), ophrank, a.as_matrix().shape(1));
        bv = linalg::transpose(HA, {1, 0, 2});
    }
    
    template <typename Optype, typename Atype, typename Btype>
    static inline void sop_ttn_contraction_branch(const Optype& cinf, const Atype& a, Btype& b, bool include_constant_contribution, const T& Eshift, T coeff = T(1.0))
    {
        size_type N = cinf.size();
        const auto& op = cinf();

        //set up arrays containing the total number of child indices and also get the index
        //associated with the identity operator along each bond
        std::vector<size_type> opdims(N+1);
        std::vector<size_type> Adims(N+1);
        std::vector<size_type> idind(N);

        //determine whether we need to increment a new identity term and work out the operator hrank
        bool identity_added = false;
        size_type ophrank = op.nterms(); 
        if (include_constant_contribution && !cinf.is_root())
        {
            identity_added = node_has_identity(op);
            ophrank += identity_added ? 1 : 0;
        }

        //now iterate through each of the child nodes of this node and determine their hrank and 
        for(size_type i = 0; i < N; ++i)
        {
            opdims[i] = cinf[i]().nterms();
            bool child_identity_added = false;
            if(include_constant_contribution)
            {
                child_identity_added = node_has_identity(cinf[i]());
                opdims[i] += child_identity_added ? 1 : 0;
            }

            Adims[i] = a.dim(i);

            //now if we haven't incremented a child identity then we set the identity element by searching the child for an identity
            if(!child_identity_added)
            {
                for(size_type ind = 0; ind < cinf[i]().nterms(); ++ind)
                {
                    if(cinf[i]()[ind].is_identity_spf())
                    {
                        idind[i] = ind;
                        break;
                    }
                }
            }
            //but if we have added on an identity then it is the last index
            else
            {
                idind[i] = cinf[i]().nterms();
            }
        }

        //now finish setting up the dimension arrays.
        Adims[N] = a.hrank();
        opdims[N] = ophrank;


        //Construct the composite dimensions and a stride array
        std::vector<size_type> dims(2*N+2);
        std::vector<size_type> strides(2*N+2);

        //set up the total dims array
        for(size_type i =0; i < N+1; ++i)
        {
            dims[2*i] = opdims[i];
            dims[2*i+1] = Adims[i];
        }

        //now set up the strides arrays
        strides[2*N+1] = 1;
        for(size_type j = 1; j < strides.size(); ++j)
        {
            strides[2*N+1-j] = strides[2*N+2-j]*dims[2*N+2-j];
        }


        //and store the index of the operator element
        std::vector<size_type> opind(N+1);

        //get the matrix representation of A
        const auto& Ai = a.as_matrix();

        //compress the B object to a rank 1 object
        auto Bv = b.as_rank_1();

        //now in this case we don't need to perform any contractions it is simply a sparse tensor, tensor kron prod
        for(size_type ind = 0; ind < op.nterms(); ++ind)
        {
            opind[N] = ind;

            T rcoeff = 1;
            if(cinf.is_root())
            {
                rcoeff *= coeff*op[ind].coeff();
            }

            if(op[ind].is_identity_spf())
            {
                //set the current indexing object.  This is done by first filling it with the identity terms then updating
                //from the required spf_indexing object
                std::copy_n(idind.begin(), N, opind.begin());

                //expand out the index
                size_type coeff_index = expand_coeff_index(opind, strides);

                //and add the identity block to the matrix
                add_block(rcoeff, coeff_index, Ai, Bv, Adims, strides);
            }
            else
            {
                for(size_type ri = 0; ri < op[ind].nspf_terms(); ++ri)
                {
                    //set the current indexing object.  This is done by first filling it with the identity terms then updating
                    //from the required spf_indexing object
                    std::copy_n(idind.begin(), N, opind.begin());

                    //determine the multi-index associated with the operator node
                    const auto& spinds = op[ind].spf_indexing()[ri];
                    for(size_type ni=0; ni<spinds.size(); ++ni)
                    {
                        size_type nu = spinds[ni][0];
                        size_type cri = spinds[ni][1];
                        opind[nu] = cri;
                    }

                    //expand out the index
                    size_type coeff_index = expand_coeff_index(opind, strides);
                  
                    //compute the coefficient from the operator term
                    T lcoeff = rcoeff*op[ind].spf_coeff(ri);

                    //set the block of the result
                    add_block(lcoeff, coeff_index, Ai, Bv, Adims, strides);
                }
            }
        }

        //now if we have added on an identity term, then we can add it to the result matrix
        if(identity_added)
        {
            ASSERT(!cinf.is_root(), "Cannot append additional index to root node");
            T rcoeff(1);

            //set up the operator index element
            std::copy_n(idind.begin(), N, opind.begin());
            opind[N] = op.nterms();

            //expand out the index
            size_type coeff_index = expand_coeff_index(opind, strides);

            //and add the identity block to the matrix
            add_block(rcoeff, coeff_index, Ai, Bv, Adims, strides);
        }

        if(cinf.is_root() && include_constant_contribution)
        {
            T rcoeff = coeff*Eshift;

            //set up the operator index element
            std::copy_n(idind.begin(), N, opind.begin());
            opind[N] = 0;

            //expand out the index
            size_type coeff_index = expand_coeff_index(opind, strides);

            //and add the identity block to the matrix
            add_block(rcoeff, coeff_index, Ai, Bv, Adims, strides);
        }
    }

  template <typename A1type, typename Bvtype>
    static inline void add_block(const T& lcoeff, size_type coeff_index, const A1type& Ai, Bvtype& Bv, const std::vector<size_type>& Adims, const std::vector<size_type>& strides)
    {
        size_type N = Adims.size()-1;
        std::vector<size_type> Aind(N+1);   
        //reset the a index array to zero
        std::fill(Aind.begin(), Aind.end(), 0);
        //now we iterate over the rows and oclumns of the Ai matrix
        for(size_type ad = 0; ad < Ai.shape(0); ++ad)
        {   
            //set its final index
            for(size_type ah = 0; ah < Ai.shape(1); ++ah)
            {
                Aind[N] = ah;

                //now combine the coefficient index and the matrix index to get the output index
                size_type index = coeff_index+expand_matrix_index(Aind, strides);

                //and set its value as the product of the two terms
                Bv[index] += lcoeff * Ai(ad, ah);
            }

            //and increment the current index. 
            //attempt to increment the current last index
            size_type mind = N-1;  ++Aind[mind];
            //while we have reached the limit of the currently active index
            while(Aind[mind] == Adims[mind])
            {
                //reset it to zero
                Aind[mind] = 0;
                //and as long as we haven't reach the fron
                if(mind != 0)
                {
                    //decrement the index we are updating
                    mind = mind-1;

                    //and increment that value
                    ++Aind[mind];
                }
                else
                {
                    break;
                }
            }
        }
    }


    static inline size_type expand_coeff_index(const std::vector<size_type>& id, const std::vector<size_type>& strides)
    {
        size_type ret = 0;
        for(size_type i = 0; i < id.size(); ++i)
        {
            ret += id[i]*strides[2*i];
        }
        return ret;
    }


    static inline size_type expand_matrix_index(const std::vector<size_type>& id, const std::vector<size_type>& strides)
    {
        size_type ret = 0;
        for(size_type i = 0; i < id.size(); ++i)
        {
            ret += id[i]*strides[2*i+1];
        }
        return ret;
    }

};


}   //namespace ttns


#endif  //TTNS_SOP_TTN_CONTRACTION_HPP//
