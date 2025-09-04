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

#ifndef PYTTN_TTNS_LIB_TTN_TTN_HPP_
#define PYTTN_TTNS_LIB_TTN_TTN_HPP_

#include <random>
#include <common/tmp_funcs.hpp>
#include <common/exception_handling.hpp>

#include "ttn_nodes/ms_ttn_node.hpp"
#include "ttn_nodes/ttn_node.hpp"
#include "../op.hpp"
#include "../operators/site_operators/site_operator.hpp"
#include "../operators/product_operator.hpp"

#include "ttnbase.hpp"
#include "ms_ttn.hpp"

namespace ttns
{

    template <typename T, typename backend>
    using ttn_node = typename tree<ttn_node_data<T, backend>>::node_type;

    template <typename T, typename backend = linalg::blas_backend>
    class ttn : public ttn_base<ttn_node_data, T, backend>
    {
    public:
        using base_type = ttn_base<ttn_node_data, T, backend>;
        using real_type = typename base_type::real_type;
        using matrix_type = typename base_type::matrix_type;

        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using size_type = typename backend::size_type;

        using node_reference = typename base_type::node_reference;
        using const_node_reference = typename base_type::const_node_reference;

        using tree_type = typename base_type::tree_type;
        using tree_reference = typename base_type::tree_reference;
        using const_tree_reference = typename base_type::const_tree_reference;

        using node_type = typename base_type::node_type;
        using ancestor_index = typename base_type::ancestor_index;
        using bond_matrix_type = typename node_type::bond_matrix_type;

        template <typename U, typename be>
        friend class ttn;

    private:
        // provide access to base class operators
        using base_type::m_dim_sizes;
        using base_type::m_has_orthogonality_centre;
        using base_type::m_hrengine;
        using base_type::m_leaf_indices;
        using base_type::m_nleaves;
        using base_type::m_nodes;
        using base_type::m_orthog;
        using base_type::m_rengine;
        using base_type::rng;

    public:
        ttn() : base_type() {}

        ttn(const ttn &other)
        try : base_type(other) {}
        catch (const std::exception &ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct TTN object.");
        }

        template <typename U, typename = typename std::enable_if<not std::is_same<T, U>::value, void>::type>
        ttn(const ttn<U, backend> &other)
        try : base_type(other) {}
        catch (const std::exception &ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct TTN object.");
        }

        template <typename U, typename be, bool CONST>
        ttn(multiset_ttn_slice<U, be, CONST> other) : base_type()
        {
            CALL_AND_RETHROW(tree_type::construct_topology(static_cast<const typename ms_ttn<U, be>::tree_type &>(other.obj())));
            for (auto z : common::zip(m_nodes, other.obj()))
            {
                CALL_AND_HANDLE(std::get<0>(z)() = std::get<1>(z)()[other.slice_index()], "Failed when assigning slice index.");
            }

            this->m_dim_sizes = other.obj().mode_dimensions();
            this->m_leaf_indices = other.obj().leaf_indices();
            this->m_nset = 1;

            this->m_orthog.clear();

            this->m_orthogonality_centre = other.obj().orthogonality_centre();
            this->m_has_orthogonality_centre = other.obj().has_orthogonality_centre();

            this->m_euler_tour = other.obj().euler_tour();
            this->m_euler_tour_initialised = other.obj().euler_tour_initialised();
        }

        // template <typename U>
        // ttn(const tree<sttn_node_data>& sTTN) : base_type();
        //{
        //     CALL_AND_HANDLE(this->operator=(sTTN), "Failed to construct ttn object from sparse ttn node information.");
        // }

        template <typename INTEGER, typename Alloc>
        ttn(const ntree<INTEGER, Alloc> &topology, bool collapse_bond_matrices=true, bool purification = false)
        try : base_type(topology, 1, collapse_bond_matrices, purification) {}
        catch (const std::exception &ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct TTN object.");
        }

        template <typename INTEGER, typename Alloc>
        ttn(const ntree<INTEGER, Alloc> &topology, const ntree<INTEGER, Alloc> &capacity, bool collapse_bond_matrices=true, bool purification = false)
        try : base_type(topology, capacity, 1, collapse_bond_matrices, purification) {}
        catch (const std::exception &ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct TTN object.");
        }

        ttn(const std::string &_topology, bool collapse_bond_matrices=true, bool purification = false)
        try : base_type(_topology, 1, collapse_bond_matrices, purification) {}
        catch (const std::exception &ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct TTN object.");
        }

        ttn(const std::string &_topology, const std::string &_capacity, bool collapse_bond_matrices=true, bool purification = false)
        try : base_type(_topology, _capacity, 1, collapse_bond_matrices, purification) {}
        catch (const std::exception &ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct TTN object.");
        }

        size_type nset() const noexcept { return 1; }

    public:
        ttn &operator=(const ttn &other) = default;
        template <typename U, typename be>
        ttn &operator=(const ttn<U, be> &other)
        {
            CALL_AND_RETHROW(base_type::operator=(other));
            return *this;
        }

        // assign ttn from a multiset ttn slice
        template <typename U, typename be, bool CONST>
        ttn &operator=(multiset_ttn_slice<U, be, CONST> other)
        {
            // if these are all the same size then we just do the simple assignment operator
            if (has_same_structure(other.obj(), *this) && other.obj().mode_dimensions() == this->mode_dimensions())
            {
                bool all_fit = true;
                // first check to see if the current structure can fit the assigned structure.  If it can then we don't have any problems
                for (auto z : common::zip(m_nodes, other.obj()))
                {
                    if (!std::get<0>(z).can_fit_node(std::get<1>(z)()[other.slice_index()]))
                    {
                        all_fit = false;
                    }
                }

                for (auto z : common::zip(m_nodes, other.obj()))
                {
                    CALL_AND_HANDLE(std::get<0>(z)() = std::get<1>(z)()[other.slice_index()], "Failed when assigning slice index.");
                }
                if (!all_fit)
                {
                    this->m_orthog.clear();
                }

                this->m_orthogonality_centre = other.obj().orthogonality_centre();
                this->m_has_orthogonality_centre = other.obj().has_orthogonality_centre();
            }
            else
            {
                this->clear();
                CALL_AND_RETHROW(tree_type::construct_topology(static_cast<const typename ms_ttn<U, be>::tree_type &>(other.obj())));
                for (auto z : common::zip(m_nodes, other.obj()))
                {
                    CALL_AND_HANDLE(std::get<0>(z)() = std::get<1>(z)()[other.slice_index()], "Failed when assigning slice index.");
                }

                this->m_dim_sizes = other.obj().mode_dimensions();
                this->m_leaf_indices = other.obj().leaf_indices();
                this->m_nset = 1;

                this->m_orthog.clear();

                this->m_orthogonality_centre = other.obj().orthogonality_centre();
                this->m_has_orthogonality_centre = other.obj().has_orthogonality_centre();

                this->m_euler_tour = other.obj().euler_tour();
                this->m_euler_tour_initialised = other.obj().euler_tour_initialised();
            }
            return *this;
        }

        size_type maximum_bond_dimension() const
        {
            size_type mbd = 0;
            for (const auto &n : m_nodes)
            {
                if (n.hrank() > mbd)
                {
                    mbd = n.hrank();
                }
            }
            return mbd;
        }

        size_type minimum_bond_dimension() const
        {
            size_type mbd = 0;
            bool first_call = true;
            for (const auto &n : m_nodes)
            {
                if (!first_call)
                {
                    if (n.hrank() < mbd || mbd == 0)
                    {
                        mbd = n.hrank();
                    }
                }
                else
                {
                    first_call = false;
                }
            }
            return mbd;
        }

    public:
        template <typename int_type>
        void set_state(const std::vector<int_type> &si, bool random_unoccupied_initialisation = true, bool randomise_internal = true) { CALL_AND_RETHROW(this->_set_state(si, 0, false, random_unoccupied_initialisation, randomise_internal)); }

        template <typename int_type>
        void set_state_purification(const std::vector<int_type> &si, bool random_initialisation = true, bool randomise_internal = true) { CALL_AND_RETHROW(this->_set_state(si, 0, true, random_initialisation, randomise_internal)); }

        template <typename U, typename be>
        void set_product(const std::vector<linalg::vector<U, be>> &ps, bool randomise_internal = true) { CALL_AND_RETHROW(this->_set_product(ps, 0, randomise_internal)); }

        template <typename Rvec>
        void sample_product_state(std::vector<size_t> &state, const std::vector<Rvec> &relval, bool randomise_internal = true) { CALL_AND_RETHROW(this->_sample_product_state(state, relval, 0, randomise_internal)); }

        void set_identity_purification(bool randomise_internal = true)
        {
            CALL_AND_RETHROW(this->_set_purification(0, randomise_internal));
        }

    public:
        real_type bond_entropy(size_t /* bond_index */)
        {
            /*
                    try
                    {
                        if(!m_orthog.is_initialised()){m_orthog.init(*this, m_maxsize, m_maxcapacity);}

                        ASSERT(this->has_orthogonality_centre(), "The orthogonality centre must be specified in order to allow for it to be shifted.");
                        ASSERT(bond_index < m_nodes[m_orthogonality_centre].nbonds(), "Failed to shift orthogonality centre along bond.  Bond index out of bounds.");

                        bool bond_shifted = false;
                        //if we aren't at the root node - then we first check whether the bond_index is 0 in which case we should shift the
                        if(!m_nodes[m_orthogonality_centre].is_root())
                        {
                            //if the bond index is zero then we shift the orthogonality centre up the tree
                            if(bond_index == 0)
                            {
                                //perform the svd of this node
                                CALL_AND_HANDLE(l2r_core::evaluate(m_ortho_engine, m_nodes[m_orthogonality_centre]), "Failed to evaluate the leaf_to_root_decomposition for a given node.");
                                bond_shifted = true;
                            }
                            else
                            {
                                --bond_index;
                            }
                        }
                        if(!bond_shifted)
                        {
                            //evaluate the root to leaf decomposition provided we aren't at the leaf node and update the mean field hamiltonian
                            CALL_AND_HANDLE(r2l_core::evaluate(m_ortho_engine, m_nodes[m_orthogonality_centre], m_U, bond_index), "Failed to compute the root to leaf decomposition for a node.");
                        }

                        real_type be = 0.0;
                        for(size_type i = 0; i < m_orthog.eng().Shost().size(); ++i)
                        {
                            real_type si = m_orthog.eng().Shost()(i, i);
                            be += si*si*std::log(si*si);
                        }
                        return be;
                    }
                    catch(const std::exception& ex)
                    {
                        std::cerr << ex.what() << std::endl;
                        RAISE_EXCEPTION("Failed to shift orthogonality centre.");
                    }
            */
            return 0;
        }

        real_type compute_maximum_bond_entropy()
        {
            /*
                    try
                    {
                        //first we ensure that the ttn is orthogonalised to the root node
                        CALL_AND_HANDLE(this->orthogonalise(), "Failed to orthogonalise ttn object.");

                        //now we perform an euler tour of the tree structure and truncate each bond on our first pass through ultimately shifting the orthogonality centre to the root of the tree.
                        if(!m_euler_tour_initialised)
                        {
                            traversal_path::initialise_euler_tour(*this, m_euler_tour);
                        }

                        m_maximum_bond_entropy = 0;

                        m_euler_tour.reset_visits();
                        //now perform the euler tour
                        for(size_type id : m_euler_tour)
                        {
                            size_type times_visited = m_euler_tour.times_visited(id);
                            m_euler_tour.visit(id);

                            const auto& A = m_nodes[id];
                            //now provided this isn't the first time we've traversed the node we will need to apply a root to leaf node decomposition to
                            //it so that we can propagate factors down the tree structure to its children.
                            if(!m_euler_tour.last_visit(id))
                            {
                                //get the index of the child we will be performing the decomposition for
                                size_type mode = times_visited;

                                //if we aren't at the leaf node we shift the orthogonality centre down the correct node
                                if(!A.is_leaf())
                                {
                                    ASSERT(id == m_orthogonality_centre, "Something went wrong when performing euler tour.");
                                    this->shift_orthogonality_centre(mode + (id == 0 ? 0 : 1), 0.0, 0, false, true);
                                }
                            }
                            //if it is our final time accessing the node we shift the orthogonality centre back up the tree
                            else
                            {
                                if(!A.is_root())
                                {
                                    ASSERT(id == m_orthogonality_centre, "Something went wrong when performing euler tour final.");
                                    this->shift_orthogonality_centre(0, 0.0, 0, false, true);
                                }
                            }
                        }
                        m_euler_tour.reset_visits();
                        return m_maximum_bond_entropy;
                    }
                    catch(const std::exception& ex)
                    {
                        std::cerr << ex.what() << std::endl;
                        RAISE_EXCEPTION("Failed to truncate ttn object.");
                    }
            */
            return 0;
        }

    public:
        // scalar inplace multiplication and division
        template <typename U>
        typename std::enable_if<linalg::is_number<U>::value, ttn &>::type operator*=(const U &u)
        {
            base_type::operator*=(u);
            return *this;
        }

        template <typename U>
        typename std::enable_if<linalg::is_number<U>::value, ttn &>::type operator/=(const U &u)
        {
            base_type::operator/=(u);
            return *this;
        }
#ifdef CEREAL_LIBRARY_FOUND
    public:
        template <typename archive>
        void save(archive &ar) const
        {
            CALL_AND_HANDLE(ar(cereal::base_class<base_type>(this)), "Failed to serialise ttn object.  Error when serialising the base object.");
        }

        template <typename archive>
        void load(archive &ar)
        {
            CALL_AND_HANDLE(ar(cereal::base_class<base_type>(this)), "Failed to serialise ttn object.  Error when serialising the base object.");
        }
#endif

    protected:
        void _setup_orthogonality_1bop(size_type index, bool shift_orthogonality = true)
        {
            if (shift_orthogonality)
            {
                CALL_AND_HANDLE(this->set_orthogonality_centre(index), "Failed to apply one body operator.  Failed to shift orthogonality centre.");
            }
            else
            {
                m_has_orthogonality_centre = false;
            }
        }

        void _apply_one_body_operator(const linalg::matrix<T, backend> &M, size_type index, bool shift_orthogonality = true)
        {
            CALL_AND_RETHROW(_setup_orthogonality_1bop(index, shift_orthogonality));

            try
            {
                // now we apply the operator to the state
                linalg::matrix<T, backend> temp = M * m_nodes[index]().as_matrix();
                m_nodes[index]().as_matrix() = temp;
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply one body operator.  Error when contracting one body operator onto state.");
            }
        }

        template <typename OpType>
        typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, void>::type
        _apply_one_body_operator(OpType &op, size_type index, bool shift_orthogonality = true)
        {
            CALL_AND_RETHROW(_setup_orthogonality_1bop(index, shift_orthogonality));

            try
            {
                // now we apply the operator to the state
                linalg::matrix<T, backend> temp = m_nodes[index]().as_matrix();
                op.apply(temp, m_nodes[index]().as_matrix());
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply one body operator.  Error when contracting one body operator onto state.");
            }
        }
        void _apply_one_body_operator(std::shared_ptr<ops::primitive<T, backend>> op, size_type index, bool shift_orthogonality = true)
        {
            CALL_AND_RETHROW(_setup_orthogonality_1bop(index, shift_orthogonality));

            try
            {
                // now we apply the operator to the state
                linalg::matrix<T, backend> temp = m_nodes[index]().as_matrix();
                op->apply(temp, m_nodes[index]().as_matrix());
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply one body operator.  Error when contracting one body operator onto state.");
            }
        }

    protected:
        //a function for applying a two-body operator to a tensor network.  The implementation provided here
        //performs this contraction using a two site approach
        void _apply_two_body_operator(const linalg::tensor<T, 4, backend>& _A, const linalg::tensor<T, 4, backend>& _B, size_type i1, size_type i2, real_type tol = real_type(-1), size_type nchi=0, bool zipup = false)
        {
            try
            {
                if(_A.shape(0) != 1 || _B.shape(3) != 1)
                {
                    RAISE_EXCEPTION("Failed to apply two body operator.  Internal buffers are not the correct shape")
                }
                auto A = _A.reinterpret_shape(_A.shape(1), _A.shape(2), _A.shape(3));
                auto B = _B.reinterpret_shape(_B.shape(0), _B.shape(1), _B.shape(2));
                size_type chi = _A.shape(3);

                //we first shift the orthogonality centre to the left node that is being acted upon
                CALL_AND_HANDLE(this->set_orthogonality_centre(i2), "Failed to apply two body operator.  Failed to shift orthogonality centre.");
                
                //get the path through the tensor network connecting nodes i1 and i2
                std::list<size_type> path;  this->path(i2, i1, path);

                //now apply the A operator to node i1.  This leaves us with an indexing of the form
                //                                      dangling bond (1)
                //                                           |
                //    primitive_degree of freedom (0) - active_tensor - upward pointing bond (2)
                linalg::tensor<T, 3, backend> active_tensor;
                linalg::tensor<T, 3, backend> working_tensor;


                //now we perform the contraction acting at node i2.  Again setting the tensor so 
                //that it is the first index pointing upwards
                CALL_AND_HANDLE(active_tensor = linalg::tensordot(B, m_nodes[i2]().as_matrix(), std::array<int, 1>{{2}}, std::array<int, 1>{{0}}), "Failed to apply two body operator.  Failed to apply operator mpo to tensor.")
                auto d = active_tensor.shape();
                
                CALL_AND_HANDLE(m_nodes[i2]().resize_bond(m_nodes[i2]().nmodes(), d[0]*d[2]), "Failed to resize tensor node so that it can fit the required node");
                CALL_AND_HANDLE(working_tensor = linalg::transpose(active_tensor, {1, 0, 2}), "Failed to reorder tensor indices.");
                CALL_AND_HANDLE(m_nodes[i2]().as_matrix() = working_tensor.reinterpret_shape(d[1], d[0]*d[2]), "Failed to set final tensor node.");
                
                size_type prev = i2;

                //now iterate over the nodes connecting i1 to i2.  
                //That is traverse the path to one element less than its end.
                for(auto it = path.begin(); std::next(it) != path.end(); ++it)
                {   
                    size_type curr = 0;
                    size_type i = *it;
                    bool curr_found = false;
                    size_type m1 = 0;

                    //figure out the current node index.  Additional determine which bond connects the current node to the previous node
                    if(!m_nodes[prev].is_root())
                    {
                        //if i == 0 we are moving upwards so the current node is the parent of the previous
                        //node and it is connected by the bond corresponding to the previous nodes parent index.
                        if(i == 0)
                        {
                            curr = m_nodes[prev].parent_pointer()->id();
                            curr_found = true;
                            m1 = m_nodes[prev].child_id();
                        }
                        
                        //if we aren't moving up decrement i so it points in the children index
                        else{--i;}
                    }

                    //now at this stage if we haven't assigned the current node the only option is the ith
                    //child of the previous node and it is connected by the upward pointing bond
                    if(!curr_found)
                    {
                        curr = m_nodes[prev].child_pointer(i)->id();
                        m1 = m_nodes[curr].nmodes();
                    }

                    //and kron this with the identity tensor acting along the path
                    bool next_found = false;
                    size_type m2 = 0;
                    size_type ix = *(std::next(it));
                    //now we want to determine which bond points towards the next node in the path.
                    //if the current node is not the root
                    if(!m_nodes[curr].is_root())
                    {
                        //we first check to see if we are still traversing upwards.  In this case m2 is
                        //set to point up.
                        if(ix == 0)
                        {
                            next_found = true;
                            m2 = m_nodes[curr].nmodes();
                        }
                        //otherwise decrement the indexing so it points to the child index
                        else{--ix;}
                    }
                    //if at the stage we haven't found the bond pointing to the next index it is the bond corresponding 
                    //to the ith child of this node. 
                    if(!next_found){m2 = ix;}
                    if(m1 > m2)
                    {
                        size_type mtemp = m1;
                        m1 = m2;
                        m2 = mtemp;
                    }

                    //get the rank 5 representation of the tensor with indices pointing along the path spanning this tree
                    auto M = m_nodes[curr]().as_rank_5(m1, m2);
                    std::array<size_type, 5> d2 = M.shape();

                    //now resize the active tensor so that it can fit the buffer needed for the result of the outer product
                    CALL_AND_HANDLE(active_tensor.resize(d2[0]*d2[1]*chi*d2[2]*chi*d2[3]*d2[4], 1, 1), "Failed to resize the active tensor");
                    active_tensor.fill_zeros();
                    std::array<size_type, 5> dest_dims{{d2[0], d2[1]*chi, d2[2], d2[3]*chi, d2[4]}};
                    std::array<size_type, 5> skip{{0, 0, 0, 0, 0}};

                    //now we go ahead and set the blocks needed in a rank 5 view of the active tensor using the set_tensor_block function provided
                    for(size_type ind = 0; ind < chi; ++ind)   
                    {                 
                        skip[1] = d2[1]*ind;
                        skip[3] = d2[3]*ind;
                        backend::set_tensor_block(m_nodes[curr]().buffer(), d2, active_tensor.buffer(), dest_dims, skip);
                    }

                    //now we resize the current node object so that it can fit the active tensor and copy the results
                    std::vector<size_type> dims = m_nodes[curr]().dims();
                    //if this is the root of the subtree then we need to set the new size to adjust two downward pointing indices
                    if(m2 != m_nodes[curr].nmodes())
                    {
                        dims[m1] *= chi;
                        dims[m2] *= chi;
                        CALL_AND_HANDLE(m_nodes[curr]().resize(m_nodes[curr]().hrank(), dims), "Failed to resize node object");
                    }
                    //otherwise we need to adjust the size of one downward pointing index and one upward pointing index.
                    else
                    {
                        dims[m1] *= chi;
                        CALL_AND_HANDLE(m_nodes[curr]().resize(m_nodes[curr]().hrank()*chi, dims), "Failed to resize node object");
                    }
                    ASSERT(active_tensor.size() == m_nodes[curr]().as_matrix().size(), "Invalid size for active tensor object.");
                    CALL_AND_HANDLE(m_nodes[curr]().as_matrix().set_buffer(active_tensor.buffer(), active_tensor.size()), "Failed to set current tensor buffer");
                    
                    prev = curr;
                    if(zipup)
                    {
                        CALL_AND_HANDLE(this->set_orthogonality_centre(curr, tol, nchi), "Failed to reorthogonalise tensor");
                    }
                }
                CALL_AND_HANDLE(active_tensor = linalg::tensordot(A, m_nodes[i1]().as_matrix(), std::array<int, 1>{{1}}, std::array<int, 1>{{0}}), "Failed to apply two body operator.  Failed to apply operator mpo to tensor.")
                d = active_tensor.shape();
                //now set the value of active_tensor so that it can set the leaf tensor value.  
                //Here we make it so that the dangling bond becomes the first index pointing upwards
                //of the bond
                CALL_AND_HANDLE(m_nodes[i1]().resize_bond(m_nodes[i1]().nmodes(), d[1]*d[2]), "Failed to resize tensor node so that it can fit the required node");
                CALL_AND_HANDLE(m_nodes[i1]().as_matrix() = active_tensor.reinterpret_shape(d[0], d[1]*d[2]), "Failed to set node of tensor given active tensor");
                if(zipup)
                {
                    CALL_AND_HANDLE(this->set_orthogonality_centre(i1), "Failed to reorthogonalise tensor");
                }

                if(!zipup)
                {
                    CALL_AND_HANDLE(this->set_orthogonality_centre(i1), "Failed to reorthogonalise tensor");
                    if(tol > real_type(0) || nchi > 0)
                    {
                        CALL_AND_HANDLE(this->set_orthogonality_centre(i2, tol, nchi), "Failed to turncate tensor");
                    }
                }
                
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply_two_body operator to ttn.");
            }
        }
        
        size_type nthreads() const { return 1; }
        void set_nthreads(size_t ) const{}

    public:
        ttn &apply_product_operator(product_operator<T, backend> &op, bool shift_orthogonality = true)
        {
            for (auto &_op : op)
            {
                CALL_AND_HANDLE(apply_one_body_operator(_op, shift_orthogonality), "Failed to apply product operator error when applying one body operator.");
            }
            if (this->has_orthogonality_centre())
            {
                m_nodes[this->m_orthogonality_centre] *= op.coeff();
            }
            else
            {
                m_nodes[0] *= op.coeff();
            }
            return *this;
        }

        ttn &apply_one_body_operator(const linalg::matrix<T, backend> &op, size_type index, bool shift_orthogonality = true)
        {
            ASSERT(index < this->nmodes(), "Failed to apply one body operator to ttn. Index out of bounds.");
            ASSERT(op.shape(0) == m_dim_sizes[index] && op.shape(1) == m_dim_sizes[index], "Failed to apply one body operator to ttn. Incompatible dimensions.");

            CALL_AND_RETHROW(_apply_one_body_operator(op, m_leaf_indices[index], shift_orthogonality));
            return *this;
        }

        ttn &apply_one_body_operator(const Op<T, backend> &op, bool shift_orthogonality = true)
        {
            ASSERT(op.ndim() == 1, "Failed to apply one body operator.  Operator is not one body.");
            ASSERT(op.indices()[0] < this->nmodes(), "Failed to apply one body operator to ttn. Index out of bounds.");
            ASSERT(op.dims()[0] == m_dim_sizes[op.indices()[0]], "Failed to apply one body operator to ttn. Incompatible dimensions.");

            CALL_AND_RETHROW(_apply_one_body_operator(op(), m_leaf_indices[op.indices()[0]], shift_orthogonality));
            return *this;
        }

        template <typename OpType>
        typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, ttn &>::type
        apply_one_body_operator(OpType &op, size_type index, bool shift_orthogonality = true)
        {
            ASSERT(index < this->nmodes(), "Failed to apply one body operator to ttn. Index out of bounds.");
            ASSERT(op.size() == m_dim_sizes[index], "Failed to apply one body operator to ttn. Incompatible dimensions.");

            CALL_AND_RETHROW(_apply_one_body_operator(op, m_leaf_indices[index], shift_orthogonality));
            return *this;
        }

        ttn &apply_one_body_operator(site_operator<T, backend> &op, bool shift_orthogonality = true)
        {
            size_type index = op.mode();
            ASSERT(index < this->nmodes(), "Failed to apply one body operator to ttn. Index out of bounds.");
            ASSERT(op.size() == m_dim_sizes[index], "Failed to apply one body operator to ttn. Incompatible dimensions.");

            CALL_AND_RETHROW(_apply_one_body_operator(op.op(), m_leaf_indices[index], shift_orthogonality));
            return *this;
        }

        ttn &apply_one_body_operator(site_operator<T, backend> &op, size_type index, bool shift_orthogonality = true)
        {
            ASSERT(index < this->nmodes(), "Failed to apply one body operator to ttn. Index out of bounds.");
            ASSERT(op.size() == m_dim_sizes[index], "Failed to apply one body operator to ttn. Incompatible dimensions.");

            CALL_AND_RETHROW(_apply_one_body_operator(op.op(), m_leaf_indices[index], shift_orthogonality));
            return *this;
        }

        ttn &apply_one_body_operator(std::shared_ptr<ops::primitive<T, backend>> op, size_type index, bool shift_orthogonality = true)
        {
            ASSERT(index < this->nmodes(), "Failed to apply one body operator to ttn. Index out of bounds.");
            ASSERT(op.size() == m_dim_sizes[index], "Failed to apply one body operator to ttn. Incompatible dimensions.");

            CALL_AND_RETHROW(_apply_one_body_operator(op, m_leaf_indices[index], shift_orthogonality));
            return *this;
        }

        // allow for a generic operator type object.
        // ttn& apply_one_body_operator
        ttn &apply_operator(const Op<T, backend> &op, real_type tol = real_type(0), size_type nchi = 0, bool zipup=false)
        {
            // first check that the operator is consistent with the TTN we are acting on
            for (size_type i = 0; i < op.ndim(); ++i)
            {
                ASSERT(op.indices()[i] < this->nmodes(), "Failed to apply operator to ttn. Index out of bounds.");
            }
            for (size_type i = 0; i < op.ndim(); ++i)
            {
                ASSERT(op.dims()[i] == m_dim_sizes[op.indices()[i]], "Failed to apply operator to ttn. Incompatible dimensions.");
            }

            // currently we only support the application of one and two body operator.  3 body and above operators
            // will require decomposition of a general operator into a TTNO with the structure of the subtree
            // containing the nodes it acts on and the entire path up to their lowest common ancestor
            if (op.ndim() == 1)
            {
                CALL_AND_RETHROW(_apply_one_body_operator(op(), m_leaf_indices[op.indices()[0]], true));
            }
            else if (op.ndim() == 2)
            {
                auto opmpo = op.as_mpo();
                CALL_AND_RETHROW(_apply_two_body_operator(opmpo[0], opmpo[1], m_leaf_indices[op.indices()[0]], m_leaf_indices[op.indices()[1]], tol, nchi, zipup));
            }
            else
            {
                RAISE_EXCEPTION("N>2 body operator applications are currently not supported.");
            }
            return *this;
        }

        ttn &apply_operator(site_operator<T, backend> &op, bool shift_orthogonality = true)
        {
            CALL_AND_RETHROW(return apply_one_body_operator(op, shift_orthogonality));
        }

        ttn &apply_operator(product_operator<T, backend> &op, bool shift_orthogonality = true)
        {
            CALL_AND_RETHROW(return apply_product_operator(op, shift_orthogonality));
        }

    public:
        // here the collapse algorithm will be implemented in place.  This will be done iteratively shifting the orthogonality centre of the tree to a leaf.  Computing the probability of observing the state in each possible configuration of that leaf.  Then sampling the state based on this.
        real_type collapse(std::vector<size_t> &state, bool truncate = false, real_type tol = real_type(0), size_type nchi = 0)
        {
            state.resize(m_nleaves);
            real_type pitot = 1.0;
            this->orthogonalise();
            this->normalise();
            for (size_t i = 0; i < m_nleaves; ++i)
            {
                std::vector<real_type> pi(m_dim_sizes[i]);
                // shift orthogonality centre to leaf index
                measure_without_collapse(i, pi);

                real_type pisum = 0.0;
                for (size_t j = 0; j < m_dim_sizes[i]; ++j)
                {
                    pisum += pi[j];
                }

                // now sample from the projection expectation values
                std::discrete_distribution<std::size_t> d{pi.begin(), pi.end()};
                auto &A = m_nodes[m_leaf_indices[i]];
                auto &a = A().as_matrix();
                size_t ind = d(rng());
                state[i] = ind;

                // now that we have sampled the index to retain we need to collapse the state onto this index.
                for (size_t j = 0; j < m_dim_sizes[i]; ++j)
                {
                    // if we aren't in the measured state send to zero
                    if (j != ind)
                    {
                        a[j] *= 0.0;
                    }
                    // otherwise divide by the probability of observing this state to ensure correct normalisation.
                    else
                    {
                        a[j] /= std::sqrt(pi[j]);
                    }
                }
                pitot *= (pi[ind] / pisum);
                this->force_set_orthogonality_centre(m_leaf_indices[i]);
            }
            // after having collapsed each state.  We can now go through and ensure normalisation and truncate.  Due to the projective measurement this should be bond-dimension 1
            this->set_orthogonality_centre(0);
            this->normalise();
            if (truncate)
            {
                this->truncate(tol, nchi);
                this->normalise();
            }

            return pitot;
        }

        real_type collapse_basis(std::vector<linalg::matrix<T, backend>> &U, std::vector<size_t> &state, bool truncate = false, real_type tol = real_type(0), size_type nchi = 0)
        {
            ASSERT(U.size() == m_nleaves, "Failed to collapse in user specified basis.  Basis transformation vectors are not compatible with ");
            state.resize(m_nleaves);
            real_type pitot = 1.0;
            this->orthogonalise();
            this->normalise();
            for (size_t i = 0; i < m_nleaves; ++i)
            {
                // for each basis we construct a random
                std::vector<real_type> pi(m_dim_sizes[i]);
                // shift orthogonality centre to leaf index
                this->set_orthogonality_centre(m_leaf_indices[i]);

                linalg::matrix<T, backend> b = U[i] * m_nodes[m_leaf_indices[i]]().as_matrix();

                real_type pisum = 0.0;
                for (size_t j = 0; j < m_dim_sizes[i]; ++j)
                {
                    pi[j] = linalg::real(linalg::dot_product(linalg::conj(b[j]), b[j]));
                    pisum += pi[j];
                }

                // now sample from the projection expectation values
                std::discrete_distribution<std::size_t> d{pi.begin(), pi.end()};
                size_t ind = d(rng());
                state[i] = ind;

                // now that we have sampled the index to retain we need to collapse the state onto this index.
                for (size_t j = 0; j < m_dim_sizes[i]; ++j)
                {
                    // if we aren't in the measured state send to zero
                    if (j != ind)
                    {
                        b[j] *= 0.0;
                    }
                    // otherwise divide by the probability of observing this state to ensure correct normalisation.
                    else
                    {
                        b[ind] /= std::sqrt(pi[ind]);
                    }
                }
                m_nodes[m_leaf_indices[i]]().as_matrix() = linalg::adjoint(U[i]) * b;
                pitot *= (pi[ind] / pisum);
                this->force_set_orthogonality_centre(m_leaf_indices[i]);
            }
            // after having collapsed each state.  We can now go through and ensure normalisation and truncate.  Due to the projective measurement this should be bond-dimension 1
            this->set_orthogonality_centre(0);
            this->normalise();

            if (truncate)
            {
                this->truncate(tol, nchi);
                this->normalise();
            }

            return pitot;
        }

        // function for measuring a single qubit
        void measure_without_collapse(size_type i, std::vector<real_type> &res)
        {
            ASSERT(i < m_nleaves, "Cannot measure on requested mode.  Index out of bounds.");
            res.resize(m_dim_sizes[i]);
            // shift orthogonality centre to leaf index
            this->set_orthogonality_centre(m_leaf_indices[i]);

            const auto &A = m_nodes[m_leaf_indices[i]];
            const auto &a = A().as_matrix();
            for (size_t j = 0; j < m_dim_sizes[i]; ++j)
            {
                res[j] = linalg::real(linalg::dot_product(linalg::conj(a[j]), a[j]));
            }
        }

        // function for performing a measurement on all modes.
        // void measure_all_without_collapse(std::vector<std::vector<real_type>>& res)
        //{
        //     res.resize(m_nleaves);
        //     for(size_t i = 0; i < m_nleaves; ++i)
        //     {
        //         measure_without_collapse(i, res[i]);
        //     }
        // }
    };

    template <typename T, typename backend, typename real_type = typename linalg::get_real_type<T>::type>
    real_type collapse_wavefunction(const ttn<T, backend> &o, ttn<T, backend> &res, std::vector<size_t> &state, bool truncate = false, real_type tol = real_type(0), typename backend::size_type nchi = 0)
    {
        // first we copy the res array into o
        res = o;

        // now perform the inplace collapse on o
        return o.collapse(state, truncate, tol, nchi);
    }

    template <typename T, typename backend>
    std::ostream &operator<<(std::ostream &os, const ttn<T, backend> &t)
    {
        os << "dims: [";
        for (size_t i = 0; i < t.nmodes(); ++i)
        {
            os << t.dim(i) << (i + 1 != t.nmodes() ? ", " : "]");
        }
        os << std::endl
           << static_cast<const tree<ttn_node_data<T, backend>> &>(t);
        return os;
    }

    template <typename T, typename backend>
    using httensor = ttn<T, backend>;
} // namespace ttns

#endif // PYTTN_TTNS_LIB_TTN_TTN_HPP_ //
