#ifndef TTNS_LIB_SOP_AUTOSOP_HPP
#define TTNS_LIB_SOP_AUTOSOP_HPP

#include <utility>
#include <utils/product_iterator.hpp>
#include <utils/bipartite_graph.hpp>


#include "autoSOP_node.hpp"
#include "sop_tree.hpp"
#include "compressedSOP.hpp"
#include "../ttn/ttn.hpp"
#include "coeff_type.hpp"

//#define TIMING 0
#include <common/timing_macro.hpp>

namespace ttns
{

//TODO: Need to fix accumulation coefficients when working with specific hamiltonian types
//      Need to debug this code carefully something is going very wrong - but it works when
//      all accumulation coefficients are set to 1.
template <typename T>
class autoSOP
{
public:
    using site_ops_type = typename compressedSOP<T>::site_ops_type;
protected:


    template <typename node_type>
    static inline void update_spf_leaf(const compressedSOP<T>& csop, node_type& n)
    {
        ASSERT(n.is_leaf(), "Failed to process spf data.  Node is not a leaf.")
        n().spf().clear();
        auto& spf = n().spf();
        spf.clear();
        size_t nu = n.leaf_index();

        //for the leaf term we set the r values for the spf to just be the r values for each term
        const auto& mode_ops = csop(nu);

        size_t identity_index = csop.identity_index(nu);

        spf.reserve(mode_ops.size());
        size_t count = 0;
        for(const auto& op : mode_ops)
        {
            spf.push_back(auto_sop::operator_data<T>({{{count}}, count == identity_index}, op.second));
            ++count;
        }
    }

    //union based extraction of single particle operators at a node from its children.  This call iterates over the product of each of the children operators
    //and so works well when there are few terms for each children.  When there are many terms this becomes much less efficient than simply iterating over 
    //each term in the Hamiltonian
    template <typename node_type>
    static inline void update_spf_union(node_type& n, size_t nterms)
    {
        ASSERT(!n.is_leaf(), "Failed to process spf data.  Node is a leaf.")
        auto& spf = n().spf();
        spf.clear();

        //initialise an object to keep count of how many of the operator terms have been bound for each specific operator
        size_t nmodes = n.size();
        std::vector<std::vector<size_t>> m_nbound(nmodes);
        std::vector<std::vector<size_t>> m_index(nmodes);

        size_t nprod = 1;

        for(size_t i = 0; i < n.size(); ++i)
        {
            m_nbound[i].resize(n[i]().spf().size());
            std::fill(m_nbound[i].begin(), m_nbound[i].end(), 0);

            m_index[i].resize(n[i]().spf().size());
            for(size_t j = 0; j < m_index[i].size(); ++j)
            {
                m_index[i][j] = j;
            }
            nprod *= n[i]().spf().size();
        }
        utils::product_iterator<size_t> bound_iter = utils::prod_begin(m_nbound);
        utils::product_iterator<size_t> index_iter = utils::prod_begin(m_index);

        utils::term_indexing_array<size_t> r(nterms);
        utils::term_indexing_array<size_t> rintersect(nterms);
        //iterate over the outer product of children mode objects.  That is if we have two children with modes [a, b, c] and [d, e, f, g] we will iterate over the pairs
        //  ad, ae, af, ag, bd, be, bf, bg, cd, ce, cf, cg
        //and for each pair determine the common r indices of these operators.  
        for(auto prod_iter = n.spf_prod_begin(); 
            prod_iter != n.spf_prod_end() || bound_iter != utils::prod_end(m_nbound) || index_iter != utils::prod_end(m_index); 
            ++prod_iter, ++bound_iter, ++index_iter)
        {

            //now for each of the iterators see if any of them have already had all of their elements bound (e.g. nbound = size m_inds)
            //and if they have already been fully bound then we can skip checking whether we need to bind any of the terms
            bool fully_bound = false;
            for(size_t i = 0; i < nmodes; ++i)
            {
                if(prod_iter[i].m_inds.size() == bound_iter[i]){fully_bound = true;}
            }

            if(!fully_bound)
            {
                //if they have not been fully bound then we keep working out the set intersection of the r-index arrays of each of the current children operators
                rintersect = prod_iter[0].m_inds;
                bool all_identity = prod_iter[0].is_identity();

                for(size_t i = 1; i < nmodes; ++i)
                {
                    if(rintersect.size() != 0)
                    {
                        utils::term_indexing_array<size_t>::set_intersection(rintersect, prod_iter[i].m_inds, r);
                        all_identity = all_identity && prod_iter[i].is_identity();
                        rintersect = r;
                    }
                }

                //now if we get to the end and the union size is non-zero then we need to add this object to the 
                if(rintersect.size() != 0)
                {
                    auto_sop::operator_data<T> lop;
                    lop.m_opdef.m_indices.resize(1);
                    lop.m_opdef.m_indices[0].resize(nmodes);
                    for(size_t j = 0; j < nmodes; ++j)
                    {
                        //construct the new operator index
                        lop.m_opdef.m_indices[0][j] = index_iter[j];

                        //append to each of these operators the number of r terms that have been bound by adding in this matrix.
                        bound_iter[j] += rintersect.size();
                    }
                    lop.m_inds=rintersect;
                    lop.is_identity() = all_identity;
                    
                    //now add this to the spf data object
                    spf.push_back(lop);
                }
            }
        }
    }

    //term iteration based extraction of single particle operators at a node from its children.  This call iterates over each term in the Hamiltonian and
    //figures out the child operators present in the term binding them into forming a new operator at this node.  This call has roughly constant cost 
    //for each node in the tree but for systems with large numbers of terms in the Hamiltonian this can be quite a high cost, in which case it may be 
    //beneficial to use the union based scheme. 
    template <typename node_type>
    static inline void update_spf_term(node_type& n, size_t nterms)
    {
        ASSERT(!n.is_leaf(), "Failed to process spf data.  Node is a leaf.")
        auto& spf = n().spf();
        spf.clear();

        std::vector<size_t> spfr(nterms);
        std::vector<std::vector<size_t>> cspfr(n.size());
        for(size_t i = 0; i < n.size(); ++i)
        {
            cspfr[i].resize(nterms);
            n[i]().generate_spfr(cspfr[i], nterms);
        }

        std::map<std::vector<size_t>, size_t> ops;
        std::vector<size_t> currop(n.size());
        size_t ops_bound = 0;

        for(size_t r = 0; r <nterms; ++r)
        {
            bool all_identity = true;
            //build the child operator for this node 
            for(size_t ci = 0; ci < n.size(); ++ci)
            {
                currop[ci] = cspfr[ci][r];
                all_identity = all_identity && n[ci]().spf()[currop[ci]].is_identity();
            }

            //and see if this operator has been bound
            auto it = ops.find(currop);
            size_t opindex = ops_bound;
            if(it != ops.end())
            {
                opindex = it->second;
            }
            else
            {
                ops[currop] = ops_bound;
                spf.push_back(auto_sop::operator_data<T>(auto_sop::opinfo({currop}, all_identity)));
                ++ops_bound;
            }
            spfr[r] =  opindex;
        }
        n().setup_spf_r_indexing(spfr);
    }

    //determine the total number of terms we get in the proudct of the children nodes.  
    template <typename node_type>
    static inline size_t number_spf_products(node_type& n)
    {
        size_t nprod = 1;

        for(size_t i = 0; i < n.size(); ++i)
        {
            nprod *= n[i]().spf().size();
        }
        return nprod;
    }

    template <typename node_type>
    static inline void update_spf(node_type& n, size_t nterms)
    {
        ASSERT(!n.is_leaf(), "Incorrect SPF update for the current node.");
        n().spf().clear();

        size_t ncterms = number_spf_products(n);
        if(ncterms < nterms)
        {
            update_spf_union(n, nterms);
        }
        else
        {
            update_spf_term(n, nterms);
        }
    }

protected:
    //setup the spf operator information. This function switches between union and array based implementations depending on the number of terms 
    //that will be treated in each term.  Doing so can dramatically increase the efficiency of the code.
    static inline void setup_spf(const compressedSOP<T>& csop, tree<auto_sop::node_op_info<T>>& bp)
    {
        INIT_TIMER;
        for(auto& n : reverse(bp))
        {
            //if we are at a leaf node then all we need to do is copy the literal spf object info into this node
            if(n.is_leaf())
            {
                update_spf_leaf(csop, n);
            }
            //if we aren't at a leaf node then we iterate through all of the children of this node and construct the composite operators that are the spf operators here
            else
            {
                update_spf(n, csop.nterms());
            }
        }   
    }

    static inline void setup_mf(tree<auto_sop::node_op_info<T>>& bp, size_t nterms)
    {
        INIT_TIMER;
        for(auto& n : bp)
        {
            //If we are at the root node the mf operator is trivial
            if(n.is_root())
            {
                setup_root_node(n, nterms);
            }
            //if this isn't the root node then we need to actually work out the mf indexing objects.  
            else
            {
                update_mf(n, nterms);
            }
        }   
    }

protected:
    template <typename node_type>
    static void setup_root_node(node_type& n, size_t nterms)
    {
        ASSERT(n.is_root(), "node is not root.  Cannot set it to be a root.");
        n().mf().clear();

        utils::term_indexing_array<size_t> rempty;  rempty.reserve(nterms);   
        utils::term_indexing_array<size_t> rfull;   utils::term_indexing_array<size_t>::complement(rempty, rfull);

        n().mf().push_back(auto_sop::operator_data<T>(auto_sop::opinfo({{0}}, true), rfull));
    }

    template <typename node_type>
    static void update_mf(node_type& n, size_t nterms)
    {
        ASSERT(!n.is_root(), "Cannot update mf operator for root.");
        auto& mf = n().mf();
        mf.clear();

        auto& np = n.parent();

        std::map<std::vector<size_t>, size_t> ops;
        std::vector<size_t> currop(np.size());
        size_t ops_bound = 0;


        std::vector<size_t> pmfr(nterms);
        std::vector<size_t> mfr(nterms);

        np().generate_mfr(pmfr, nterms);
        std::vector<std::vector<size_t>> cspfr(np.size());
        for(size_t i = 0; i < np.size(); ++i)
        {
            if(i != n.child_id())
            {
                cspfr[i].resize(nterms);
                np[i]().generate_spfr(cspfr[i], nterms);
            }
        }

        for(size_t r = 0; r < nterms; ++r)
        {
            //add on the parent node term in the first index
            currop[0] = pmfr[r];

            bool all_identity = np().mf()[currop[0]].is_identity();

            size_t sibling_index = 1;
            //now the sibling operator indices 
            for(size_t ci = 0; ci < np.size(); ++ci)
            {
                if(ci != n.child_id())
                {
                    size_t ri = cspfr[ci][r];
                    currop[sibling_index] = ri;
                    all_identity = all_identity && np[ci]().spf()[ri].is_identity();
                    ++sibling_index;
                }
            }

            //and see if this operator has been bound
            auto it = ops.find(currop);
            size_t opindex = ops_bound;
            if(it != ops.end())
            {
                opindex = it->second;
                mf[opindex].is_identity() = mf[opindex].is_identity() && all_identity;
            }
            else
            {
                ops[currop] = ops_bound;
                mf.push_back(auto_sop::operator_data<T>(auto_sop::opinfo({currop}, all_identity)));
                ++ops_bound;
            }
            mfr[r] = opindex;
        }
        n().setup_mf_r_indexing(mfr);
    }


    //determine the total number of terms we get in the proudct of the children nodes.  
    template <typename node_type>
    static inline size_t number_mf_products(node_type& n)
    {
        auto& np = n.parent();
        size_t nprod = np().mf().size();

        for(size_t i = 0; i < np.size(); ++i)
        {
            if(i != n.child_id())
            {
                nprod *= np[i]().spf().size();
            }
        }
        return nprod;
    }

protected:
    //optimise this code to take into account the fact that we have a better storage of the r-index object.
    template <typename node_type>
    static void form_bpg(const node_type& n, size_t nterms, utils::bipartite_graph<auto_sop::opinfo, size_t >& bpg, std::map<std::pair<size_t, size_t>, std::vector<size_t>>& repeated_edges, size_t spf_skip = 0)
    {
        const auto& spf = n().spf();
        const auto& mf = n().mf();

        //and now that we have reexpressed the spf node info for the node.  We go through and compute the optimal bipartition to compact the spf info
        //go ahead and compute the bipartite bipartite graph decomposition
        bpg.resize(spf.size(), mf.size());

        for(size_t i = spf_skip; i < spf.size(); ++i)
        {
            CALL_AND_HANDLE
            (
                bpg.add_U(spf[i].m_opdef),
                "Failed to add spf operator to bipartite graph."
            );
            
        }
        for(const auto& v : mf)
        {
            CALL_AND_HANDLE
            (
                bpg.add_V(v.m_opdef),
                "Failed to add mf operator to bipartite graph."
            );
        }

        std::vector<size_t> mfr(nterms);    n().generate_mfr(mfr, nterms);

        //iterate over the spf terms
        for(size_t i = spf_skip; i < spf.size(); ++i)
        {
            //now add the element into the tree if it is non-zero
            for(size_t r : spf[i].m_inds)
            {
                size_t mfi = mfr[r];

                bool repeated_edge = !bpg.add_edge(i-spf_skip, mfi, r);
                //if we weren't able to add the edge because it is repeated then we simply
                if(repeated_edge)
                {
                    std::pair<size_t, size_t> m_inds({i, mfi});
                    repeated_edges[m_inds].push_back(r);
                }
            }
            
        }
    }

protected:
    template <typename Bgraph, typename node_type>
    static void optimise_spf_graph(const std::list<Bgraph>& sbpg, const std::map<std::pair<size_t, size_t>, std::vector<size_t>>& repeated_edges, const std::vector<literal::coeff<T>>& cr, node_type& n)
    {
        auto& spf = n().spf();
        size_t nterms = cr.size();
        std::vector<auto_sop::operator_data<T>> new_spf;     new_spf.reserve(spf.size());
        std::vector<size_t> spfr(nterms);    n().generate_spfr(spfr, nterms);
        std::vector<size_t> mfr(nterms);    n().generate_mfr(mfr, nterms);

        utils::term_indexing_array<size_t> rbound(nterms);
        std::list<std::pair<std::vector<size_t>, std::vector<size_t>> > vertex_covers;
        //iterate over each subgraph
        for(const auto& g : sbpg)
        {
            utils::bipartite_matching bpm(g);
            auto m = bpm.edges();

            //compute the minimum vertex cover of the edges - this allows us to construct the optimised spf operators.  
            //Namely for any node in the minimum vertex cover of the mean-field operators.  We can construct a optimised
            //SPF operator by combining all of the nodes present in the corresponding 
            auto vc = bpm.minimum_vertex_cover(g);
            vertex_covers.push_back(vc);
            auto& _U = std::get<0>(vc); auto& _V = std::get<1>(vc);

            //now for all of the nodes in _V (the mf terms) we can take all of the connected nodes and construct from these a sum operator 
            for(auto v : _V)
            {
                const auto& ve = g.V_edges(v);
                const auto& vd = g.V_edges_data(v);

                //get the r indices included in this sum
                utils::term_indexing_array<size_t> composite_rs(nterms);

                auto_sop::opinfo oinf;
                std::vector<literal::coeff<T>> accum_coeff;
                bool all_identity = true;

                for(auto z : common::zip(ve, vd))
                {
                    const auto& _u = std::get<0>(z);
                    const auto& _r = std::get<1>(z);

                    //at this stage we only include spf terms that are not in the minimum vertex cover on both sides.  If we 
                    //didn't do this then the optimisation of the MF operators becomes considerably more challenging.
                    if(std::find(_U.begin(), _U.end(), _u) == _U.end())
                    {
                        composite_rs.insert(_r);
                      
                        //now we also go through and flag any additional rs that were repeated in the map and added those to 
                        std::pair<size_t, size_t> m_inds({spfr[_r], mfr[_r]});

                        //now insert the current spf term into oinf.  
                        //if this term is a single term then the opdef object just has the first opdef object inserted and we get the accumulation coefficient from the matrix.
                        ASSERT(spf[spfr[_r]].nterms() == 1, "Something went wrong when optimising the SPF terms.  A composite SPF was encountered when simple SPFs were expected.");
                        oinf.push_back(spf[spfr[_r]].m_opdef.m_indices[0]);
                        all_identity = all_identity && spf[spfr[_r]].is_identity();

                        if(auto it = repeated_edges.find(m_inds); it != repeated_edges.end())
                        {
                            if(it->second.size() == 0)
                            {
                                accum_coeff.push_back(cr[_r]);
                            }
                            else{accum_coeff.push_back(literal::coeff<T>(T(1.0)));}
                            for(size_t rsi = 0; rsi < it->second.size(); ++rsi)
                            {
                                composite_rs.insert(it->second[rsi]);
                            }
                        }
                        else{accum_coeff.push_back(cr[_r]);}
                    }
                }

                oinf.m_is_identity = all_identity;

                if(accum_coeff.size() == 1){accum_coeff[0] = T(1.0);}
                new_spf.push_back(auto_sop::operator_data<T>(oinf, composite_rs, accum_coeff));

                utils::term_indexing_array<size_t> rtemp(nterms);
                utils::term_indexing_array<size_t>::set_union(rbound, composite_rs, rtemp);

                rbound = rtemp;
            }
        }
        
        //if we haven't bound all of the elements in the tree.  Then we want to go through and iterate over the spf operators 
        //in the minimum vertex cover.  That is all spf operators that have not been considered as part of a composite operator
        //or were not added as a composite operator (in the event that they are connected to a MF element of the minimum vertex 
        //cover but are themselves an element of the minimum vertex cover). We do this and simply add them back into the new 
        //spf operator list, with the only change being potentially removing any r-indices of theirs which 
        if(rbound.size() != nterms)
        {
            utils::term_indexing_array<size_t> r_remaining(nterms);    utils::term_indexing_array<size_t>::complement(rbound, r_remaining);

            for(auto z : common::zip(sbpg, vertex_covers))
            {
                const auto& g = std::get<0>(z);
                auto& vc = std::get<1>(z);
                auto& _U = std::get<0>(vc);

                //now for all of the nodes in _U (the spf terms) insert all of the terms that haven't been inserted 
                for(auto u : _U)
                {                
                    const auto& ud = g.U_edges_data(u);

                    //get the r indices included in this sum
                    utils::term_indexing_array<size_t> temp_rs(nterms);
                    utils::term_indexing_array<size_t> connected_rs(nterms);

                    for(const auto& r : ud)
                    {
                        connected_rs.insert(r);
                        temp_rs.insert(r);
                    }

                    for(const auto& r : connected_rs)
                    {
                        //now we also go through and flag any additional rs that were repeated in the map and added those to 
                        std::pair<size_t, size_t> m_inds({spfr[r], mfr[r]});
                        if(auto it = repeated_edges.find(m_inds); it != repeated_edges.end())
                        {
                            for(size_t rsi = 0; rsi < it->second.size(); ++rsi)
                            {
                                temp_rs.insert(it->second[rsi]);
                            }
                        }
                    }

                    //get the r indices that are present and have not been bound
                    utils::term_indexing_array<size_t>::set_intersection(r_remaining, temp_rs, connected_rs);
                    ASSERT
                    (
                        temp_rs.size() == connected_rs.size(), 
                        "Assertion failed these nodes should not have had any r indices bound."
                    );

                    std::vector<literal::coeff<T>> accum_coeff(1);  accum_coeff[0] = T(1.0);
                    new_spf.push_back(auto_sop::operator_data<T>(g.U(u), connected_rs, accum_coeff));

                    //now get the complement of the connected_rs
                    utils::term_indexing_array<size_t>::complement(connected_rs, temp_rs);

                    // and get the intersection of the remaing terms and this complement to remove the terms that were bound
                    utils::term_indexing_array<size_t>::set_intersection(r_remaining, temp_rs, connected_rs);
                    r_remaining = connected_rs;
                }
            }
            ASSERT
            (
                r_remaining.size() == 0, 
                "Critical error. Not all terms in the sum have been bound."
            );
        }
        spf.clear();
        spf = new_spf;
    }

protected:
    static void optimise_spf_operators(tree<auto_sop::node_op_info<T>>& bp, const std::vector<literal::coeff<T>>& cr)
    {
        sweeping::traversal_path euler_tour;
        sweeping::traversal_path::initialise_euler_tour(bp, euler_tour);

        size_t nterms = cr.size();
        INIT_TIMER;
        //now that we have the euler tour object setup we begin the forward tour to update the optimized SPF operators
        for(size_t id : euler_tour)
        {
            size_t times_visited = euler_tour.times_visited(id);
            euler_tour.visit(id);

            auto& n = bp[id];

            //as long as it isn't our last time accessing this node we need to update the mean field operators for it
            if(times_visited != (n.is_leaf() ? 1 : n.size()))
            {
                //if it is the first time we are visiting the node then we should compute the current mean-field bipartition for the node.
                //If we are at the root node the mf operator is trivial
                if(n.is_root())
                {
                    setup_root_node(n, nterms);
                }
                //if this isn't the root node then we need to actually work out the mf indexing objects.  
                else
                {
                    update_mf(n, nterms);
                }
            }
            //if it is our final time accessing this node throughout this tour, we go ahead and compute any composite operators that can 
            //be used to reduce the number of matrix element calculations
            else
            {
                //provided we aren't at a leaf node we start by recomputing the spf node info associated with this node.  This will use 
                //any optimised child representations that have been constructed in previous steps of the algorithm.
                if(!n.is_leaf())
                {
                    update_spf(n, nterms);
                }

                //and now that we have reexpressed the spf node info for the node.  We go through and compute the optimal bipartition 
                //to compact the spf info go ahead and compute the bipartite graph decomposition

                //The first step is to construct a bipartite graph from the spf and mf node info
                START_TIMER;
                utils::bipartite_graph<auto_sop::opinfo, size_t > bpg;
                std::map<std::pair<size_t, size_t>, std::vector<size_t>> repeated_edges;
                form_bpg(n, nterms, bpg, repeated_edges, 0);
                STOP_TIMER("Form BPG");


                //and iterate through each graph separating out each connected subgraph
                START_TIMER;
                std::list<decltype(bpg)> sbpg;
                decltype(bpg)::generate_connected_subgraphs(bpg, sbpg);
                STOP_TIMER("Form connected Subgraphs");

                //now that we have formed the connected sub graphs we no longer need the bpg
                bpg.clear();

                //now that we have the connected sub graphs we can iterate through each subgraph and work out minimum vertex cover of the 
                //graph which allows us to define an optimised set of single particle function operators
                START_TIMER;
                optimise_spf_graph(sbpg, repeated_edges, cr, n);
                STOP_TIMER("Optimised SPF representation");
            }
        }
    }

protected:
    template <typename Bgraph, typename node_type>
    static void optimise_mf_graph(const std::list<Bgraph>& sbpg, const std::map<std::pair<size_t, size_t>, std::vector<size_t>>& repeated_edges, const std::vector<literal::coeff<T>>& cr, size_t ncomposite_spf, node_type& n, bool reorder_mf = false)
    {
        auto& spf = n().spf();
        auto& mf = n().mf();

        size_t nterms = cr.size();

        std::vector<auto_sop::operator_data<T>> new_mf;     new_mf.reserve(spf.size());

        std::vector<size_t> spfr(nterms);    n().generate_spfr(spfr, nterms);
        std::vector<size_t> mfr(nterms);    n().generate_mfr(mfr, nterms);

        utils::term_indexing_array<size_t> rbound(nterms);
        std::list<std::pair<std::vector<size_t>, std::vector<size_t>> > vertex_covers;

        //first go through and add on the mf terms that come from the composite spf object.  
        //Here we skip inserting any terms that existed in the composite spf space as by definition these
        //terms cannot be combined with a composite mf term.  Here this is the case because we have 
        //at this stage not handled any terms that have nodes in both sides of the minimum vertex cover
        for(size_t i = 0; i < ncomposite_spf; ++i)
        {
            auto& spi = spf[i];
            bool mf_bound = false;
            utils::term_indexing_array<size_t> rintersect(nterms);

            for(size_t j = 0; j < mf.size() && !mf_bound; ++j)
            {
                auto& mfi = mf[j];
                utils::term_indexing_array<size_t>::set_intersection(spi.m_inds, mfi.m_inds, rintersect);

                if(rintersect.size() > 0 )
                {
                    std::vector<literal::coeff<T>> accum_coeff(1);  accum_coeff[0] = T(1.0);
                    new_mf.push_back(auto_sop::operator_data<T>(mfi.m_opdef, rintersect, accum_coeff));
                    mf_bound = true;

                    //now we necessarily require that the set intersect and spi.m_inds are the same
                    ASSERT(spi.m_inds.size() == rintersect.size(), "Error when adding mf term corresponding to a compressed spf term.");
                }
            }
            ASSERT(mf_bound, "Failed to bind mf corresponding to composite spf.  This should never happen.");

            utils::term_indexing_array<size_t> rtemp(nterms);
            utils::term_indexing_array<size_t>::set_union(rbound, rintersect, rtemp);
            rbound = rtemp;
        }
        ASSERT(new_mf.size() == ncomposite_spf, "We were unable to bind a mean field operator for each composite spf object.");

        //now if we haven't bound each term in the array we attempt to construct composite mean-field operators
        if(rbound.size() != nterms)
        {
            //iterate over each subgraph
            size_t sbpg_ind = 0;
            for(const auto& g : sbpg)
            {
                ++sbpg_ind;
                utils::bipartite_matching bpm(g);
                auto m = bpm.edges();

                //compute the minimum vertex cover of the edges - this allows us to construct the optimised mf operators.  
                //Namely for any node in the minimum vertex cover of the single particle operators.  We can construct a optimised
                //MF operator by combining all of the nodes present in the corresponding 
                auto vc = bpm.minimum_vertex_cover(g);
                vertex_covers.push_back(vc);
                auto& _U = std::get<0>(vc); auto& _V = std::get<1>(vc);

                //now for all of the nodes in _U (the spf terms) we can take all of the connected nodes and construct from these a sum operator 
                for(auto u : _U)
                {
                    const auto& ve = g.U_edges(u);
                    const auto& vd = g.U_edges_data(u);

                    auto_sop::opinfo oinf;
                    std::vector<literal::coeff<T>> accum_coeff;
                    bool all_identity = true;

                    //get the r indices included in this sum
                    utils::term_indexing_array<size_t> composite_rs(nterms);
                    for(auto z : common::zip(ve, vd))
                    {
                        const auto& _v = std::get<0>(z);
                        const auto& _r = std::get<1>(z);

                        //once again we will skip the terms that are in both minimum vertex covers
                        if(std::find(_V.begin(), _V.end(), _v) == _V.end())
                        {
                            composite_rs.insert(_r);

                            std::pair<size_t, size_t> m_inds({spfr[_r], mfr[_r]});
                            ASSERT(mf[mfr[_r]].nterms() == 1, "Something went wrong when optimising the MF terms.  A composite MF was encountered when simple MFs were expected.");
                            oinf.push_back(mf[mfr[_r]].m_opdef.m_indices[0]);
                            all_identity = all_identity && mf[mfr[_r]].is_identity();

                            //now we also go through and flag any additional rs that were repeated in the map and added those to 
                            if(auto it = repeated_edges.find(m_inds); it != repeated_edges.end())
                            {
                                if(it->second.size() == 0){accum_coeff.push_back(cr[_r]);}
                                else{accum_coeff.push_back(literal::coeff<T>(T(1.0)));}

                                for(size_t rsi = 0; rsi < it->second.size(); ++rsi)
                                {
                                    composite_rs.insert(it->second[rsi]);
                                }
                            }
                            else{accum_coeff.push_back(cr[_r]);}
                        }
                    }

                    //now whenever we add a new composite spf term into new_spf we need to expand the number of terms bound
                    oinf.m_is_identity = all_identity;
                    if(accum_coeff.size() == 1){accum_coeff[0] = T(1.0);}
                    new_mf.push_back(auto_sop::operator_data<T>(oinf, composite_rs, accum_coeff));
                    utils::term_indexing_array<size_t> rtemp(nterms);
                    utils::term_indexing_array<size_t>::set_union(rbound, composite_rs, rtemp);
                    rbound = rtemp;
                }
            }
        
            if(rbound.size() != nterms)
            {
                utils::term_indexing_array<size_t> r_remaining(nterms);    utils::term_indexing_array<size_t>::complement(rbound, r_remaining);
                for(auto z : common::zip(sbpg, vertex_covers))
                {
                    const auto& g = std::get<0>(z);
                    auto& vc = std::get<1>(z);
                    auto& _V = std::get<1>(vc);

                    //now for all of the nodes in _V (the mf terms) and insert all of the terms that haven't been inserted 
                    for(auto v : _V)
                    {                
                        const auto& ve = g.V_edges(v);
                        const auto& vd = g.V_edges_data(v);

                        //get the r indices included in this sum
                        utils::term_indexing_array<size_t> temp_rs(nterms);
                        utils::term_indexing_array<size_t> connected_rs(nterms);
                        for(auto z2 : common::zip(ve, vd))
                        {
                            const auto& _r = std::get<1>(z2);
                            connected_rs.insert(_r);
                            temp_rs.insert(_r);
                        }

                        for(const auto& r : connected_rs)
                        {
                            //now we also go through and flag any additional rs that were repeated in the map and added those to 
                            std::pair<size_t, size_t> m_inds({spfr[r], mfr[r]});
                            if(auto it = repeated_edges.find(m_inds); it != repeated_edges.end())
                            {
                                for(size_t rsi = 0; rsi < it->second.size(); ++rsi)
                                {
                                    temp_rs.insert(it->second[rsi]);
                                }
                            }
                        }
                        utils::term_indexing_array<size_t>::set_intersection(r_remaining, temp_rs, connected_rs);
                        
                        //if the intersection or r_remaining and temp_rs is less than the number of connected_rs this means that one
                        //of these terms has already been bound.  This will have been bound when handling composite operators
                        if(temp_rs.size() != connected_rs.size())
                        {
                            bool found_operator = false;
                            utils::term_indexing_array<size_t> rrem(connected_rs);
                            utils::term_indexing_array<size_t> rtemp(nterms);
                            for(size_t i = 0; i < ncomposite_spf; ++i)
                            {
                                if(new_mf[i].m_opdef == g.V(v))
                                {
                                    found_operator = true;
                                    utils::term_indexing_array<size_t>::set_union(new_mf[i].m_inds, rrem, rtemp);
                                    rrem = rtemp;
                                }
                            }
                            ASSERT(found_operator, "Error when binding residual terms.  There is no mf operator in the list accounting for the missing terms.");
                            ASSERT(rrem.size() == temp_rs.size(), "The remaining bound term does not account for the residual terms.");
                        }

                        std::vector<literal::coeff<T>> accum_coeff(1);  accum_coeff[0] = T(1.0);
                        new_mf.push_back(auto_sop::operator_data<T>(g.V(v), connected_rs, accum_coeff));

                        //now get the complement of the connected_rs
                        utils::term_indexing_array<size_t>::complement(connected_rs, temp_rs);

                        // and get the intersection of the remaing terms and this complement to remove the terms that were bound
                        utils::term_indexing_array<size_t>::set_intersection(r_remaining, temp_rs, connected_rs);
                        r_remaining = connected_rs;
                    }
                }
                ASSERT(r_remaining.size() == 0, "Critical error.  Not all terms in hamiltonian sum bound.");
            }
        }
        mf.clear();

        if(reorder_mf)
        {
            ASSERT(new_mf.size() == spf.size(), "Cannot pair mf and spf objects.  Incompatible sizes encountered.");

            //now we reordered the mf array so that the mean field terms and spf terms acting on the same sets of r indices have the same index in their
            //respective arrays
            mf.resize(new_mf.size());

            std::vector<bool> ind_bound(new_mf.size()); std::fill(ind_bound.begin(), ind_bound.end(), false);
            //iterate through all of the mean field iterators and 
            for(auto& m : new_mf)
            {
                bool bind_term = true;
                //iterater through the spf objects
                for(size_t i = 0; i < spf.size() && bind_term; ++i)
                {
                    //and if we haven't already bound this index
                    if(!ind_bound[i])
                    {
                        //check to see if these two objects correspond to the same index arrays
                        if(spf[i].m_inds == m.m_inds)
                        {
                            mf[i] = m;
                            ind_bound[i] = true;
                            bind_term = false;
                        }
                    }
                }
            }
            for(bool bound  : ind_bound)
            {
                ASSERT(bound, "Reached end of reordering MF without all terms being bound.");
            }
        }
        else
        {
            mf = new_mf;
        }
    }

    static void optimise_mf_operators(tree<auto_sop::node_op_info<T>>& bp, const std::vector<literal::coeff<T>>& cr)
    {
        INIT_TIMER;
        size_t nterms = cr.size();
        for(auto& n : bp)
        {
            //first update the mean field operators taking advantage of the newly optimised spf operators.
            if(!n.is_root())
            {
                update_mf(n, nterms);

                //now determine the total number of composite spf operators present in the object.  When forming the bipartite graph we will 
                //skip these elements
                size_t ncomposite_spf = 0;
                for(auto& sp : n().spf()){if(sp.m_opdef.size() > 1){++ncomposite_spf;}}

                //and now that we have reexpressed the spf node info for the node.  We go through and compute the optimal bipartition to compact 
                //the spf info go ahead and compute the bipartite graph decomposition

                //The first step is to construct a bipartite graph from the spf and mf node info
                START_TIMER;
                utils::bipartite_graph<auto_sop::opinfo, size_t > bpg;
                std::map<std::pair<size_t, size_t>, std::vector<size_t>> repeated_edges;
                form_bpg(n, nterms, bpg, repeated_edges, ncomposite_spf);
                STOP_TIMER("Form BPG");

                //and iterate through each graph separating out each connected subgraph
                START_TIMER;
                std::list<decltype(bpg)> sbpg;
                decltype(bpg)::generate_connected_subgraphs(bpg, sbpg);
                STOP_TIMER("Form connected Subgraphs");

                //now that we have formed the connected sub graphs we no longer need the bpg
                bpg.clear();

                //now that we have the connected sub graphs we can iterate through each subgraph and work out minimum vertex cover of the 
                //graph which allows us to define an optimised set of mean field operators
                START_TIMER;
                optimise_mf_graph(sbpg, repeated_edges, cr, ncomposite_spf, n, true);
                STOP_TIMER("Optimised MF representation");
            }
        }
    }

    static void set_coeffs(const std::vector<literal::coeff<T>>& cr, tree<auto_sop::node_op_info<T>>& bp)
    {
        //setup the accumulation coefficients for the spf objects.  This is done by iterating over all nodes in the tree in 
        //reverse order which ensures we update information about the child of a node before we attempt to update the node
        //itself
        for(auto& n : reverse(bp))
        {
            ASSERT(n().spf().size() == n().mf().size(), "Cannot setup hamiltonian coefficients the array sizes are incorrect.");

            n().coeff().resize(n().spf().size());
            //iterate through the nodes and set up the coefficient arrays correctly.  
            for(auto z : common::zip(n().spf(), n().mf(), n().coeff()))
            {
                auto& s = std::get<0>(z);
                auto& m = std::get<1>(z);
                auto& c = std::get<2>(z);

                c = T(1.0);

                if(s.m_accum_coeff.size() != s.m_opdef.size())
                {
                    ASSERT(s.m_opdef.size() == 1, "Invalid size.");
                    s.m_accum_coeff.resize(s.m_opdef.size());
                    std::fill(s.m_accum_coeff.begin(), s.m_accum_coeff.end(), T(1.0));
                }

                if(m.m_accum_coeff.size() != m.m_opdef.size())
                {
                    ASSERT(m.m_opdef.size() == 1, "Invalid size.");
                    m.m_accum_coeff.resize(m.m_opdef.size());   
                    std::fill(m.m_accum_coeff.begin(), m.m_accum_coeff.end(), 1.0);
                }

                if(m.m_accum_coeff.size() == 1 && s.m_accum_coeff.size() == 1)
                {
                    if(s.m_inds.size() == 1  && m.m_inds.size() == 1)
                    {
                        size_t r = *(s.m_inds.begin());
                        c = cr[r];
                    }
                }
            }
        }
    }

protected:
    template <typename node_type>
    static inline void update_spf_leaf_literal(const compressedSOP<T>& csop, node_type& n)
    {
        ASSERT(n.is_leaf(), "Failed to process spf data.  Node is not a leaf.")
        n().spf().clear();
        auto& spf = n().spf();
        spf.clear();
        size_t nu = n.leaf_index();

        //for the leaf term we set the r values for the spf to just be the r values for each term
        const auto& mode_ops = csop(nu);

        size_t identity_index = csop.identity_index(nu);
        size_t nterms = csop.nterms();

        spf.resize(nterms);
        size_t count = 0;
        for(const auto& op : mode_ops)
        {
            for(const auto& r : op.second)
            {
                utils::term_indexing_array<size_t> rinds(nterms);   rinds.insert(r);
                auto_sop::node_op_info<T> nop;
                spf[r] = auto_sop::operator_data<T>({{{count}}, (count == identity_index)}, rinds, {T(1.0)});
            }
            ++count;
        }
    }


    template <typename node_type>
    static inline void update_spf_literal(node_type& n, size_t nterms)
    {
        ASSERT(!n.is_leaf(), "Failed to process spf data.  Node is a leaf.")
        auto& spf = n().spf();
        spf.clear();
        spf.reserve(nterms);
        
        std::vector<size_t> currop(n.size());

        for(size_t r = 0; r <nterms; ++r)
        {
            bool all_identity = true;
            //build the child operator for this node 
            for(size_t ci = 0; ci < n.size(); ++ci)
            {
                currop[ci] = r;
                all_identity = all_identity && n[ci]().spf()[r].is_identity();
            }

            utils::term_indexing_array<size_t> rinds(nterms);   rinds.insert(r);
            spf.push_back(auto_sop::operator_data<T>({{currop}, all_identity}, rinds, {T(1.0)}));
        }
    }

    //setup the spf operator information. This function switches between union and array based implementations depending on the number of terms 
    //that will be treated in each term.  Doing so can dramatically increase the efficiency of the code.
    static inline void setup_spf_literal(const compressedSOP<T>& csop, tree<auto_sop::node_op_info<T>>& bp)
    {
        INIT_TIMER;
        const auto& cr = csop.coeff();
        for(auto& n : reverse(bp))
        {
            //if we are at a leaf node then all we need to do is copy the literal spf object info into this node
            if(n.is_leaf())
            {
                update_spf_leaf_literal(csop, n);
            }
            //if we aren't at a leaf node then we iterate through all of the children of this node and construct the composite operators that are the spf operators here
            else
            {
                update_spf_literal(n, csop.nterms());
            }
            auto& rcoeff = n().coeff();
            rcoeff.clear();
            rcoeff.resize(cr.size());
            for(size_t r = 0; r < cr.size(); ++r)
            {
                rcoeff[r] = cr[r];
            }
        }   
    }

    template <typename node_type>
    static void setup_root_node_literal(node_type& n, size_t nterms)
    {
        ASSERT(n.is_root(), "node is not root.  Cannot set it to be a root.");
        n().mf().clear();
        n().mf().reserve(nterms);
        for(size_t r = 0; r < nterms; ++r)
        {
            utils::term_indexing_array<size_t> rinds(nterms);   rinds.insert(r);
            n().mf().push_back(auto_sop::operator_data<T>(auto_sop::opinfo({{0}}, true), rinds, {T(1.0)}));
        }

    }


    template <typename node_type>
    static void update_mf_literal(node_type& n, size_t nterms)
    {
        auto& mf = n().mf();
        mf.clear();
        mf.reserve(nterms);

        auto& np = n.parent();
        std::vector<size_t> currop(np.size());
        
        for(size_t r = 0; r < nterms; ++r)
        {
            //add on the parent node term in the first index
            currop[0] = r;

            bool all_identity = np().mf()[r].is_identity();

            size_t sibling_index = 1;
            //now the sibling operator indices 
            for(size_t ci = 0; ci < np.size(); ++ci)
            {
                if(ci != n.child_id())
                {
                    currop[sibling_index] = r;
                    all_identity = all_identity && np[ci]().spf()[r].is_identity();
                    ++sibling_index;
                }
            }

            utils::term_indexing_array<size_t> rinds(nterms);   rinds.insert(r);
            mf.push_back(auto_sop::operator_data<T>({{currop}, all_identity}, rinds, {T(1.0)}));

        }
    }

    static inline void setup_mf_literal(tree<auto_sop::node_op_info<T>>& bp, size_t nterms)
    {
        INIT_TIMER;
        for(auto& n : bp)
        {
            //If we are at the root node the mf operator is trivial
            if(n.is_root())
            {
                setup_root_node_literal(n, nterms);
            }
            //if this isn't the root node then we need to actually work out the mf indexing objects.  
            else
            {
                update_mf_literal(n, nterms);
            }
        }   
    }



public:
    template <typename tree_type>
    static void compressed(const SOP<T>& sop, const tree_type& A, tree<auto_sop::node_op_info<T>>& bp, site_ops_type& site_ops)
    {
        //first ensure that the SOP object has the same dimensionality as the bp object.
        ASSERT(sop.nmodes() == A.nleaves(), "Failed to compute trivial tree bipartitioning of the SOP.  SOP and Tree do not have the same dimension.");

        INIT_TIMER;

        START_TIMER;
        ttns::compressedSOP<T> csop(sop);
        STOP_TIMER("Compressed SOP");

        auto coeff = csop.coeff();
        bp.clear();
        bp.construct_topology(A);
        
        //now we go through and initialise the trivial bipartitioning at each node.  To do this we want to have a tree containing the mode dimensions
        tree<std::vector<size_t>> mode_tree;    mode_tree.construct_topology(A);

        START_TIMER;
        setup_spf(csop, bp);
        STOP_TIMER("literal spf");

        //now extract the site operator types and clear the csop object as we no longer need it from this point
        site_ops = csop.site_operators();
        csop.clear();

        START_TIMER;
        optimise_spf_operators(bp, coeff);
        STOP_TIMER("optimised spf");

        START_TIMER;
        optimise_mf_operators(bp, coeff);
        STOP_TIMER("optimised mf");
      
        //now finally iterate through the tree and set up the coefficient arrays
        START_TIMER;
        set_coeffs(coeff, bp);
        STOP_TIMER("coefficients set");
    }

    template <typename ttn_type>
    static void literal(const SOP<T>& sop, const ttn_type& A, tree<auto_sop::node_op_info<T>>& bp, site_ops_type& site_ops)
    {
        //first ensure that the SOP object has the same dimensionality as the bp object.
        ASSERT(sop.nmodes() == A.nleaves(), "Failed to compute trivial tree bipartitioning of the SOP.  SOP and Tree do not have the same dimension.");
        bp.clear();
        bp.construct_topology(A);

        size_t nterms = sop.nterms();

        //now we go through and initialise the trivial bipartitioning at each node.  To do this we want to have a tree containing the mode dimensions
        INIT_TIMER;
        START_TIMER;
        ttns::compressedSOP<T> csop(sop);
        site_ops = csop.site_operators();

        STOP_TIMER("Compressed SOP");

        START_TIMER;
        setup_spf_literal(csop, bp);
        STOP_TIMER("literal spf");

        START_TIMER;
        setup_mf_literal(bp, nterms);
        STOP_TIMER("literal mf");
    }


public:
    template <typename tree_type>
    static bool construct(const SOP<T>& sop, const tree_type& A, tree<auto_sop::node_op_info<T>>& bp, site_ops_type& site_ops, bool compress = true)
    {
        if(sop.nterms() > 0)
        {
            
            //we only compress the term if it has at least two terms
            if(compress && sop.nterms() > 1)
            {
                autoSOP<T>::compressed(sop, A, bp, site_ops);
            }
            else
            {
                autoSOP<T>::literal(sop, A, bp, site_ops);
            }
            return true;
        }
        else
        {
            ASSERT(sop.nmodes() == A.nleaves(), "Failed to compute trivial tree bipartitioning of the SOP.  SOP and Tree do not have the same dimension.");
            bp.clear();
            bp.construct_topology(A);
            site_ops.resize(sop.nmodes());
        }
        return false;
    }
};
}   //namespace ttns


#endif  //TTNS_LIB_SOP_AUTOSOP_HPP


  
