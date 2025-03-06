
#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <string>

#include <ttns_lib/ttn/tree/tree.hpp>
#include <ttns_lib/ttn/ttn_nodes/node_traits/bool_node_traits.hpp>

TEST_CASE("tree", "[topology]")
{
    using namespace ttns;

    SECTION("We can build an empty tree")
    {
        tree<size_t> tree;
        REQUIRE(tree.nleaves() == 0);
        REQUIRE(tree.size() == 0);
        REQUIRE(tree.empty());

        SECTION("Accessing elements of an empty tree throws exceptions")
        {
            REQUIRE_THROWS(tree.at(0));
            REQUIRE_THROWS(tree({0,2,3}));
            REQUIRE_THROWS(tree.id_at({0,2,3}));

            std::vector<size_t> inds = {0, 2, 3};
            REQUIRE_THROWS(tree(inds.begin(), inds.end()));
        }
    }

    /* build the following tree using a ntree
     *          0
     *         / \
     *        /   \
     *       1     2
     *      / \     \
     *     3   4     5
     *        / \    |
     *       6   7   8
     *              / \
     *             9   10
     */
    std::string str("(0(1(3)(4(6)(7)))(2(5(8(9)(10)))))");
    ntree<size_t> topology(str);

    std::string str2("(0(1(3)(4(6)(7)))(2(5)))");
    ntree<size_t> topology2(str2);

    SECTION("We can construct/assign trees")
    {
        std::vector<size_t> traversal_order(11);
        traversal_order[0] = 0;
        traversal_order[1] = 1;
        traversal_order[2] = 3;
        traversal_order[3] = 4;
        traversal_order[4] = 6;
        traversal_order[5] = 7;
        traversal_order[6] = 2;
        traversal_order[7] = 5;
        traversal_order[8] = 8;
        traversal_order[9] = 9;
        traversal_order[10] = 10;

        SECTION("We can build a tree from a ntree.")
        {
            tree<size_t> tree(topology);

            SECTION("We can access elements of the tree")
            {
                REQUIRE(tree.root()() == 0);
            }

            REQUIRE(tree.nleaves() == topology.nleaves());
            REQUIRE(tree.size() == topology.size());
            REQUIRE(!tree.empty());


            SECTION("We can query properties of nodes in the tree")
            {
                SECTION("Nodes can correctly report their children.")
                {
                    REQUIRE(tree.root()[0]() == 1);
                    REQUIRE(tree.root().at(0)() == 1);
                    REQUIRE(tree.root().at(0).data() == 1);
                    REQUIRE(tree.root().at(0).value() == 1);
                    REQUIRE(tree.root().child(0)() == 1);
                    REQUIRE(tree.root().at(0).at(1)() == 4);
                    REQUIRE_THROWS(tree.root().at(0).at(0).at(0));
                    REQUIRE_THROWS(tree.root().at(3));
                    REQUIRE_THROWS(tree.root()[3]);
                }

                SECTION("We can access data stored in the nodes")
                {
                    REQUIRE(tree.root()[0]() == 1);
                    REQUIRE(tree.root()[0].value() == 1);
                    REQUIRE(tree.root()[0].data() == 1);
                }

                SECTION("We can request the first and last child of a node")
                {
                    REQUIRE(&(tree.root().front()) == &(tree.root()[0]));
                    REQUIRE(&(tree.root().back()) == &(tree.root()[1]));

                    REQUIRE_THROWS(tree.root()[0][0].front());
                    REQUIRE_THROWS(tree.root()[0][0].back());
                }

                SECTION("We can iterate over the children of a node")
                {
                    size_t c = 0;
                    for(auto& n : tree.root())
                    {
                        REQUIRE(n.child_id() == c);
                        REQUIRE(&n == &tree.root()[c]);
                        ++c;
                    }

                    c = 0;
                    for(auto& n : reverse(tree.root()))
                    {
                        REQUIRE(n.child_id() == 1-c);
                        REQUIRE(&n == &tree.root()[1-c]);
                        ++c;
                    }
                }

                SECTION("We can access elements in the tree specifying a traversal path")
                {
                    //test the node at call
                    REQUIRE(tree.root().at({0,1, 0}).is_leaf());
                    REQUIRE(tree.at({0,1,0})() == 6);

                    //test the tree at call
                    REQUIRE(tree.at({0,1,0}).is_leaf());
                    REQUIRE(tree.at({0,1,0})() == 6);
                }

                SECTION("A node knows if it is the root or a leaf.")
                {
                    REQUIRE(tree.root().is_root());
                    REQUIRE(!tree.root()[0].is_root());

                    REQUIRE(tree.root()[1][0][0][1].is_leaf());
                    REQUIRE(tree.root()[0][1][1].is_leaf());
                    REQUIRE(!tree.root()[0][1].is_leaf());
                    REQUIRE(tree.root()[0][0].is_leaf());
                }

                SECTION("A node knows how many children it has.")
                {
                    REQUIRE(tree.root()[0].size() == 2);
                    REQUIRE(tree.root()[1].size() == 1);
                    REQUIRE(tree.root()[1][0].size() == 1);
                }

                SECTION("A node knows if it has no children.")
                {
                    REQUIRE(!tree.root()[0].empty());
                    REQUIRE(!tree.root()[1].empty());
                    REQUIRE(!tree.root()[1][0].empty());
                    REQUIRE(tree.root()[1][0][0][1].empty());
                    REQUIRE(tree.root()[1][0][0][0].empty());
                    REQUIRE(tree.root()[0][0].empty());
                }

                SECTION("The node knows how far from the root it is.")
                {
                    REQUIRE(tree.root().level() == 0);
                    REQUIRE(tree.root()[0].level() == 1);
                    REQUIRE(tree.root()[0][1].level() == 2);
                    REQUIRE(tree.root()[0][1][0].level() == 3);
                }

            }   

            SECTION("The forward iterator finds the correct nodes in the tree.")
            {
                using iterator = typename tree<size_t>::iterator;
                size_t count = 0;
                size_t leaf_counter = 0;
                for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                {
                    REQUIRE(traversal_order[count] == iter->data());
                    REQUIRE(iter->id() == count);
                    auto index = iter->index();
                    REQUIRE(index.size() == iter->level());
                    REQUIRE(tree.at(index)() == iter->data());
                    REQUIRE(tree.id_at(index) == count);
                    REQUIRE(tree[count]() == iter->data());

                    if(!iter->is_leaf())
                    {
                        for(auto& c: iter->children())
                        {
                            REQUIRE(c->is_child_of(count));
                        }
                        REQUIRE_THROWS(iter->leaf_index());
                    }

                    if(iter->is_leaf())
                    {
                        REQUIRE(iter->leaf_index() == leaf_counter);
                        ++leaf_counter;
                    }

                    ++count;
                }
            }

            SECTION("The reverse iterator finds the correct nodes in the tree.")
            {
                using reverse_iterator = typename tree<size_t>::reverse_iterator;
                size_t count = 0;
                for(reverse_iterator iter = tree.rbegin(); iter != tree.rend(); ++iter)
                {
                    REQUIRE(traversal_order[11-(count+1)] == iter->data());
                    ++count;
                }
            }

            SECTION("We can clear a tree to create an empty tree")
            {
                tree.clear();

                REQUIRE(tree.nleaves() == 0);
                REQUIRE(tree.size() == 0);
                REQUIRE(tree.empty());

                SECTION("Accessing elements of an empty tree throws exceptions")
                {
                    REQUIRE_THROWS(tree.at(0));
                    REQUIRE_THROWS(tree({0,2,3}));
                    REQUIRE_THROWS(tree.id_at({0,2,3}));

                    std::vector<size_t> inds = {0, 2, 3};
                    REQUIRE_THROWS(tree(inds.begin(), inds.end()));
                }
            }
        }

        SECTION("We can assign a tree from a ntree.")
        {
            tree<size_t> tree;
            tree = topology;

            REQUIRE(tree.nleaves() == topology.nleaves());
            REQUIRE(tree.size() == topology.size());
            REQUIRE(!tree.empty());

            REQUIRE(has_same_structure(tree, tree));
            //now iterate through the tree and make sure it has the correct structure

            SECTION("The forward iterator finds the correct nodes in the tree.")
            {
                using iterator = typename tree<size_t>::iterator;
                size_t count = 0;
                for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                {
                    REQUIRE(traversal_order[count] == iter->data());
                    ++count;
                }
            }

            SECTION("We can reassign a tree")
            {
                tree = topology2;
                REQUIRE(tree.nleaves() == topology2.nleaves());
                REQUIRE(tree.size() == topology2.size());

                using iterator = typename tree<size_t>::iterator;
                size_t count = 0;
                for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                {
                    REQUIRE(traversal_order[count] == iter->data());
                    ++count;
                }
            }
        }

        SECTION("We can build a tree from a tree.")
        {
            tree<size_t> other(topology);
            tree<size_t> tree(other);

            REQUIRE(tree.nleaves() == topology.nleaves());
            REQUIRE(tree.size() == topology.size());
            REQUIRE(!tree.empty());

            SECTION("The forward iterator finds the correct nodes in the tree.")
            {
                using iterator = typename tree<size_t>::iterator;
                size_t count = 0;
                for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                {
                    REQUIRE(traversal_order[count] == iter->data());
                    ++count;
                }
            }

            SECTION("The reverse iterator finds the correct nodes in the tree.")
            {
                using reverse_iterator = typename tree<size_t>::reverse_iterator;
                size_t count = 0;
                for(reverse_iterator iter = tree.rbegin(); iter != tree.rend(); ++iter)
                {
                    REQUIRE(traversal_order[11-(count+1)] == iter->data());
                    ++count;
                }
            }
        }

        SECTION("We can assign a tree from a tree.")
        {
            tree<size_t> other(topology);
            tree<size_t> tree=other;

            REQUIRE(has_same_structure(tree, other));

            REQUIRE(tree.nleaves() == topology.nleaves());
            REQUIRE(tree.size() == topology.size());
            REQUIRE(!tree.empty());

            REQUIRE(has_same_structure(tree, tree));
            //now iterate through the tree and make sure it has the correct structure

            SECTION("The forward iterator finds the correct nodes in the tree.")
            {
                using iterator = typename tree<size_t>::iterator;
                size_t count = 0;
                for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                {
                    REQUIRE(traversal_order[count] == iter->data());
                    ++count;
                }
            }

            SECTION("We can reassign a tree")
            {
                other = topology2;
                tree = other;

                using iterator = typename tree<size_t>::iterator;
                size_t count = 0;
                for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                {
                    REQUIRE(traversal_order[count] == iter->data());
                    ++count;
                }
            }
        }
    }

    SECTION("We can resize/reallocate trees")
    {
        tree<size_t>  ref(topology);
        std::vector<size_t> traversal_order(11);
        traversal_order[0] = 0;
        traversal_order[1] = 0;
        traversal_order[2] = 0;
        traversal_order[3] = 0;
        traversal_order[4] = 0;
        traversal_order[5] = 0;
        traversal_order[6] = 0;
        traversal_order[7] = 0;
        traversal_order[8] = 0;
        traversal_order[9] = 0;
        traversal_order[10] = 0;

        SECTION("We can resize a tree from a ntree.")
        {
            tree<size_t> tree;     

            REQUIRE(!has_same_structure(tree, ref));

            tree.resize(topology);

            REQUIRE(has_same_structure(tree, ref));

            REQUIRE(tree.nleaves() == topology.nleaves());
            REQUIRE(tree.size() == topology.size());
            REQUIRE(!tree.empty());

            SECTION("The forward iterator finds the correct nodes in the tree.")
            {
                using iterator = typename tree<size_t>::iterator;
                size_t count = 0;
                for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                {
                    REQUIRE(traversal_order[count] == iter->data());
                    ++count;
                }
            }

            SECTION("We can resize a tree again")
            {
                tree.resize(topology2);
                REQUIRE(tree.nleaves() == topology2.nleaves());
                REQUIRE(tree.size() == topology2.size());
                REQUIRE(!tree.empty());

                SECTION("The forward iterator finds the correct nodes in the tree.")
                {
                    using iterator = typename tree<size_t>::iterator;
                    size_t count = 0;
                    for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                    {
                        REQUIRE(traversal_order[count] == iter->data());
                        ++count;
                    }
                }
            }
        }

        SECTION("We can resize a tree from a tree.")
        {
            tree<size_t> tree;     

            REQUIRE(!has_same_structure(tree, ref));

            tree.resize(ref);

            REQUIRE(has_same_structure(tree, ref));

            REQUIRE(tree.nleaves() == topology.nleaves());
            REQUIRE(tree.size() == topology.size());
            REQUIRE(!tree.empty());

            SECTION("The forward iterator finds the correct nodes in the tree.")
            {
                using iterator = typename tree<size_t>::iterator;
                size_t count = 0;
                for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                {
                    REQUIRE(traversal_order[count] == iter->data());
                    ++count;
                }
            }
        }

        SECTION("We can reallocate a tree from a ntree.")
        {
            tree<size_t> tree;     

            REQUIRE(!has_same_structure(tree, ref));

            tree.reallocate(topology);

            REQUIRE(has_same_structure(tree, ref));

            REQUIRE(tree.nleaves() == topology.nleaves());
            REQUIRE(tree.size() == topology.size());
            REQUIRE(!tree.empty());

            SECTION("The forward iterator finds the correct nodes in the tree.")
            {
                using iterator = typename tree<size_t>::iterator;
                size_t count = 0;
                for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                {
                    REQUIRE(traversal_order[count] == iter->data());
                    ++count;
                }
            }

            SECTION("We can reallocate a tree again")
            {
                tree.reallocate(topology2);
                REQUIRE(tree.nleaves() == topology2.nleaves());
                REQUIRE(tree.size() == topology2.size());
                REQUIRE(!tree.empty());

                SECTION("The forward iterator finds the correct nodes in the tree.")
                {
                    using iterator = typename tree<size_t>::iterator;
                    size_t count = 0;
                    for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                    {
                        REQUIRE(traversal_order[count] == iter->data());
                        ++count;
                    }
                }


            }
        }

        SECTION("We can reallocate a tree from a tree.")
        {
            tree<size_t> tree;     

            REQUIRE(!has_same_structure(tree, ref));

            tree.reallocate(ref);

            REQUIRE(has_same_structure(tree, ref));

            REQUIRE(tree.nleaves() == topology.nleaves());
            REQUIRE(tree.size() == topology.size());
            REQUIRE(!tree.empty());

            SECTION("The forward iterator finds the correct nodes in the tree.")
            {
                using iterator = typename tree<size_t>::iterator;
                size_t count = 0;
                for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                {
                    REQUIRE(traversal_order[count] == iter->data());
                    ++count;
                }
            }
        }

        SECTION("We can construct tree with a given topology from an ntree.")
        {
            tree<size_t> tree;     

            REQUIRE(!has_same_structure(tree, ref));

            tree.construct_topology(topology);

            REQUIRE(has_same_structure(tree, ref));

            REQUIRE(tree.nleaves() == topology.nleaves());
            REQUIRE(tree.size() == topology.size());
            REQUIRE(!tree.empty());

            SECTION("The forward iterator finds the correct nodes in the tree.")
            {
                using iterator = typename tree<size_t>::iterator;
                size_t count = 0;
                for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                {
                    REQUIRE(traversal_order[count] == iter->data());
                    ++count;
                }
            }

            SECTION("We can reallocate a tree again")
            {
                tree.construct_topology(topology2);

                REQUIRE(tree.nleaves() == topology2.nleaves());
                REQUIRE(tree.size() == topology2.size());
                REQUIRE(!tree.empty());

                SECTION("The forward iterator finds the correct nodes in the tree.")
                {
                    using iterator = typename tree<size_t>::iterator;
                    size_t count = 0;
                    for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                    {
                        REQUIRE(traversal_order[count] == iter->data());
                        ++count;
                    }
                }


            }
        }

        SECTION("We can construct tree with a given topology from an tree.")
        {
            tree<size_t> tree;     

            REQUIRE(!has_same_structure(tree, ref));

            tree.construct_topology(ref);

            REQUIRE(has_same_structure(tree, ref));

            REQUIRE(tree.nleaves() == topology.nleaves());
            REQUIRE(tree.size() == topology.size());
            REQUIRE(!tree.empty());

            SECTION("The forward iterator finds the correct nodes in the tree.")
            {
                using iterator = typename tree<size_t>::iterator;
                size_t count = 0;
                for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                {
                    REQUIRE(traversal_order[count] == iter->data());
                    ++count;
                }
            }
        }
    }
}
