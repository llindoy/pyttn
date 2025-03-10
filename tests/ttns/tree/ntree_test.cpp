#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <string>

#include <ttns_lib/ttn/tree/ntree.hpp>

TEST_CASE("NTree", "[topology]")
{
    using namespace ttns;
    //first build an empty ntree object
    ntree<size_t> tree;
    SECTION("A tree can have a root inserted")
    {
        //and check that all of the operations work as expected given an empty tree
        REQUIRE(tree.empty());
        REQUIRE(tree.nleaves() == 0);
        REQUIRE(tree.size() == 0);
        
        std::vector<size_t> inds(1);    inds[0] = 0;
        //check that all of the node accessor methods throw an exception on an empty tree
        REQUIRE_THROWS(tree());
        REQUIRE_THROWS(tree[0]);
        REQUIRE_THROWS(tree.at(inds));
        REQUIRE_THROWS(tree.root());

        //check that the node functions work correctly
        tree.insert(0);
        //check that this has resized the tree
        REQUIRE(!tree.empty());
        REQUIRE(tree.nleaves() == 1);
        REQUIRE(tree.size() == 1);

        //check that accessing children of the root node through the tree still throws an exception
        REQUIRE_THROWS(tree[0]);
        REQUIRE_THROWS(tree.at(inds));

        //check that we can access the root node and get the data stored in it
        REQUIRE(tree.root()() == 0);
        REQUIRE(tree()() == 0);
        REQUIRE(tree().data() == 0);
        REQUIRE(tree().value() == 0);

        REQUIRE(&tree.root().tree() == &tree);

        REQUIRE(tree.root().level() == 0);
        REQUIRE(tree.root().nleaves() == 1);
        REQUIRE(tree.root().size() == 0);
        REQUIRE(tree.root().subtree_size() == 1);

        REQUIRE(tree.root().empty());
        REQUIRE(tree.root().is_root());
        REQUIRE(tree.root().is_leaf());

        REQUIRE_THROWS(tree.root().parent());
        REQUIRE_THROWS(tree.root().at(0));
        REQUIRE_THROWS(tree.root()[0]);
        REQUIRE_THROWS(tree.root().at(inds, 0));
        REQUIRE_THROWS(tree.root().back());
        REQUIRE_THROWS(tree.root().front());
    }

    std::vector<size_t> traversal_order(8);
    traversal_order[0] = 0;
    traversal_order[1] = 1;
    traversal_order[2] = 3;
    traversal_order[3] = 4;
    traversal_order[4] = 6;
    traversal_order[5] = 7;
    traversal_order[6] = 2;
    traversal_order[7] = 5;

    /* 
     * now insert several nodes building to construct the tree
     *          0
     *         / \
     *        /   \
     *       1     2
     *      / \    |
     *     3   4   5
     *        / \
     *       6   7
     * and query the properties of the tree to ensure it has been constructed correctly
     */
    SECTION("A tree can be built element by element")
    {
        tree.insert(0);
        tree().insert(1);
        tree()[0].insert(3);
        tree()[0].insert(4);
        tree()[0][1].insert(6);
        tree()[0][1].insert(7);
        tree().insert(2);
        tree()[1].insert(5);

        //check that this has resized the tree
        REQUIRE(!tree.empty());
        REQUIRE(tree.nleaves() == 4);
        REQUIRE(tree.size() == 8);

        SECTION("We can query properties of nodes in the tree")
        {
            //test that the node can correctly report its parent
            SECTION("Nodes can correctly report their parents.")
            {
                REQUIRE_THROWS(tree().parent());
                REQUIRE(&tree()[0].parent()==&tree());
            }

            SECTION("Nodes can correctly report their children.")
            {
                REQUIRE(tree()[0]() == 1);
                REQUIRE(tree().at(0)() == 1);
                REQUIRE(tree().at(0).at(1)() == 4);
                REQUIRE_THROWS(tree().at(0).at(0).at(0));
                REQUIRE_THROWS(tree().at(3));
                REQUIRE_THROWS(tree()[3]);
            }

            SECTION("We can access data stored in the nodes")
            {
                REQUIRE(tree()[0]() == 1);
                REQUIRE(tree()[0].value() == 1);
                REQUIRE(tree()[0].data() == 1);
            }

            SECTION("We can request the first and last child of a node")
            {
                REQUIRE(&(tree().front()) == &(tree()[0]));
                REQUIRE(&(tree().back()) == &(tree()[1]));

                REQUIRE_THROWS(tree()[0][0].front());
                REQUIRE_THROWS(tree()[0][0].back());
            }

            SECTION("We can iterate over the children of a node")
            {
                size_t c = 0;
                for(auto& n : tree())
                {
                    REQUIRE(&n == &tree()[c]);
                    ++c;
                }

                c = 0;
                for(auto& n : reverse(tree()))
                {
                    REQUIRE(&n == &tree()[1-c]);
                    ++c;
                }
            }

            SECTION("We can access elements in the tree specifying a traversal path")
            {
                //test the node at call
                REQUIRE(tree().at({0,1, 0}).is_leaf());
                REQUIRE(tree.at({0,1,0})() == 6);

                //test the tree at call
                REQUIRE(tree.at({0,1,0}).is_leaf());
                REQUIRE(tree.at({0,1,0})() == 6);
            }

            SECTION("Nodes are aware of the tree they belong to")
            {
                REQUIRE(&tree()[0][1][0].tree() == &tree);
            }

            SECTION("A node knows if it is the root or a leaf.")
            {
                REQUIRE(tree().is_root());
                REQUIRE(!tree()[0].is_root());

                REQUIRE(tree()[0][1][0].is_leaf());
                REQUIRE(!tree()[0][1].is_leaf());
            }

            SECTION("A node knows how many children it has.")
            {
                REQUIRE(tree()[0].size() == 2);
                REQUIRE(tree()[1].size() == 1);
                REQUIRE(tree()[1][0].size() == 0);
            }

            SECTION("A node knows if it has no children.")
            {
                REQUIRE(!tree()[0].empty());
                REQUIRE(!tree()[1].empty());
                REQUIRE(tree()[1][0].empty());
            }

            SECTION("A node knows how large the subtree it is the root of is.")
            {
                REQUIRE(tree()[0].subtree_size() == 5);
                REQUIRE(tree()[0][1].subtree_size() == 3);
                REQUIRE(tree()[0][1][0].subtree_size() == 1);
            }

            SECTION("A node knows how many leaves are in its subtree.")
            {
                REQUIRE(tree().nleaves() == 4);
                REQUIRE(tree()[0].nleaves() == 3);
                REQUIRE(tree()[0][1].nleaves() == 2);
                REQUIRE(tree()[0][1][0].nleaves() == 1);
            }

            SECTION("The node knows how far from the root it is.")
            {
                REQUIRE(tree().level() == 0);
                REQUIRE(tree()[0].level() == 1);
                REQUIRE(tree()[0][1].level() == 2);
                REQUIRE(tree()[0][1][0].level() == 3);
            }
        }   

        SECTION("We can use a pre-order DFS iterator")
        {
            using iterator = typename ntree<size_t>::dfs_iterator;

            size_t count = 0;
            for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
            {
                REQUIRE(traversal_order[count] == iter->data());
                ++count;
            }
        }

        SECTION("We can use a post-order DFS iterator")
        {
            using iterator = typename ntree<size_t>::post_iterator;
            std::vector<size_t> potraversal_order(8);
            potraversal_order[0] = 3;
            potraversal_order[1] = 6;
            potraversal_order[2] = 7;
            potraversal_order[3] = 4;
            potraversal_order[4] = 1;
            potraversal_order[5] = 5;
            potraversal_order[6] = 2;
            potraversal_order[7] = 0;

            size_t count = 0;
            for(iterator iter = tree.post_begin(); iter != tree.post_end(); ++iter)
            {
                REQUIRE(potraversal_order[count] == iter->data());
                ++count;
            }
        }

        SECTION("We can use a BFS iterator")
        {
            using iterator = typename ntree<size_t>::euler_iterator;
            std::vector<size_t> euler_order(19);
            euler_order[0] = 0;
            euler_order[1] = 1;
            euler_order[2] = 3;
            euler_order[3] = 3;
            euler_order[4] = 1;
            euler_order[5] = 4;
            euler_order[6] = 6;
            euler_order[7] = 6;
            euler_order[8] = 4;
            euler_order[9] = 7;
            euler_order[10] = 7;
            euler_order[11] = 4;
            euler_order[12] = 1;
            euler_order[13] = 0;
            euler_order[14] = 2;
            euler_order[15] = 5;
            euler_order[16] = 5;
            euler_order[17] = 2;
            euler_order[18] = 0;

            size_t count = 0;
            for(iterator iter = tree.euler_begin(); iter != tree.euler_end(); ++iter)
            {
                REQUIRE(euler_order[count] == iter->data());
                ++count;
            }
        }

        SECTION("We can iterate over the leaves of the tree")
        {
            using leaf_iterator = typename ntree<size_t>::leaf_iterator;
            std::vector<size_t> lvals(4);  
            lvals[0] = 3;
            lvals[1] = 6;
            lvals[2] = 7;
            lvals[3] = 5;
            size_t count = 0;
            for(leaf_iterator iter = tree.leaf_begin(); iter != tree.leaf_end(); ++iter)
            {
                REQUIRE(lvals[count] == iter->data());
                ++count;
            }
        }

        SECTION("We can insert at a specific position in the tree")
        {
            tree.insert_at({0, 1, 1}, 8);
            REQUIRE(!tree()[0][1][1].is_leaf());
            REQUIRE(tree()[0][1][1].size() == 1);
            REQUIRE(tree()[0][1][1][0]() == 8);
        }

        SECTION("Inserting in an invalid position throws an error")
        {
            REQUIRE_THROWS(tree.insert_at({3, 1, 1, 2, 5}, 8));
        }

        SECTION("We can remove children from a node.")
        {
            SECTION("We can remove children from the start of the node")
            {
                tree()[0][1].remove_child(0);
                REQUIRE(tree().subtree_size() == 7);
                REQUIRE(tree()[0].subtree_size() == 4);
                REQUIRE(tree()[0][1].subtree_size() == 2);
                REQUIRE(!tree()[0][1].is_leaf());

                REQUIRE(tree()[0][1][0]() == 7);

                tree()[0][1].remove_child(0);
                REQUIRE(tree().subtree_size() == 6);
                REQUIRE(tree()[0].subtree_size() == 3);
                REQUIRE(tree()[0][1].subtree_size() == 1);
                REQUIRE(tree()[0][1].is_leaf());
            }

            SECTION("We can remove children from the end of the node")
            {
                tree()[0][1].remove_child(1);
                REQUIRE(tree().subtree_size() == 7);
                REQUIRE(tree()[0].subtree_size() == 4);
                REQUIRE(tree()[0][1].subtree_size() == 2);
                REQUIRE(!tree()[0][1].is_leaf());

                REQUIRE(tree()[0][1][0]() == 6);

                tree()[0][1].remove_child(0);
                REQUIRE(tree().subtree_size() == 6);
                REQUIRE(tree()[0].subtree_size() == 3);
                REQUIRE(tree()[0][1].subtree_size() == 1);
                REQUIRE(tree()[0][1].is_leaf());
            }
        }

        SECTION("We can clear a tree leaving it empty")
        {
            tree.clear();

            //and check that all of the operations work as expected given an empty tree
            REQUIRE(tree.empty());
            REQUIRE(tree.nleaves() == 0);
            REQUIRE(tree.size() == 0);
            
            std::vector<size_t> inds(1);    inds[0] = 0;
            //check that all of the node accessor methods throw an exception on an empty tree
            REQUIRE_THROWS(tree());
            REQUIRE_THROWS(tree[0]);
            REQUIRE_THROWS(tree.at(inds));
            REQUIRE_THROWS(tree.root());
        }

        SECTION("We can clear a node in a tree deleting the subtree")
        {
            tree()[0][1].clear();
            REQUIRE(tree().subtree_size() == 6);
            REQUIRE(tree()[0].subtree_size() == 3);
            REQUIRE(tree()[0][1].subtree_size() == 1);
            REQUIRE(tree()[0][1].is_leaf());

            REQUIRE(tree().nleaves() == 3);
            REQUIRE(tree()[0].nleaves() == 2);
            REQUIRE(tree()[0][1].nleaves() == 1);
            REQUIRE(tree()[0][1].empty());
        }

        SECTION("We can print an ntree to a string")
        {
            std::string str;
            std::ostringstream oss;
            oss << tree();
            str = oss.str();
            std::string ref_str("(0(1(3)(4(6)(7)))(2(5)))");
            REQUIRE(str == ref_str);
        }
    }

    /* 
     * build the following tree from a string
     *          0
     *         / \
     *        /   \
     *       1     2
     *      / \    |
     *     3   4   5
     *        / \
     *       6   7
     * and query the properties of the tree to ensure it has been constructed correctly
     */
    SECTION("A tree can be built from a string")
    {
        std::string str("(0(1(3)(4(6)(7)))(2(5)))");
        tree.load(str);

        SECTION("We can use a pre-order DFS iterator")
        {
            using iterator = typename ntree<size_t>::dfs_iterator;

            size_t count = 0;
            for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
            {
                REQUIRE(traversal_order[count] == iter->data());
                ++count;
            }
        }

        SECTION("A tree can be construct from a string")
        {
            ntree<size_t> other(str);

            REQUIRE(!other.empty());
            REQUIRE(other.nleaves() == tree.nleaves());
            REQUIRE(other.size() == tree.size());

            using iterator = typename ntree<size_t>::dfs_iterator;

            size_t count = 0;
            for(iterator iter = other.begin(); iter != other.end(); ++iter)
            {
                REQUIRE(traversal_order[count] == iter->data());
                ++count;
            }
        }


        SECTION("A tree can be built from another tree")
        {
            ntree<size_t> other(tree);

            REQUIRE(!other.empty());
            REQUIRE(other.nleaves() == tree.nleaves());
            REQUIRE(other.size() == tree.size());

            using iterator = typename ntree<size_t>::dfs_iterator;

            size_t count = 0;
            for(iterator iter = other.begin(); iter != other.end(); ++iter)
            {
                REQUIRE(traversal_order[count] == iter->data());
                ++count;
            }
        }

        SECTION("A tree can be copy assigned from another tree")
        {
            ntree<size_t> other;    other=tree;

            REQUIRE(!other.empty());
            REQUIRE(other.nleaves() == tree.nleaves());
            REQUIRE(other.size() == tree.size());

            using iterator = typename ntree<size_t>::dfs_iterator;

            size_t count = 0;
            for(iterator iter = other.begin(); iter != other.end(); ++iter)
            {
                REQUIRE(traversal_order[count] == iter->data());
                ++count;
            }
        }
    }   

    /* 
     * build the following tree from a string
     *          0
     *         / \
     *        /   \
     *       1     2
     *      / \    |
     *     3   4   5
     *        / \
     *       6   7
     * and insert the tree 
     *
     *          8
     *         / \
     *        9   10
     * at node 5 to give the tree
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
    SECTION("A set of nodes can be inserted into a tree.")
    {
        std::string str("(0(1(3)(4(6)(7)))(2(5)))");
        tree.load(str);

        std::vector<size_t> new_traversal_order(11);
        new_traversal_order[0] = 0;
        new_traversal_order[1] = 1;
        new_traversal_order[2] = 3;
        new_traversal_order[3] = 4;
        new_traversal_order[4] = 6;
        new_traversal_order[5] = 7;
        new_traversal_order[6] = 2;
        new_traversal_order[7] = 5;
        new_traversal_order[8] = 8;
        new_traversal_order[9] = 9;
        new_traversal_order[10] = 10;

        std::string str_to_add("(8(9)(10)");
        ntree<size_t> tree_to_add(str_to_add);
        tree()[1][0].insert(tree_to_add.root());

        SECTION("We can use a pre-order DFS iterator")
        {
            using iterator = typename ntree<size_t>::dfs_iterator;

            size_t count = 0;
            for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
            {
                REQUIRE(new_traversal_order[count] == iter->data());
                ++count;
            }
        }
    }
}
