#include <catch2/catch_test_macros.hpp>

#include <ttns_lib/ttn/tree/ntree.hpp>
#include <ttns_lib/ttn/tree/ntree_builder.hpp>

TEST_CASE("NTreeBuilderTucker", "[topology]")
{
    using namespace ttns;
    
    SECTION("We can build balanced trees.")
    {
        size_t nmodes = 16;
        std::vector<size_t> mode_dims(nmodes);
        for(size_t i = 0; i < nmodes; ++i){mode_dims[i] = i+1;} 

        SECTION("We can build a balanced binary tree with a fixed internal value.")
        {
            size_t degree = 2;
            size_t internal_value = 32;
            ntree<size_t> tree = ntree_builder<size_t>::htucker_tree(mode_dims, degree, internal_value);

            REQUIRE(tree.nleaves() == nmodes);
            REQUIRE(tree.size() == 47);

            {
                using iterator = typename ntree<size_t>::dfs_iterator;
                for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                {
                    if(!iter->is_leaf())
                    {
                        if(iter->is_root()){REQUIRE(iter->data() == 1);}
                        else{REQUIRE(internal_value == iter->data());}
                    }
                }
            }

            {
                using iterator = typename ntree<size_t>::leaf_iterator;
                size_t counter = 0;
                for(iterator iter = tree.leaf_begin(); iter != tree.leaf_end(); ++iter)
                {
                    REQUIRE(mode_dims[counter] == iter->data());
                    ++counter;
                }
            }
        }


        SECTION("We can build a balanced binary tree with a level dependent internal value.")
        {
            size_t degree = 2;
            ntree<size_t> tree = ntree_builder<size_t>::htucker_tree(mode_dims, degree, [](size_t l){return l+2;});

            REQUIRE(tree.nleaves() == nmodes);
            REQUIRE(tree.size() == 47);

            {
                using iterator = typename ntree<size_t>::dfs_iterator;
                for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                {
                    if(!iter->is_leaf())
                    {
                        if(iter->is_root()){REQUIRE(iter->data() == 1);}
                        else{REQUIRE(iter->level()+2 == iter->data());}
                    }
                }
            }

            {
                using iterator = typename ntree<size_t>::leaf_iterator;
                size_t counter = 0;
                for(iterator iter = tree.leaf_begin(); iter != tree.leaf_end(); ++iter)
                {
                    REQUIRE(mode_dims[counter] == iter->data());
                    ++counter;
                }
            }
        }

        SECTION("We can build a balanced ternary tree with a level dependent internal value.")
        {
            size_t degree = 3;
            ntree<size_t> tree = ntree_builder<size_t>::htucker_tree(mode_dims, degree, [](size_t l){return l+2;});

            REQUIRE(tree.nleaves() == nmodes);
            REQUIRE(tree.size() == 43);

            {
                using iterator = typename ntree<size_t>::dfs_iterator;
                for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                {
                    if(!iter->is_leaf())
                    {
                        if(iter->is_root()){REQUIRE(iter->data() == 1);}
                        else{REQUIRE(iter->level()+2 == iter->data());}
                    }
                }
            }

            {
                using iterator = typename ntree<size_t>::leaf_iterator;
                size_t counter = 0;
                for(iterator iter = tree.leaf_begin(); iter != tree.leaf_end(); ++iter)
                {
                    REQUIRE(mode_dims[counter] == iter->data());
                    ++counter;
                }
            }
        }
    }

    /* 
     * build the tree
     *          0
     *         / \
     *        /   \
     *       1     2
     *      / \    |
     *     3   4   5
     *        / \
     *       6   7
     * and add a hierarchical tucker subtree at node 5
     */
    SECTION("We can add balanced trees to other trees.")
    {
        ntree<size_t> tree;
        tree.insert(0);
        tree().insert(1);
        tree()[0].insert(3);
        tree()[0].insert(4);
        tree()[0][1].insert(6);
        tree()[0][1].insert(7);
        tree().insert(2);
        tree()[1].insert(5);

        size_t nmodes = 16;
        std::vector<size_t> mode_dims(nmodes);
        for(size_t i = 0; i < nmodes; ++i){mode_dims[i] = i+1;} 

        size_t degree = 2;
        ntree_builder<size_t>::htucker_subtree(tree.at({1, 0}), mode_dims, degree, [](size_t l){return l+2;});

        REQUIRE(tree.nleaves() == 19);
        REQUIRE(tree.size() == 54);

        {
            using iterator = typename ntree<size_t>::dfs_iterator;
            std::vector<size_t> traversal_order(8);
            traversal_order[0] = 0;
            traversal_order[1] = 1;
            traversal_order[2] = 3;
            traversal_order[3] = 4;
            traversal_order[4] = 6;
            traversal_order[5] = 7;
            traversal_order[6] = 2;
            traversal_order[7] = 5;

            size_t count = 0;
            for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
            {
                if(count < 8)
                {
                  REQUIRE(traversal_order[count] == iter->data());
                  ++count;
                }
                else
                {
                    if(!iter->is_leaf())
                    {
                        REQUIRE( (iter->level()-2) + 2 == iter->data());
                    }
                }
            }
        }
        {
            using leaf_iterator = typename ntree<size_t>::leaf_iterator;
            std::vector<size_t> lvals(3+nmodes);  
            lvals[0] = 3;
            lvals[1] = 6;
            lvals[2] = 7;
            for(size_t j=0; j<nmodes; ++j)
            {
                lvals[3+j] = mode_dims[j];
            }
            size_t count = 0;
            for(leaf_iterator iter = tree.leaf_begin(); iter != tree.leaf_end(); ++iter)
            {
                REQUIRE(lvals[count] == iter->data());
                ++count;
            }
        }
    }


    SECTION("We can sanitise a tree to get it ready for TTN based calculations.")
    {
        /* 
         * build the tree
         *          11
         *          |
         *          0
         *         / \
         *        /   \
         *       1     1000
         *      / \    |
         *     5  100  5
         *     |  / \
         *     3 7   8
         *     | |   |
         *     5 9   10
         *     |
         *     6
         * and sanitise it giving
         *          1
         *        _/|\_
         *       /  |  \
         *      3   56  5
         *     /   / \   \
         *    6   7   8   5
         *        |   |
         *        9   10
         */
        SECTION("Sanitising a tree removes bond matrices, straight line sections and enforces that the value stored at a site is no larger than its childen.")
        {
            ntree<size_t> tree;
            tree.insert(11);
            tree().insert(0);
            tree()[0].insert(1);
            tree()[0][0].insert(5);
            tree()[0][0][0].insert(3);
            tree()[0][0][0][0].insert(5);
            tree()[0][0][0][0][0].insert(6);
            tree()[0][0].insert(100);
            tree()[0][0][1].insert(7);
            tree()[0][0][1][0].insert(9);
            tree()[0][0][1].insert(8);
            tree()[0][0][1][1].insert(10);
            tree()[0].insert(1000);
            tree()[0][1].insert(5);

            ntree_builder<size_t>::sanitise_tree(tree);

            using iterator = typename ntree<size_t>::dfs_iterator;
            std::vector<size_t> traversal_order(10);
            traversal_order[0] = 1;
            traversal_order[1] = 3;
            traversal_order[2] = 6;
            traversal_order[3] = 56;
            traversal_order[4] = 7;
            traversal_order[5] = 9;
            traversal_order[6] = 8;
            traversal_order[7] = 10;
            traversal_order[8] = 5;
            traversal_order[9] = 5;

            size_t count = 0;
            for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
            {
                REQUIRE(traversal_order[count] == iter->data());
                ++count;
            }
        }

    }

}

/*  
TEST_CASE("NTreeBuilderMPS", "[topology]")
{
    using namespace ttns;
    
    SECTION("We can build MPS trees with local bases.")
    {
        size_t nmodes = 16;
        std::vector<size_t> mode_dims(nmodes);
        for(size_t i = 0; i < nmodes; ++i){mode_dims[i] = i+1;} 

        SECTION("We can build an mps tree with a fixed internal value and maximum size internal basis.")
        {
            size_t internal_value = 32;
            ntree<size_t> tree = ntree_builder<size_t>::mps_tree(mode_dims, internal_value);

            REQUIRE(tree.nleaves() == nmodes);
            REQUIRE(tree.size() == 47);

            {
                using iterator = typename ntree<size_t>::dfs_iterator;
                for(iterator iter = tree.begin(); iter != tree.end(); ++iter)
                {
                    if(!iter->is_leaf())
                    {
                        if(iter->is_root()){REQUIRE(iter->data() == 1);}
                        else{REQUIRE(internal_value == iter->data());}
                    }
                }
            }

            {
                using iterator = typename ntree<size_t>::leaf_iterator;
                size_t counter = 0;
                for(iterator iter = tree.leaf_begin(); iter != tree.leaf_end(); ++iter)
                {
                    REQUIRE(mode_dims[counter] == iter->data());
                    ++counter;
                }
            }
        }
    }

}*/

