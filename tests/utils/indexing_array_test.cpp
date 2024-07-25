#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <string>
#include <complex>

#include <utils/term_indexing_array.hpp>

TEST_CASE("term indexing array", "[utils]")
{
    using namespace utils;
    
    SECTION("We can create and modify a term indexing array with few elements that does not store the complement array.")
    {
        size_t capacity = 100;
        term_indexing_array<size_t> r(capacity);
        term_indexing_array<size_t> rx(capacity);
        REQUIRE(r.size() == 0);
        REQUIRE(r.nelems() == 0);
        REQUIRE(r.capacity() == capacity);
        REQUIRE(!r.store_complement());
        REQUIRE(r == rx);

        std::vector<size_t> vals({0, 1, 2, 3, 4, 10, 15, 25, 50, 99});
        std::vector<size_t> valsc({1, 2, 3, 4, 10, 15, 25, 50, 99});
        std::vector<size_t> vals2({1, 2, 3, 4, 10, 15, 25, 30, 27, 22, 50, 99});
        std::vector<size_t> vals3({1, 2, 4, 15, 25, 30, 27, 22, 50, 99});

        SECTION("An empty indexing array contains no elements.")
        {
            REQUIRE(!r.contains(0));
            REQUIRE(!r.contains(10));
            REQUIRE(!r.contains(20));
            REQUIRE(!r.contains(capacity-1));
            REQUIRE(!r.contains(capacity+1));
        }

        SECTION("We can get a vector from an empty r index array but it is empty.")
        {
            auto v = r.get();
            REQUIRE(v.size() == 0);

            r.get(v);
            REQUIRE(v.size() == 0);

            REQUIRE(r.r().size() == 0);
        }


        SECTION("We can set an index array from a vector of elements.")
        {
            r.set(vals);
            REQUIRE(r != rx);
            rx.set(vals);
            REQUIRE(r == rx);

            REQUIRE(r.size() == vals.size());
            REQUIRE(r.capacity()==100);
            REQUIRE(!r.store_complement());
            for(size_t i = 0; i < vals.size(); ++i)
            {
                REQUIRE(r.contains(vals[i]));
            }

            REQUIRE(!r.contains(75));
            REQUIRE(!r.contains(5));

            std::sort(vals.begin(), vals.end());
            size_t counter = 0; 
            for(auto z : r)
            {
                REQUIRE(z == vals[counter]);    ++counter;
            }

            SECTION("We can clear a term indexing array to make it empty.")
            {
                r.clear();
                REQUIRE(r.size() == 0);
                REQUIRE(r.capacity()==100);
                REQUIRE(!r.store_complement());

                for(size_t i = 0; i < vals.size(); ++i)
                {
                    REQUIRE(!r.contains(vals[i]));
                }
            }
        }

        SECTION("We can insert elements into an index array term by term.")
        {
            for(auto v : valsc)
            {
                r.insert(v);
            }
            
            REQUIRE(r.size() == valsc.size());
            REQUIRE(r.capacity()==100);
            REQUIRE(!r.store_complement());
            for(size_t i = 0; i < valsc.size(); ++i)
            {
                REQUIRE(r.contains(valsc[i]));
            }

            std::sort(valsc.begin(), valsc.end());
            size_t counter = 0; 
            for(auto z : r)
            {
                REQUIRE(z == valsc[counter]);    ++counter;
                REQUIRE(counter <= valsc.size());
            }
            SECTION("Inserting elements outside of the bounds of the array throws an error")
            {
                REQUIRE_THROWS(r.insert(150));
                REQUIRE_THROWS(r.insert(100));
            }
        }

        SECTION("We can insert elements into an index array using iterators.")
        {
            r.insert(valsc.begin(), valsc.end());
            
            REQUIRE(r.size() == valsc.size());
            REQUIRE(r.capacity()==100);
            REQUIRE(!r.store_complement());
            for(size_t i = 0; i < valsc.size(); ++i)
            {
                REQUIRE(r.contains(valsc[i]));
            }
            
            std::sort(valsc.begin(), valsc.end());
            size_t counter = 0; 
            for(auto z : r)
            {
                REQUIRE(z == valsc[counter]);    ++counter;
                REQUIRE(counter <= valsc.size());
            }

            SECTION("We can insert an r indexing array into another r-indexing array")
            {
                term_indexing_array<size_t> r2(100);
                r2.insert(30);
                r2.insert(27);
                r2.insert(22);
                r2.insert(r);

                REQUIRE(r2.size() == vals2.size());
                REQUIRE(r2.capacity()==100);
                REQUIRE(!r2.store_complement());
                for(size_t i = 0; i < vals2.size(); ++i)
                {
                    REQUIRE(r2.contains(vals2[i]));
                }
            }
        }

        SECTION("We can construct an r index from an array and capacity")
        {
            term_indexing_array<size_t> r2(vals2, 100);

            REQUIRE(r2.size() == vals2.size());
            REQUIRE(r2.capacity()==100);
            REQUIRE(!r2.store_complement());
            for(size_t i = 0; i < vals2.size(); ++i)
            {
                REQUIRE(r2.contains(vals2[i]));
            }
        }

        term_indexing_array<size_t> r2(vals2, 100);

        SECTION("We can copy construct an r index array.")
        {
            term_indexing_array<size_t> r3(r2);
            REQUIRE(r3.capacity()==100);
            REQUIRE(!r3.store_complement());
            REQUIRE(r3.size() == vals2.size());
            for(size_t i = 0; i < vals2.size(); ++i)
            {
                REQUIRE(r3.contains(vals2[i]));
            }
        }

        SECTION("We can copy assign an r index array.")
        {
            term_indexing_array<size_t> r3(200);
            r3 = r2;
            REQUIRE(r3.size() == vals2.size());
            REQUIRE(r3.capacity()==100);
            REQUIRE(!r3.store_complement());
            for(size_t i = 0; i < vals2.size(); ++i)
            {
                REQUIRE(r3.contains(vals2[i]));
            }
        }

        SECTION("We can remove elements from an r index array")
        {
            r2.erase(3);
            REQUIRE(r2.size() == vals2.size() - 1);
            REQUIRE(r2.capacity() == 100);
            REQUIRE(!r2.store_complement());
            r2.erase(10);
            REQUIRE(r2.size() == vals2.size() - 2);
            REQUIRE(r2.capacity() == 100);
            REQUIRE(!r2.store_complement());

            for(size_t i = 0; i < vals3.size(); ++i)
            {
                REQUIRE(r2.contains(vals3[i]));
            }
        }
    }

    SECTION("We can create and modify a term indexing array with few elements that does stores the complement array.")
    {
        size_t capacity = 20;
        term_indexing_array<size_t> r(capacity);
        REQUIRE(r.size() == 0);
        REQUIRE(r.nelems() == 0);
        REQUIRE(r.capacity() == capacity);
        REQUIRE(!r.store_complement());

        std::vector<size_t> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16, 17, 13, 10, 15});
        std::vector<size_t> valsc({1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16, 17, 13, 10, 15});
        std::vector<size_t> vals2({1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16, 17, 13, 10, 15, 18, 14});
        std::vector<size_t> vals3({1, 2, 4, 5, 6, 7, 8, 9, 11, 16, 17, 13, 15});

        SECTION("An empty indexing array contains no elements.")
        {
            REQUIRE(!r.contains(0));
            REQUIRE(!r.contains(10));
            REQUIRE(!r.contains(20));
            REQUIRE(!r.contains(capacity-1));
            REQUIRE(!r.contains(capacity+1));
        }

        SECTION("We can get a vector from an empty r index array but it is empty.")
        {
            auto v = r.get();
            REQUIRE(v.size() == 0);

            r.get(v);
            REQUIRE(v.size() == 0);

            REQUIRE(r.r().size() == 0);
        }


        SECTION("We can set an index array from a vector of elements.")
        {
            term_indexing_array<size_t> rx(capacity);
            r.set(vals);
            rx.set(vals);
            REQUIRE(r.size() == vals.size());
            REQUIRE(r.capacity()==20);
            REQUIRE(r.store_complement());

            REQUIRE(r == rx);
            for(size_t i = 0; i < vals.size(); ++i)
            {
                REQUIRE(r.contains(vals[i]));
            }

            REQUIRE(!r.contains(75));

            std::sort(vals.begin(), vals.end());
            size_t counter = 0; 
            for(auto z : r)
            {
                REQUIRE(z == vals[counter]);    ++counter;
            }

            SECTION("We can clear a term indexing array to make it empty.")
            {
                r.clear();
                REQUIRE(r.size() == 0);
                REQUIRE(r.capacity()==20);
                REQUIRE(!r.store_complement());

                for(size_t i = 0; i < vals.size(); ++i)
                {
                    REQUIRE(!r.contains(vals[i]));
                }
            }
        }

        SECTION("We can insert elements into an index array term by term.")
        {
            for(auto v : valsc)
            {
                r.insert(v);
            }
            
            REQUIRE(r.size() == valsc.size());
            REQUIRE(r.capacity()==20);
            REQUIRE(r.store_complement());
            for(size_t i = 0; i < valsc.size(); ++i)
            {
                REQUIRE(r.contains(valsc[i]));
            }

            std::sort(valsc.begin(), valsc.end());

            size_t counter = 0;
            for(auto z : r)
            {
                REQUIRE(z == valsc[counter]);    ++counter;
                REQUIRE(counter <= valsc.size());
            }
            SECTION("Inserting elements outside of the bounds of the array throws an error")
            {
                REQUIRE_THROWS(r.insert(150));
                REQUIRE_THROWS(r.insert(100));
            }
        }

        SECTION("We can insert elements into an index array using iterators.")
        {
            r.insert(valsc.begin(), valsc.end());
            
            REQUIRE(r.size() == valsc.size());
            REQUIRE(r.capacity()==20);
            REQUIRE(r.store_complement());
            for(size_t i = 0; i < valsc.size(); ++i)
            {
                REQUIRE(r.contains(valsc[i]));
            }
            
            std::sort(valsc.begin(), valsc.end());
            size_t counter = 0; 
            for(auto z : r)
            {
                REQUIRE(z == valsc[counter]);    ++counter;
                REQUIRE(counter <= valsc.size());
            }

            SECTION("We can insert an r indexing array into another r-indexing array")
            {
                term_indexing_array<size_t> r2(20);
                r2.insert(18);
                r2.insert(14);
                r2.insert(r);

                REQUIRE(r2.size() == vals2.size());
                REQUIRE(r2.capacity()==20);
                REQUIRE(r2.store_complement());
                for(size_t i = 0; i < vals2.size(); ++i)
                {
                    REQUIRE(r2.contains(vals2[i]));
                }
            }
        }

        SECTION("We can construct an r index from an array and capacity")
        {
            term_indexing_array<size_t> r2(vals2, 20);

            REQUIRE(r2.size() == vals2.size());
            REQUIRE(r2.capacity()==20);
            REQUIRE(r2.store_complement());
            for(size_t i = 0; i < vals2.size(); ++i)
            {
                REQUIRE(r2.contains(vals2[i]));
            }
        }

        term_indexing_array<size_t> r2(vals2, 20);
        REQUIRE(r2.size() == vals2.size());
        REQUIRE(r2.capacity()==20);
        REQUIRE(r2.store_complement());

        SECTION("We can copy construct an r index array.")
        {
            term_indexing_array<size_t> r3(r2);
            REQUIRE(r3.size() == vals2.size());
            REQUIRE(r3.capacity()==20);
            REQUIRE(r3.store_complement());
            for(size_t i = 0; i < vals2.size(); ++i)
            {
                REQUIRE(r3.contains(vals2[i]));
            }
        }

        SECTION("We can copy assign an r index array.")
        {
            term_indexing_array<size_t> r3(200);
            r3 = r2;
            REQUIRE(r3.size() == vals2.size());
            REQUIRE(r3.capacity()==20);
            REQUIRE(r3.store_complement());
            for(size_t i = 0; i < vals2.size(); ++i)
            {
                REQUIRE(r3.contains(vals2[i]));
            }
        }
        SECTION("We can remove elements from an r index array")
        {
            r2.erase(3);
            REQUIRE(r2.size() == vals2.size() - 1);
            REQUIRE(r2.capacity() == 20);
            REQUIRE(r2.store_complement());
            r2.erase(10);
            REQUIRE(r2.size() == vals2.size() - 2);
            REQUIRE(r2.capacity() == 20);
            REQUIRE(r2.store_complement());

            for(size_t i = 0; i < vals3.size(); ++i)
            {
                REQUIRE(r2.contains(vals3[i]));
            }
            r2.erase(1);
            r2.erase(2);
            r2.erase(4);
            r2.erase(5);
            r2.erase(11);
            r2.erase(16);
            r2.erase(17);
            r2.erase(13);
            REQUIRE(r2.capacity() == 20);
            REQUIRE(r2.size() == vals2.size()-10);
            REQUIRE(!r2.store_complement());
        }
    }


    SECTION("We have an iterator for the term indexing arrays.")
    {
        SECTION("Iterating over array not storing the complement but including zero")
        {
            std::vector<size_t> vals({0, 1, 2, 3, 4, 10, 15, 25, 50, 99});
            size_t capacity = 100;
            term_indexing_array<size_t> r(vals, capacity);
            REQUIRE(!r.store_complement());

            size_t c = 0;
            for(size_t rv : r)
            {
                REQUIRE(rv == vals[c]); ++c;
            }
        }

        SECTION("Iterating over array not storing the complement but not including zero")
        {
            std::vector<size_t> vals({2, 3, 4, 10, 15, 25, 50, 99});
            size_t capacity = 100;
            term_indexing_array<size_t> r(vals, capacity);
            REQUIRE(!r.store_complement());

            size_t c = 0;
            for(size_t rv : r)
            {
                REQUIRE(rv == vals[c]); ++c;
            }
        }
        
        SECTION("Iterating over array storing the complement but including zero")
        {
            std::vector<size_t> vals({0, 1, 2, 3, 4, 6, 8, 9});
            size_t capacity = 10;
            term_indexing_array<size_t> r(vals, capacity);
            REQUIRE(r.store_complement());

            size_t c = 0;
            for(size_t rv : r)
            {
                REQUIRE(rv == vals[c]); ++c;
            }
        }

        SECTION("Iterating over array storing the complement but not including zero")
        {
            std::vector<size_t> vals({2, 3, 4, 6, 8, 9});
            size_t capacity = 10;
            term_indexing_array<size_t> r(vals, capacity);
            REQUIRE(r.store_complement());

            size_t c = 0;
            for(size_t rv : r)
            {
                REQUIRE(rv == vals[c]); ++c;
            }
        }
    }

    SECTION("We can compute unions and intersections of sets.")
    {
        SECTION("We can compute the union and intersection of two sets that are not storing the complement.")
        {
            std::vector<size_t> Uv({0, 1, 2, 3, 5});
            std::vector<size_t> Vv({1, 5, 6, 8, 9});


            term_indexing_array<size_t> U(Uv, 20);
            term_indexing_array<size_t> V(Vv, 20);

            SECTION("Compute the Intersection")
            {
                std::vector<size_t> resv({1, 5});
                term_indexing_array<size_t> res(20);

                term_indexing_array<size_t>::set_intersection(U, V, res);

                REQUIRE(res.capacity() == 20);
                REQUIRE(!res.store_complement());
                REQUIRE(res.size() == resv.size());
                for(const auto& v : resv)
                {
                    REQUIRE(res.contains(v));
                }
            }

            SECTION("Compute the Union")
            {
                std::vector<size_t> resv({0, 1, 2, 3, 5, 6, 8, 9});
                term_indexing_array<size_t> res(20);

                term_indexing_array<size_t>::set_union(U, V, res);

                REQUIRE(res.capacity() == 20);
                REQUIRE(!res.store_complement());
                REQUIRE(res.size() == resv.size());
                for(const auto& v : resv)
                {
                    REQUIRE(res.contains(v));
                }
            }
        }

        SECTION("We can compute the union and intersection of two sets where one stores the complement and the other does not.")
        {
            std::vector<size_t> Uv({0, 1, 2, 3, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
            std::vector<size_t> Vv({1, 5, 6, 8, 9});

            term_indexing_array<size_t> U(Uv, 20);
            term_indexing_array<size_t> V(Vv, 20);

            SECTION("Compute the Intersection")
            {
                std::vector<size_t> resv({1, 5});
                term_indexing_array<size_t> res(20);

                term_indexing_array<size_t>::set_intersection(U, V, res);

                REQUIRE(res.capacity() == 20);
                REQUIRE(!res.store_complement());
                REQUIRE(res.size() == resv.size());
                for(const auto& v : resv)
                {
                    REQUIRE(res.contains(v));
                }
            }

            SECTION("Compute the Union")
            {
                std::vector<size_t> resv({0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
                term_indexing_array<size_t> res(20);

                term_indexing_array<size_t>::set_union(U, V, res);

                REQUIRE(res.capacity() == 20);
                REQUIRE(res.store_complement());
                REQUIRE(res.size() == resv.size());
                for(const auto& v : resv)
                {
                    REQUIRE(res.contains(v));
                }
            }
        }

        //TODO: Check this test.
        SECTION("We can compute the union and intersection of two sets where both store the complement.")
        {
            std::vector<size_t> Uv({0, 1, 2, 3, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
            std::vector<size_t> Vv({1, 5, 6, 8, 9, 12, 13, 14, 16, 17, 19});

            term_indexing_array<size_t> U(Uv, 20);
            term_indexing_array<size_t> V(Vv, 20);

            SECTION("Compute the Intersection")
            {
                std::vector<size_t> resv({1, 5, 12, 13, 14, 16, 17, 19});
                term_indexing_array<size_t> res(20);

                term_indexing_array<size_t>::set_intersection(U, V, res);

                REQUIRE(res.capacity() == 20);
                REQUIRE(res.size() == resv.size());
                REQUIRE(!res.store_complement());
                for(const auto& v : resv)
                {
                    REQUIRE(res.contains(v));
                }
            }

            SECTION("Compute the Union")
            {
                std::vector<size_t> resv({0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
                term_indexing_array<size_t> res(20);

                term_indexing_array<size_t>::set_union(U, V, res);

                REQUIRE(res.capacity() == 20);
                REQUIRE(res.store_complement());
                REQUIRE(res.size() == resv.size());
                for(const auto& v : resv)
                {
                    REQUIRE(res.contains(v));
                }
            }
        }
    }

    SECTION("We can compute the complement of a set.")
    {
        std::vector<size_t> Uv({0, 1, 2, 3, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
        std::vector<size_t> Ucv({4, 6, 7, 8, 9});

        std::vector<size_t> Vv({1, 5, 6, 8, 9});
        std::vector<size_t> Vcv({0, 2, 3, 4, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
        
        term_indexing_array<size_t> U(Uv, 20);
        term_indexing_array<size_t> V(Vv, 20);

        SECTION("Inplace complement")
        {
            SECTION("Complement of vector storing complement.")
            {
                U.complement();
                REQUIRE(U.capacity() == 20);
                REQUIRE(U.size() == Ucv.size());
                for(size_t v : Uv){REQUIRE(!U.contains(v));}
                for(size_t v : Ucv){REQUIRE(U.contains(v));}
            }

            SECTION("Complement of vector not storing complement.")
            {
                V.complement();
                REQUIRE(V.capacity() == 20);
                REQUIRE(V.size() == Vcv.size());
                for(size_t v : Vv){REQUIRE(!V.contains(v));}
                for(size_t v : Vcv){REQUIRE(V.contains(v));}
            }
        }

        SECTION("out of place complement")
        {
            SECTION("Complement of vector storing complement.")
            {
                term_indexing_array<size_t> Uc;
                term_indexing_array<size_t>::complement(U, Uc);
                REQUIRE(Uc.capacity() == 20);
                REQUIRE(Uc.size() == Ucv.size());
                for(size_t v : Uv){REQUIRE(!Uc.contains(v));}
                for(size_t v : Ucv){REQUIRE(Uc.contains(v));}
            }

            SECTION("Complement of vector not storing complement.")
            {
                term_indexing_array<size_t> Vc;
                term_indexing_array<size_t>::complement(V, Vc);
                REQUIRE(Vc.capacity() == 20);
                REQUIRE(Vc.size() == Vcv.size());
                for(size_t v : Vv){REQUIRE(!Vc.contains(v));}
                for(size_t v : Vcv){REQUIRE(Vc.contains(v));}
            }
        }
    }
}
