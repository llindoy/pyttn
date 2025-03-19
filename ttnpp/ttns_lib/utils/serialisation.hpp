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

#ifndef PYTTN_TTNS_LIB_UTILS_SERIALISATION_HPP_
#define PYTTN_TTNS_LIB_UTILS_SERIALISATION_HPP_

#include <linalg/utils/serialisation.hpp>

#ifdef CEREAL_LIBRARY_FOUND
template <typename node_type, typename size_type>
class serialisation_node_save_wrapper
{
protected:
    struct children_serialiser
    {
        node_type *const *buf;
        size_type cap;

        template <typename children_type>
        children_serialiser(const children_type &c) : buf(&c[0]), cap(c.size()) {}
        ~children_serialiser() { buf = nullptr; }

        template <typename Archive>
        void save(Archive &archive) const
        {
            archive(cereal::make_size_tag(cap));
            for (size_type i = 0; i < cap; ++i)
            {
                CALL_AND_HANDLE(archive(buf[i]->id()), "Failed to archive child id.");
            }
        }

        template <typename Archive>
        void load(Archive & /* ar */) { RAISE_EXCEPTION("IF THIS COMES UP SOMETHING HAS GONE HORRIBLY WRONG."); }
    };

    const node_type *m_node;

public:
    serialisation_node_save_wrapper() { m_node = nullptr; }
    ~serialisation_node_save_wrapper() { m_node = nullptr; }

    void initialise(const node_type *node)
    {
        m_node = node;
    }

    template <typename archive>
    void save(archive &ar) const
    {
        ASSERT(m_node != nullptr, "Failed to serialise node wrapper object.  No node has been wrapped.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("root", m_node->is_root())), "Failed to serialise node wrapper object.  Faild to serialise whether the node is the root of its tree.");
        if (!m_node->is_root())
        {
            CALL_AND_HANDLE(ar(cereal::make_nvp("parent", m_node->m_parent->id())), "Failed to serialise node wrapper object.  Failed to serialise the nodes parent.");
        }
        CALL_AND_HANDLE(ar(cereal::make_nvp("children", children_serialiser{m_node->children()})), "Failed to serialise tree_node_base object.  Failed to serialise the children nodes.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("node", *m_node)), "Failed to serialise node wrapper object.  Failed to serialise the node data.");
    }

    template <typename archive>
    void load(archive & /* ar */)
    {
        ASSERT(false, "If you have gotten here something has really broken.");
    }
};

template <typename node_type, typename size_type>
class serialisation_node_load_wrapper
{
    node_type m_node;
    std::vector<size_type> m_children_ids;
    size_type m_parent_id;
    bool is_root;

public:
    template <typename archive>
    void save(archive & /* ar */) const { ASSERT(false, "If you have gotten here something has really broken."); }

    template <typename archive>
    void load(archive &ar)
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("root", is_root)), "Failed to serialise node wrapper object.  Faild to serialise whether the node is the root of its tree.");
        if (!is_root)
        {
            CALL_AND_HANDLE(ar(cereal::make_nvp("parent", m_parent_id)), "Failed to serialise node wrapper object.  Failed to serialise the nodes parent.");
        }
        CALL_AND_HANDLE(ar(cereal::make_nvp("children", m_children_ids)), "Failed to serialise node wrapper object.  Failed to serialise the children nodes.");
        m_node.m_children.resize(m_children_ids.size());
        CALL_AND_HANDLE(ar(cereal::make_nvp("node", m_node)), "Failed to serialise node wrapper object.  Failed to serialise the node data.");
    }

    void move_to_node(node_type &set)
    {
        set = std::move(m_node);
    }
};
#endif

#endif // PYTTN_TTNS_LIB_UTILS_SERIALISATION_HPP_//
