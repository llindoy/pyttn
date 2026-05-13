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

#ifndef PYTTN_LINALG_UTILS_SERIALISATION_HPP_
#define PYTTN_LINALG_UTILS_SERIALISATION_HPP_

#ifdef CEREAL_LIBRARY_FOUND

#include <common/exception_handling.hpp>
#include "../linalg_forward_decl.hpp"

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/complex.hpp>
#include <cereal/details/helpers.hpp>

namespace linalg
{
    namespace internal
    {
        template <typename T, typename backend>
        struct buffer_writer_wrapper
        {
            using size_type = size_t;
            T *buf;
            size_type size;
            size_type cap;

            ~buffer_writer_wrapper();

            template <typename Archive>
            void save(Archive &archive) const;

            template <typename Archive>
            void load(Archive & /* archive */);
        };

        template <typename T, typename backend>
        struct buffer_reader_wrapper
        {
            using size_type = size_t;
            T **buf;
            size_type *size;
            size_type *cap;

            ~buffer_reader_wrapper();

            template <typename U>
            buffer_reader_wrapper &operator=(const U &other) = delete;
            template <typename U>
            buffer_reader_wrapper &operator=(U &&other) = delete;

            template <typename Archive>
            void save(Archive & /* archive */) const;

            template <typename Archive>
            void load(Archive &archive);
        };
    } // namespace internal


    namespace internal
    {
        template <typename T>
        struct buffer_writer_wrapper<T, blas_backend>
        {
            using size_type = size_t;
            T *buf;
            size_type size;
            size_type cap;

            ~buffer_writer_wrapper() { buf = nullptr; }

            template <typename Archive>
            void save(Archive &archive) const
            {
                //CALL_AND_HANDLE(archive(cereal::make_size_tag(size)), "Failed to serialize cpu buffer.  Failed to save capacity.");
                CALL_AND_HANDLE(archive(cereal::make_nvp("capacity", cap)), "Failed to serialise cpu buffer.  Failed to save size.");
                std::vector<T> val(buf, buf+size);
                CALL_AND_HANDLE(archive(cereal::make_nvp("data", val)), "Failed to serialise cpu buffer.  Failed to save size.");
                //for (size_type i = 0; i < size; ++i)
                //{
                //    archive(buf[i]);
                //}
            }

            template <typename Archive>
            void load(Archive & /* archive */) { RAISE_EXCEPTION("IF THIS COMES UP SOMETHING HAS GONE HORRIBLY WRONG."); }
        };

        template <typename T>
        struct buffer_reader_wrapper<T, blas_backend>
        {
            using size_type = size_t;
            T **buf;
            size_type *size;
            size_type *cap;

            ~buffer_reader_wrapper()
            {
                buf = nullptr;
                cap = nullptr;
                size = nullptr;
            }

            template <typename U>
            buffer_reader_wrapper &operator=(const U &other) = delete;
            template <typename U>
            buffer_reader_wrapper &operator=(U &&other) = delete;

            template <typename Archive>
            void save(Archive & /* archive */) const { RAISE_EXCEPTION("IF THIS COMES UP SOMETHING HAS GONE HORRIBLY WRONG."); }

            template <typename Archive>
            void load(Archive &archive)
            {    
                using allocator = memory::allocator<T, blas_backend>;
                size_type _cap;
                std::vector<T> val;
                //CALL_AND_HANDLE(archive(cereal::make_size_tag(s)), "Failed to deserialize cpu buffer.  Failed to read capacity.");
                CALL_AND_HANDLE(archive(cereal::make_nvp("capacity", _cap)), "Failed to deserialise cpu buffer.  Failed to read size.");
                CALL_AND_HANDLE(archive(cereal::make_nvp("data", val)), "Failed to deserialise cpu buffer.  Failed to read buffer.");

                if (buf == nullptr)
                {
                    CALL_AND_HANDLE(*buf = allocator::allocate(_cap), "Failed to deserialize cpu buffer.  Error when allocating new buffer to store result in.");
                }
                else
                {
                    if (_cap != *cap)
                    {
                        CALL_AND_HANDLE(allocator::deallocate(*buf), "Failed to deserialize cpu buffer.  Error when deallocating previously allocated buffer to overwrite.");
                        CALL_AND_HANDLE(*buf = allocator::allocate(_cap), "Failed to deserialize cpu buffer.  Error when allocating new buffer to store result in.");
                    }
                }
                //std::cerr << "load: " << s << " " << _cap << std::endl;
                ASSERT(val.size() <= _cap, "Failed to deserialize cpu buffer.  Size read in is smaller than capacity read in.")

                std::copy(val.begin(), val.end(), (*buf));
                //for (size_t i = 0; i < s; ++i)
                //{
                //    archive((*buf)[i]);
                //}
                *cap = _cap;
                *size = val.size();
            }
        };
    } // namespace internal
} // namespace linalg

#endif // CEREAL_LIBRARY_FOUND

#endif // PYTTN_LINALG_UTILS_SERIALISATION_HPP_//
