/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2026 NPL Management Limited
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
#ifndef PYTHON_BINDING_ALGORITHMS_COMMON_BINDINGS_HPP
#define PYTHON_BINDING_ALGORITHMS_COMMON_BINDINGS_HPP

#include "../../common_bindings.hpp"


namespace python_bindings
{
    namespace docs
    {
        inline void add_AH_types(std::ostringstream& ss, const std::string& alg_name)
        {
            ss << ":param A: The Tree Tensor Network Object that will be optimised using the " + alg_name << " algorithm\n"
               << ":type A: ttn_complex\n"
               << ":param H: The Hamiltonian sop operator object\n"
               << ":type H: sop_operator_complex\n";
        }
        inline void add_init_types(std::ostringstream &ss, const std::string& alg_name, bool adaptive = false, bool tdvp = false)
        {
            add_AH_types(ss, alg_name);
            ss  << ":param krylov_dim: The krylov subspace dimension used for the eigensolver steps. (Default: 16)\n"
                << ":type krylov_dim: int, optional\n";
            if(tdvp)
            {
                ss << ":param nstep: The number of internal krylov steps to perform perTDVP step. (Default: 1)\n"
                   << ":type nstep: int, optional\n";
            }
            if(adaptive)
            {
                ss << ":param subspace_krylov_dim: The subspace expansion based krylov subspace dimension. This is only used if expansion=\"subspace\". (Default: 6)\n"
                   << ":type subspace_krylov_dim: int, optional\n"
                   << ":param subspace_neigs: The number of eigenvalues to evaluate when performing the subspace steps. This is only used if expansion=\"subspace\". (Default: 2)\n"
                   << ":type subspace_neigs: int, optional\n";
            }
            ss  << ":param num_threads: The number of openmp threads to be used for parallelising over the Hamiltonian sum in the solver. (Default: 1)\n"
                << ":type num_threads: int, optional\n"
                << ":param set_var_num_threads: The number of openmp threads to be used for parallelising over the set by the solver. (Default: 1)\n"
                << ":type set_var_num_threads: int, optional\n";
        }
        inline std::string constructor(const std::string& alg_name, bool adaptive = false, bool tdvp = false)
        {
            std::ostringstream ss;
            ss << "Construct a new one-site " << alg_name << " object initialising all buffers needed to perform "<< alg_name << " on a Tree Tensor Network A, with Hamiltonian H.\n\n";
            add_init_types(ss, alg_name, adaptive, tdvp);
            return ss.str();
        }

        inline std::string initialise(const std::string& alg_name, bool adaptive = false, bool tdvp = false)
        {
            std::ostringstream ss;
            ss << "Construct a new one-site " << alg_name << " object initialising all buffers needed to perform "<< alg_name << " on a Tree Tensor Network A, with Hamiltonian H.\n\n";
            add_init_types(ss, alg_name, adaptive, tdvp);
            return ss.str();
        }

        inline std::string step(const std::string& alg_name)
        {
            std::ostringstream ss;
            ss << "Performs a single step of the " << alg_name << " algorithm.\n";
            add_AH_types(ss, alg_name);
            ss << ":param update_env: Whether or not to force an update of all environment tensor at the start of the update scheme.  (Default: False) \n"
               << ":type update_env: bool, optional\n";
            return ss.str();
        }

        inline std::string prepare_environment(const std::string& alg_name)
        {
            std::ostringstream ss;
            ss << "Update all environment tensors required to perform a " << alg_name << " sweep. \n ";
            add_AH_types(ss, alg_name);
            ss << ":param update_env: Whether or not to force an update of all environment tensor at the start of the update scheme.  (Default: False) \n"
               << ":type update_env: bool, optional\n";
            return ss.str();
        }

        inline std::string class_doc(const std::string& alg_name)
        {
            return "A class implementing the " + alg_name + " algorithm on trees.";
        }
    }

    template<typename backend, typename Alg>
    void bind_sweeping_common(py::class_<Alg>& cls, const std::string& alg_name)
    {
        using _ttn = typename Alg::ttn_type;
        using _sop = typename Alg::env_type;
        using size_type = typename Alg::size_type;


        cls.def("clear", &Alg::clear, "Clear all internal buffers of the sweeping object.");
        cls.def("step", [](Alg &o, _ttn &A, _sop &sop, bool update_environment = false){ return o(A, sop, update_environment); }, 
            py::arg(), py::arg(), py::arg("update_env") = false, docs::step(alg_name).c_str());
        cls.def("__call__", [](Alg &o, _ttn &A, _sop &sop, bool update_environment = false){ return o(A, sop, update_environment); }, 
            py::arg(), py::arg(), py::arg("update_env") = false, docs::step(alg_name).c_str());
        cls.def("prepare_environment", &Alg::prepare_environment, py::arg(), py::arg(), py::arg("attempt_expansion") = false, docs::prepare_environment(alg_name).c_str());
        cls.def("backend", [](const Alg &){ return linalg::traits<backend>::label(); });
        cls.doc() = docs::class_doc(alg_name).c_str();
    }


}

#endif // PYTHON_BINDING_ALGORITHMS_COMMON_BINDINGS_HPP