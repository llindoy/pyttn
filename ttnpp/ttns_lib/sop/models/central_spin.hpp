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

#ifndef PYTTN_TTNS_LIB_SOP_MODELS_CENTRAL_SPIN_HPP_
#define PYTTN_TTNS_LIB_SOP_MODELS_CENTRAL_SPIN_HPP_

#include "../SOP.hpp"
#include "model.hpp"

namespace ttns {

template <typename value_type>
class central_spin_base : public model<value_type> {
public:
  using model<value_type>::hamiltonian;
  using model<value_type>::system_info;

public:
  using real_type = typename linalg::get_real_type<value_type>::type;

  central_spin_base() : m_eps(0), m_delta(0), m_spin_index(0) {}
  central_spin_base(size_t spin_index, real_type eps, real_type delta, size_t N)
      : m_eps(eps), m_delta(delta), m_spin_index(spin_index), m_mode_dims(N) {}

  virtual ~central_spin_base() {}

  // functions for accessing the spin index
  size_t spin_index() const { return m_spin_index; }
  size_t &spin_index() { return m_spin_index; }

  // functions for accessing the bias term
  const real_type &eps() const { return m_eps; }
  real_type &eps() { return m_eps; }

  // functions for accessing the tunneling matirx element
  const real_type &delta() const { return m_delta; }
  real_type &delta() { return m_delta; }

  // functions for accessing the boson mode dimensions
  const std::vector<size_t> &mode_dims() const { return m_mode_dims; }
  std::vector<size_t> &mode_dims() { return m_mode_dims; }
  const size_t &mode_dim(size_t i) const {
    ASSERT(i < m_mode_dims.size(), "Failed to access mode dimensions.");
    return m_mode_dims[i];
  }
  size_t &mode_dim(size_t i) {
    ASSERT(i < m_mode_dims.size(), "Failed to access mode dimensions.");
    return m_mode_dims[i];
  }

  // functions for building the different sop representations of the Hamiltonian
  virtual void system_info(system_modes &sysinf) final {
    size_t N = m_mode_dims.size();
    sysinf.resize(N + 1);
    size_t ci = 0;
    for (size_t i = 0; i < N + 1; ++i) {
      if (i != m_spin_index) {
        sysinf[i] = spin_mode(m_mode_dims[ci]);
        ++ci;
      } else {
        sysinf[i] = spin_mode(2);
      }
    }
  }

protected:
  template <typename Hop> void build_system_op(Hop &H, real_type tol) {
    // add on the spin terms
    if (linalg::abs(m_eps) > tol) {
      H += m_eps * sOP("sz", this->m_spin_index);
    }
    if (std::abs(m_delta) > tol) {
      H += m_delta * sOP("sx", m_spin_index);
    }
  }

  real_type m_eps;
  real_type m_delta;
  size_t m_spin_index;
  std::vector<size_t> m_mode_dims;
};

// a class for handling the generation of the central_spin hamiltonian with
// homogeneous bath coupling
template <typename value_type>
class central_spin_homogeneous : public central_spin_base<value_type> {
public:
  using base_type = central_spin_base<value_type>;
  using real_type = typename linalg::get_real_type<value_type>::type;
  using model<value_type>::hamiltonian;
  using model<value_type>::system_info;

public:
  central_spin_homogeneous() {}
  central_spin_homogeneous(real_type eps, real_type delta,
                           const std::vector<real_type> &_w,
                           const std::vector<value_type> &_g)
      : base_type(0, eps, delta, _w.size()), m_w(_w),
        m_g(_g){ASSERT(_w.size() == _g.size(), "Invalid boson "
                                               "parameters"
                                               ".")} central_spin_homogeneous(
            size_t spin_index, real_type eps, real_type delta,
            const std::vector<real_type> &_A)
      : base_type(spin_index, eps, delta, _w.size()), m_w(_w), m_g(_g) {
    ASSERT(_w.size() == _g.size(), "Invalid boson parameters.")
  }
  virtual ~central_spin_homogeneous() {}

  // functions for accessing the boson frequencies
  const std::vector<real_type> &w() const { return m_w; }
  std::vector<real_type> &w() { return m_w; }

  const real_type &w(size_t i) const {
    ASSERT(i < m_w.size(), "Index out of bounds.");
    return m_w[i];
  }
  real_type &w(size_t i) {
    ASSERT(i < m_w.size(), "Index out of bounds.");
    return m_w[i];
  }

  // functions for accessing the boson couplings
  const std::vector<value_type> &g() const { return m_g; }
  std::vector<value_type> &g() { return m_g; }

  const value_type &g(size_t i) const {
    ASSERT(i < m_g.size(), "Index out of bounds.");
    return m_g[i];
  }
  value_type &g(size_t i) {
    ASSERT(i < m_g.size(), "Index out of bounds.");
    return m_g[i];
  }

  // functions for building the different sop representations of the Hamiltonian
  virtual void hamiltonian(sSOP<value_type> &H, real_type tol = 1e-14) final {
    H.clear();
    // H.resize(N+1);
    build_sop_repr(H, tol);
  }

  virtual void hamiltonian(SOP<value_type> &H, real_type tol = 1e-14) final {
    H.clear();
    H.resize(this->m_mode_dims.size() + 1);
    build_sop_repr(H, tol);
  }

protected:
  template <typename Hop> void build_sop_repr(Hop &H, real_type tol) {
    ASSERT(m_w.size() == m_g.size(), "Invalid bath parameters.");

    this->build_system_op(H, tol);

    // add on the terms containing bath operators.
    size_t counter = 0;
    for (size_t i = 0; i < m_g.size(); ++i) {
      if (i == this->m_spin_index) {
        ++counter;
      }
      H += (std::sqrt(2.0) * m_g[i]) * sOP("sz", this->m_spin_index) *
           sOP("q", counter); // write the Hamiltonian in terms of q =
                              // frac{1}{\sqrt(2)}(adag + a)
      H += m_w[i] * sOP("n", counter);
      ++counter;
    }
  }

protected:
  std::vector<real_type> m_w;
  std::vector<value_type> m_g;
};

} // namespace ttns

#endif // PYTTN_TTNS_LIB_SOP_MODELS_CENTRAL_SPIN_HPP_
