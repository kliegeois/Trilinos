// @HEADER
// ***********************************************************************
//
//                           Stokhos Package
//                 Copyright (2009) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Eric T. Phipps (etphipp@sandia.gov).
//
// ***********************************************************************
// @HEADER

#ifndef KOKKOSBLAS2_GEMV_MP_VECTOR_HPP
#define KOKKOSBLAS2_GEMV_MP_VECTOR_HPP

#include <type_traits>
#include "Sacado_ConfigDefs.h"

#include "Stokhos_ViewStorage.hpp"
#include "Sacado_MP_Vector.hpp"
#include "Kokkos_View_MP_Vector.hpp"
#include "Kokkos_ArithTraits_MP_Vector.hpp"
#include "KokkosBlas.hpp"

#include "Stokhos_Sacado_Kokkos_MP_Vector.hpp"
#include "Kokkos_Core.hpp"

#include "Stokhos_config.h"

#define Sacado_MP_Vector_GEMV_Tile_Size(size) (STOKHOS_GEMV_CACHE_SIZE/size)

template <typename T>
KOKKOS_INLINE_FUNCTION void update_kernel(T *A, T alpha, T *b, T *c, int i_max)
{
  T alphab;
  alphab = alpha * b[0];

  for (size_t i = 0; i < i_max; ++i)
  {
    c[i] += alphab * A[i];
  }
}

template <typename T>
KOKKOS_INLINE_FUNCTION void inner_product_kernel(T *A, T *b, T *c)
{
  c[0] += b[0] * A[0];
}

template <class Scalar,
          class VA,
          class VX,
          class VY>
void update_MP(
    typename VA::const_value_type &alpha,
    const VA &A,
    const VX &x,
    typename VY::const_value_type &beta,
    const VY &y)
{
  // Get the dimensions
  const size_t m = y.extent(0);
  const size_t n = x.extent(0);

  const size_t N = Kokkos::DefaultExecutionSpace::impl_thread_pool_size();
  const size_t m_c_star = Sacado_MP_Vector_GEMV_Tile_Size(sizeof(Scalar));
  const size_t n_tiles_per_thread = ceil(((double)m) / (N * m_c_star));
  const size_t m_c = ceil(((double)m) / (N * n_tiles_per_thread));
  const size_t n_tiles = N * n_tiles_per_thread;

  Kokkos::parallel_for(n_tiles, KOKKOS_LAMBDA(const int i_tile) {
    size_t i_min = m_c * i_tile;
    bool last_tile = (i_tile == (n_tiles - 1));
    size_t i_max = (last_tile) ? m : (i_min + m_c);

#ifdef STOKHOS_HAVE_PRAGMA_UNROLL
#pragma unroll
#endif
    for (size_t i = i_min; i < i_max; ++i)
      y(i) = beta * y(i);

    for (size_t j = 0; j < n; ++j)
      update_kernel<Scalar>(&A(i_min, j), alpha, &x(j), &y(i_min), i_max - i_min);
  });
}

template <class Scalar,
          class VA,
          class VX,
          class VY>
void inner_products_MP(
    typename VA::const_value_type &alpha,
    const VA &A,
    const VX &x,
    typename VY::const_value_type &beta,
    const VY &y)
{
  // Get the dimensions
  const size_t m = y.extent(0);
  const size_t n = x.extent(0);

  const size_t team_size = STOKHOS_GEMV_TEAM_SIZE;

  const size_t N = Kokkos::DefaultExecutionSpace::impl_thread_pool_size();
  const size_t m_c_star = Sacado_MP_Vector_GEMV_Tile_Size(sizeof(Scalar));
  const size_t n_tiles_per_thread = ceil(((double)n) / (N * m_c_star));
  const size_t m_c = ceil(((double)n) / (N * n_tiles_per_thread));
  const size_t n_per_tile2 = m_c * team_size;

  const size_t n_i2 = ceil(((double)n) / n_per_tile2);
  using Kokkos::TeamThreadRange;

  Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(n_i2, team_size);
  typedef Kokkos::TeamPolicy<>::member_type member_type;

  Kokkos::parallel_for(m, KOKKOS_LAMBDA(const int i) {
    y(i) *= beta;
  });

  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(member_type team_member) {
    const size_t j = team_member.league_rank();
    const size_t j_min = n_per_tile2 * j;
    const size_t nj = (j_min + n_per_tile2 > n) ? (n - j_min) : n_per_tile2;
    const size_t i_min = j % m;

    for (size_t i = i_min; i < m; ++i)
    {
      Scalar tmp = 0.;
      Kokkos::parallel_reduce(TeamThreadRange(team_member, nj), [=](int jj, Scalar &tmp_sum) {
        inner_product_kernel<Scalar>(&A(jj + j_min, i), &x(jj + j_min), &tmp_sum);
      },
                              tmp);
      if (team_member.team_rank() == 0)
      {
        tmp *= alpha;
        Kokkos::atomic_add(&y(i), tmp);
      }
    }
    for (size_t i = 0; i < i_min; ++i)
    {
      Scalar tmp = 0.;
      Kokkos::parallel_reduce(TeamThreadRange(team_member, nj), [=](int jj, Scalar &tmp_sum) {
        inner_product_kernel<Scalar>(&A(jj + j_min, i), &x(jj + j_min), &tmp_sum);
      },
                              tmp);
      if (team_member.team_rank() == 0)
      {
        tmp *= alpha;
        Kokkos::atomic_add(&y(i), tmp);
      }
    }
  });
}

template <class Scalar,
          class VA,
          class VX,
          class VY>
void gemv_MP(const char trans[],
             typename VA::const_value_type &alpha,
             const VA &A,
             const VX &x,
             typename VY::const_value_type &beta,
             const VY &y)
{
  // y := alpha*A*x + beta*y,

  static_assert(VA::rank == 2, "GEMM: A must have rank 2 (be a matrix).");
  static_assert(VX::rank == 1, "GEMM: x must have rank 1 (be a vector).");
  static_assert(VY::rank == 1, "GEMM: y must have rank 1 (be a vector).");

  if (trans[0] == 'n' || trans[0] == 'N')
    update_MP<Scalar, VA, VX, VY>(alpha, A, x, beta, y);
  else
    inner_products_MP<Scalar, VA, VX, VY>(alpha, A, x, beta, y);
}

namespace KokkosBlas
{
template <typename DA, typename... PA,
          typename DX, typename... PX,
          typename DY, typename... PY>
typename std::enable_if<Kokkos::is_view_mp_vector<Kokkos::View<DA, PA...>>::value &&
                        Kokkos::is_view_mp_vector<Kokkos::View<DX, PX...>>::value &&
                        Kokkos::is_view_mp_vector<Kokkos::View<DY, PY...>>::value>::type
gemv(const char trans[],
     typename Kokkos::View<DA, PA...>::const_value_type &alpha,
     const Kokkos::View<DA, PA...> &A,
     const Kokkos::View<DX, PX...> &x,
     typename Kokkos::View<DY, PY...>::const_value_type &beta,
     const Kokkos::View<DY, PY...> &y)
{
  typedef typename Kokkos::View<DA, PA...>::value_type Scalar;
  typedef Kokkos::View<DA, PA...> VA;
  typedef Kokkos::View<DX, PX...> VX;
  typedef Kokkos::View<DY, PY...> VY;
  gemv_MP<Scalar, VA, VX, VY>(trans, alpha, A, x, beta, y);
}

} // namespace KokkosBlas
#endif
