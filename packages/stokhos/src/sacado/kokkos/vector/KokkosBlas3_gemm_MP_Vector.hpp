#ifndef KOKKOSBLAS3_GEMM_MP_VECTOR_HPP_
#define KOKKOSBLAS3_GEMM_MP_VECTOR_HPP_

template<class DA, class ... PA,
         class DB, class ... PB,
         class DC, class ... PC>
typename std::enable_if< Kokkos::is_view_mp_vector< Kokkos::View<DA,PA...> >::value &&
                         Kokkos::is_view_mp_vector< Kokkos::View<DB,PB...> >::value &&
                         Kokkos::is_view_mp_vector< Kokkos::View<DC,PC...> >::value >::type
KokkosBlas::gemm (const char transA[],
      const char transB[],
      typename Kokkos::View<DA,PA...>::const_value_type& alpha,
      const Kokkos::View<DA,PA...>& A,
      const Kokkos::View<DB,PB...>& B,
      typename Kokkos::View<DC,PC...>::const_value_type& beta,
      const Kokkos::View<DC,PC...>& C)
{
  // Assert that A, B, and C are in fact matrices
  static_assert (Kokkos::View<DA,PA...>::rank == 2, "GEMM: A must have rank 2 (be a matrix).");
  static_assert (Kokkos::View<DB,PB...>::rank == 2, "GEMM: B must have rank 2 (be a matrix).");
  static_assert (Kokkos::View<DC,PC...>::rank == 2, "GEMM: C must have rank 2 (be a matrix).");
  
  if (C.dimension_1 () == 1)
    auto x = Kokkos::subview (B, Kokkos::ALL, 0);
    auto y = Kokkos::subview (C, Kokkos::ALL, 0);
    KokkosBlas::gemv(transA,alpha,A,x,beta,y);
  else
    throw_error("GEMM");
}

#endif
