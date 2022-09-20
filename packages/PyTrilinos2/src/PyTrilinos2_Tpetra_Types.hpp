#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp>
#include <Tpetra_DistObject.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <TpetraCore_ETIHelperMacros.h>

namespace Tpetra {
    using longlong = long long;
#if defined(HAVE_TPETRA_INST_SERIAL)
    using Kokkos_Compat_KokkosSerialWrapperNode = Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>;
#endif
#if defined(HAVE_TPETRA_INST_PTHREAD)
    using Kokkos_Compat_KokkosThreadsWrapperNode = Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Threads>;
#endif
#if defined(HAVE_TPETRA_INST_OPENMP)
    using Kokkos_Compat_KokkosOpenMPWrapperNode = Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::OpenMP>;
#endif
#if defined(HAVE_TPETRA_INST_CUDA)
    using Kokkos_Compat_KokkosCudaWrapperNode = Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Cuda>;
#endif
}