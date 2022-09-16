#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp>
#include <Tpetra_DistObject.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>

namespace Tpetra {
    using longlong = long long;
    using Kokkos_Compat_KokkosSerialWrapperNode = Kokkos::Compat::KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>;
}