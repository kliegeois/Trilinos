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

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_TestingHelpers.hpp"
#include "Teuchos_UnitTestRepository.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include "Stokhos_MP_Vector_MaskTraits.hpp"

TEUCHOS_UNIT_TEST( MP_Vector_Teuchos_BLAS, ROTG_8)
{
    constexpr int ensemble_size = 8;
    
    typedef Kokkos::DefaultExecutionSpace execution_space;
    typedef Stokhos::StaticFixedStorage<int,double,ensemble_size,execution_space> storage_type;
    typedef Sacado::MP::Vector<storage_type> scalar;
    typedef ScalarTraits<scalar> STS;
    typedef ScalarTraits<scalar::value_type> STV;
    
    scalar da, db, c, s;

    da = 1.589*STS::one();
    db = -3.685*STS::one();
    
    da[2] = 0.;
    db[2] = 0.;
 
    da[3] *= -1;
    db[3] *= -1;
    
    Teuchos::BLAS<int, scalar> blas;
    blas.ROTG(&da,&db,&c,&s);
    
    TEST_EQUALITY(STV::magnitude(da[0]+4.013)<0.001,1);
    TEST_EQUALITY(da[2],0.);
    EST_EQUALITY(STV::magnitude(da[3]-4.013)<0.001,1);
}

int main( int argc, char* argv[] ) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);

  Kokkos::HostSpace::execution_space::initialize();
  if (!Kokkos::DefaultExecutionSpace::is_initialized())
    Kokkos::DefaultExecutionSpace::initialize();

  int res = Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);

  Kokkos::HostSpace::execution_space::finalize();
  if (Kokkos::DefaultExecutionSpace::is_initialized())
    Kokkos::DefaultExecutionSpace::finalize();

  return res;
}
