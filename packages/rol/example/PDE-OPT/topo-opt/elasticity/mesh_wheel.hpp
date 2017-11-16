// @HEADER
// ************************************************************************
//
//               Rapid Optimization Library (ROL) Package
//                 Copyright (2014) Sandia Corporation
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
// Questions? Contact lead developers:
//              Drew Kouri   (dpkouri@sandia.gov) and
//              Denis Ridzal (dridzal@sandia.gov)
//
// ************************************************************************
// @HEADER

#include "../../TOOLS/meshmanager.hpp"

template <class Real>
class MeshManager_Wheel : public MeshManager_Rectangle<Real> {

private:

  int nx_;
  int ny_;
  Real width_;
  Teuchos::RCP<std::vector<std::vector<std::vector<int> > > >  meshSideSets_;

public: 

  MeshManager_Wheel(Teuchos::ParameterList &parlist) : MeshManager_Rectangle<Real>(parlist)
  {
    nx_ = parlist.sublist("Geometry").get("NX", 3);
    ny_ = parlist.sublist("Geometry").get("NY", 3);
    width_ = parlist.sublist("Geometry").get("Width", 2.0);
    computeSideSets();
  }


  void computeSideSets() {

    int numSideSets = 8;
    meshSideSets_ = Teuchos::rcp(new std::vector<std::vector<std::vector<int> > >(numSideSets));

    Real pf(0.125);
    Real patchFrac = (pf < width_) ? pf/width_ : pf;
    int np = static_cast<int>(patchFrac * static_cast<Real>(nx_));
    int nz = static_cast<int>(0.5*static_cast<Real>(nx_-4*np));
    int nc = nx_-2*(np+nz);

    // Bottom
    (*meshSideSets_)[0].resize(4);
    (*meshSideSets_)[0][0].resize(np);
    (*meshSideSets_)[0][1].resize(0);
    (*meshSideSets_)[0][2].resize(0);
    (*meshSideSets_)[0][3].resize(0);
    (*meshSideSets_)[1].resize(4);
    (*meshSideSets_)[1][0].resize(nz);
    (*meshSideSets_)[1][1].resize(0);
    (*meshSideSets_)[1][2].resize(0);
    (*meshSideSets_)[1][3].resize(0);
    (*meshSideSets_)[2].resize(4);
    (*meshSideSets_)[2][0].resize(nc);
    (*meshSideSets_)[2][1].resize(0);
    (*meshSideSets_)[2][2].resize(0);
    (*meshSideSets_)[2][3].resize(0);
    (*meshSideSets_)[3].resize(4);
    (*meshSideSets_)[3][0].resize(nz);
    (*meshSideSets_)[3][1].resize(0);
    (*meshSideSets_)[3][2].resize(0);
    (*meshSideSets_)[3][3].resize(0);
    (*meshSideSets_)[4].resize(4);
    (*meshSideSets_)[4][0].resize(np);
    (*meshSideSets_)[4][1].resize(0);
    (*meshSideSets_)[4][2].resize(0);
    (*meshSideSets_)[4][3].resize(0);
    // Right
    (*meshSideSets_)[5].resize(4);
    (*meshSideSets_)[5][0].resize(0);
    (*meshSideSets_)[5][1].resize(ny_);
    (*meshSideSets_)[5][2].resize(0);
    (*meshSideSets_)[5][3].resize(0);
    // Top
    (*meshSideSets_)[6].resize(4);
    (*meshSideSets_)[6][0].resize(0);
    (*meshSideSets_)[6][1].resize(0);
    (*meshSideSets_)[6][2].resize(nx_);
    (*meshSideSets_)[6][3].resize(0);
    // Left
    (*meshSideSets_)[7].resize(4);
    (*meshSideSets_)[7][0].resize(0);
    (*meshSideSets_)[7][1].resize(0);
    (*meshSideSets_)[7][2].resize(0);
    (*meshSideSets_)[7][3].resize(ny_);
    
    for (int i=0; i<np; ++i) {
      (*meshSideSets_)[0][0][i] = i;
    }
    for (int i=0; i<nz; ++i) {
      (*meshSideSets_)[1][0][i] = i + np;
    }
    for (int i=0; i<nc; ++i) {
      (*meshSideSets_)[2][0][i] = i + (np+nz);
    }
    for (int i=0; i<nz; ++i) {
      (*meshSideSets_)[3][0][i] = i + (np+nz+nc);
    }
    for (int i=0; i<np; ++i) {
      (*meshSideSets_)[4][0][i] = i + (np+nz+nc+nz);
    }
    for (int i=0; i<ny_; ++i) {
      (*meshSideSets_)[5][1][i] = (i+1)*nx_-1;
    }
    for (int i=0; i<nx_; ++i) {
      (*meshSideSets_)[6][2][i] = i + nx_*(ny_-1);
    }
    for (int i=0; i<ny_; ++i) {
      (*meshSideSets_)[7][3][i] = i*nx_;
    }

  } // computeSideSets

  Teuchos::RCP<std::vector<std::vector<std::vector<int> > > > getSideSets(
              const bool verbose = false,
              std::ostream & outStream = std::cout) const { 
    if ( verbose ) {
      outStream << "Mesh_TopoOpt: getSideSets called" << std::endl;
      outStream << "Mesh_TopoOpt: numSideSets = " << meshSideSets_->size() << std::endl;
    }
    return meshSideSets_;
  }
  
}; // MeshManager_TopoOpt
