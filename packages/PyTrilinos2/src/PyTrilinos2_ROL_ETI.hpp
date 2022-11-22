#ifndef PYTRILINOS2_ROL_ETI
#define PYTRILINOS2_ROL_ETI

#include "ROL_Vector.hpp"
//#include "ROL_Constraint.hpp"

#define BINDER_ROL_VECTOR(SCALAR) \
  template class Vector<SCALAR>;

namespace ROL {

  BINDER_ROL_VECTOR(double)

}

#endif // PYTRILINOS2_ROL_ETI
