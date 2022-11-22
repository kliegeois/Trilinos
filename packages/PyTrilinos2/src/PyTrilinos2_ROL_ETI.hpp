#ifndef PYTRILINOS2_ROL_ETI
#define PYTRILINOS2_ROL_ETI

#include "ROL_Vector.hpp"
//#include "ROL_Constraint.hpp"

#define BINDER_ROL_VECTOR(SCALAR) \
  inline void initiate(Vector<SCALAR> p) {};

namespace ROL {

    template <typename T>
    void initiate(T) {};

  BINDER_ROL_VECTOR(double)

  using my_vector = Vector<double>;

}

#endif // PYTRILINOS2_ROL_ETI
