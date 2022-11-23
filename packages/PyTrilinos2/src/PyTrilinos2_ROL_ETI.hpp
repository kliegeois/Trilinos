#ifndef PYTRILINOS2_ROL_ETI
#define PYTRILINOS2_ROL_ETI

#include "ROL_Vector.hpp"
#include "ROL_Objective.hpp"
#include "ROL_Constraint.hpp"
#include "ROL_Solver.hpp"
#include "ROL_Problem.hpp"

#define BINDER_ROL_VECTOR(SCALAR) \
  template class Vector<SCALAR>;

#define BINDER_ROL_OBJECTIVE(SCALAR) \
  template class Objective<SCALAR>;

#define BINDER_ROL_CONSTRAINT(SCALAR) \
  template class Constraint<SCALAR>;

#define BINDER_ROL_SOLVER(SCALAR) \
  template class Solver<SCALAR>;

#define BINDER_ROL_PROBLEM(SCALAR) \
  template class Problem<SCALAR>;

namespace ROL {

  BINDER_ROL_VECTOR(double)
  BINDER_ROL_OBJECTIVE(double)
  BINDER_ROL_CONSTRAINT(double)
  //BINDER_ROL_SOLVER(double)
  //BINDER_ROL_PROBLEM(double)

}

#endif // PYTRILINOS2_ROL_ETI
