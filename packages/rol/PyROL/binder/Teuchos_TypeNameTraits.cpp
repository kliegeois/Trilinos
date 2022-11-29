#include <ROL_Elementwise_Reduce.hpp>
#include <ROL_Types.hpp>
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_DataAccess.hpp>
#include <Teuchos_ENull.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_FilteredIterator.hpp>
#include <Teuchos_ParameterEntry.hpp>
#include <Teuchos_ParameterEntryValidator.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ParameterListModifier.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_RCPNode.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_StringIndexedOrderedValueObjectContainer.hpp>
#include <Teuchos_TypeNameTraits.hpp>
#include <Teuchos_any.hpp>
#include <cwchar>
#include <deque>
#include <ios>
#include <iterator>
#include <locale>
#include <memory>
#include <ostream>
#include <sstream> // __str__
#include <streambuf>
#include <string>
#include <typeinfo>
#include <vector>

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <Teuchos_RCP.hpp>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, Teuchos::RCP<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(Teuchos::RCP<void>)
#endif

void bind_Teuchos_TypeNameTraits(std::function< pybind11::module &(std::string const &namespace_) > &M)
{

	def_Teuchos_functions(M("Teuchos"));
	// Teuchos::demangleName(const std::string &) file:Teuchos_TypeNameTraits.hpp line:77
	M("Teuchos").def("demangleName", (std::string (*)(const std::string &)) &Teuchos::demangleName, "Demangle a C++ name if valid.\n\n The name must have come from typeid(...).name() in order to be\n valid name to pass to this function.\n\n \n\n \n\nC++: Teuchos::demangleName(const std::string &) --> std::string", pybind11::arg("mangledName"));

	// Teuchos::typeName(const class Teuchos::any::placeholder &) file:Teuchos_TypeNameTraits.hpp line:115
	M("Teuchos").def("typeName", (std::string (*)(const class Teuchos::any::placeholder &)) &Teuchos::typeName<Teuchos::any::placeholder>, "C++: Teuchos::typeName(const class Teuchos::any::placeholder &) --> std::string", pybind11::arg("t"));

	// Teuchos::TestForException_incrThrowNumber() file: line:61
	M("Teuchos").def("TestForException_incrThrowNumber", (void (*)()) &Teuchos::TestForException_incrThrowNumber, "Increment the throw number.  \n\nC++: Teuchos::TestForException_incrThrowNumber() --> void");

	// Teuchos::TestForException_getThrowNumber() file: line:64
	M("Teuchos").def("TestForException_getThrowNumber", (int (*)()) &Teuchos::TestForException_getThrowNumber, "Increment the throw number.  \n\nC++: Teuchos::TestForException_getThrowNumber() --> int");

	// Teuchos::TestForException_break(const std::string &) file: line:68
	M("Teuchos").def("TestForException_break", (void (*)(const std::string &)) &Teuchos::TestForException_break, "The only purpose for this function is to set a breakpoint.\n    \n\n\nC++: Teuchos::TestForException_break(const std::string &) --> void", pybind11::arg("msg"));

	// Teuchos::TestForException_setEnableStacktrace(bool) file: line:72
	M("Teuchos").def("TestForException_setEnableStacktrace", (void (*)(bool)) &Teuchos::TestForException_setEnableStacktrace, "Set at runtime if stacktracing functionality is enabled when *\n    exceptions are thrown.  \n\n\nC++: Teuchos::TestForException_setEnableStacktrace(bool) --> void", pybind11::arg("enableStrackTrace"));

	// Teuchos::TestForException_getEnableStacktrace() file: line:76
	M("Teuchos").def("TestForException_getEnableStacktrace", (bool (*)()) &Teuchos::TestForException_getEnableStacktrace, "Get at runtime if stacktracing functionality is enabled when\n exceptions are thrown. \n\nC++: Teuchos::TestForException_getEnableStacktrace() --> bool");

	// Teuchos::TestForTermination_terminate(const std::string &) file: line:79
	M("Teuchos").def("TestForTermination_terminate", (void (*)(const std::string &)) &Teuchos::TestForTermination_terminate, "Prints the message to std::cerr and calls std::terminate. \n\nC++: Teuchos::TestForTermination_terminate(const std::string &) --> void", pybind11::arg("msg"));

}
