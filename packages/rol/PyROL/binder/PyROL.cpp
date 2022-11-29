#include <map>
#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>

typedef std::function< pybind11::module & (std::string const &) > ModuleGetter;

void bind_std_postypes(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_std_typeinfo(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_std_locale_classes(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_std_istream_tcc(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_TypeNameTraits(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_any(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_ENull(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_RCPDecl(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_Ptr(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_1(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_ROL_Ptr(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_ROL_Types(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_ROL_Elementwise_Reduce(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_2(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_DataAccess(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_ScalarTraits(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_ParameterListExceptions(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_iostream_helpers(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_FancyOStream(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Teuchos_Dependency(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_3(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_4(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_5(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_6(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_7(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_8(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_9(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_10(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_11(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_12(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_13(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_14(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_15(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_16(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_17(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_18(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_19(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_20(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_21(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_22(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_23(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_24(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_25(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_26(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_27(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_28(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_29(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_30(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_31(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_32(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_33(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_unknown_unknown_34(std::function< pybind11::module &(std::string const &namespace_) > &M);


PYBIND11_MODULE(PyROL, root_module) {
	root_module.doc() = "PyROL module";

	std::map <std::string, pybind11::module> modules;
	ModuleGetter M = [&](std::string const &namespace_) -> pybind11::module & {
		auto it = modules.find(namespace_);
		if( it == modules.end() ) throw std::runtime_error("Attempt to access pybind11::module for namespace " + namespace_ + " before it was created!!!");
		return it->second;
	};

	modules[""] = root_module;

	static std::vector<std::string> const reserved_python_words {"nonlocal", "global", };

	auto mangle_namespace_name(
		[](std::string const &ns) -> std::string {
			if ( std::find(reserved_python_words.begin(), reserved_python_words.end(), ns) == reserved_python_words.end() ) return ns;
			else return ns+'_';
		}
	);

	std::vector< std::pair<std::string, std::string> > sub_modules {
		{"", "ROL"},
		{"ROL", "Elementwise"},
		{"ROL", "Exception"},
		{"ROL", "TRUtils"},
		{"ROL", "TypeB"},
		{"ROL", "TypeE"},
		{"ROL", "TypeG"},
		{"ROL", "TypeU"},
		{"", "Teuchos"},
		{"Teuchos", "Exceptions"},
		{"Teuchos", "PtrPrivateUtilityPack"},
		{"", "std"},
	};
	for(auto &p : sub_modules ) modules[p.first.size() ? p.first+"::"+p.second : p.second] = modules[p.first].def_submodule( mangle_namespace_name(p.second).c_str(), ("Bindings for " + p.first + "::" + p.second + " namespace").c_str() );

	//pybind11::class_<std::shared_ptr<void>>(M(""), "_encapsulated_data_");

	bind_std_postypes(M);
	bind_std_typeinfo(M);
	bind_std_locale_classes(M);
	bind_unknown_unknown(M);
	bind_std_istream_tcc(M);
	bind_Teuchos_TypeNameTraits(M);
	bind_Teuchos_any(M);
	bind_Teuchos_ENull(M);
	bind_Teuchos_RCPDecl(M);
	bind_Teuchos_Ptr(M);
	bind_unknown_unknown_1(M);
	bind_ROL_Ptr(M);
	bind_ROL_Types(M);
	bind_ROL_Elementwise_Reduce(M);
	bind_unknown_unknown_2(M);
	bind_Teuchos_DataAccess(M);
	bind_Teuchos_ScalarTraits(M);
	bind_Teuchos_ParameterListExceptions(M);
	bind_Teuchos_iostream_helpers(M);
	bind_Teuchos_FancyOStream(M);
	bind_Teuchos_Dependency(M);
	bind_unknown_unknown_3(M);
	bind_unknown_unknown_4(M);
	bind_unknown_unknown_5(M);
	bind_unknown_unknown_6(M);
	bind_unknown_unknown_7(M);
	bind_unknown_unknown_8(M);
	bind_unknown_unknown_9(M);
	bind_unknown_unknown_10(M);
	bind_unknown_unknown_11(M);
	bind_unknown_unknown_12(M);
	bind_unknown_unknown_13(M);
	bind_unknown_unknown_14(M);
	bind_unknown_unknown_15(M);
	bind_unknown_unknown_16(M);
	bind_unknown_unknown_17(M);
	bind_unknown_unknown_18(M);
	bind_unknown_unknown_19(M);
	bind_unknown_unknown_20(M);
	bind_unknown_unknown_21(M);
	bind_unknown_unknown_22(M);
	bind_unknown_unknown_23(M);
	bind_unknown_unknown_24(M);
	bind_unknown_unknown_25(M);
	bind_unknown_unknown_26(M);
	bind_unknown_unknown_27(M);
	bind_unknown_unknown_28(M);
	bind_unknown_unknown_29(M);
	bind_unknown_unknown_30(M);
	bind_unknown_unknown_31(M);
	bind_unknown_unknown_32(M);
	bind_unknown_unknown_33(M);
	bind_unknown_unknown_34(M);

}
