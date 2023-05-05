# -*- cmake -*-
# @HEADER
# ***********************************************************************
#
#          PyTrilinos: Python Interfaces to Trilinos Packages
#                 Copyright (2014) Sandia Corporation
#
# Under the terms of Contract DE-AC04-94AL85000 with Sandia
# Corporation, the U.S. Government retains certain rights in this
# software.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the Corporation nor the names of the
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Questions? Contact William F. Spotz (wfspotz@sandia.gov)
#
# ***********************************************************************
# @HEADER

#
# Global configure options
IF(TPL_ENABLE_MPI)
  SET(HAVE_MPI ON)
  IF(Mpi4Py_FOUND)
    SET(HAVE_MPI4PY ON)
  ENDIF()
ENDIF()

#
# Package-specific configuration options
IF(PyTrilinos_ENABLE_AztecOO)
  IF(AztecOO_ENABLE_Teuchos)
    SET(HAVE_PYTRILINOS_AZTECOO_TEUCHOS ON)
  ENDIF()
ENDIF()

IF(EpetraExt_BUILD_MODEL_EVALUATOR)
  SET(HAVE_PYTRILINOS_EPETRAEXT_MODEL_EVALUATOR ON)
ENDIF()

IF(PyTrilinos_ENABLE_NOX)
  IF(NOX_ENABLE_ABSTRACT_IMPLEMENTATION_EPETRA)
    SET(HAVE_PYTRILINOS_NOX_EPETRA ON)
  ENDIF()

  IF(NOX_ENABLE_EpetraExt)
    SET(HAVE_PYTRILINOS_NOX_EPETRAEXT ON)
  ENDIF()

  IF(NOX_ENABLE_ABSTRACT_IMPLEMENTATION_PETSC)
    SET(HAVE_PYTRILINOS_NOX_PETSC ON)
  ENDIF()

  IF(NOX_ENABLE_LOCA)
    SET(HAVE_PYTRILINOS_LOCA ON)
  ENDIF()
ENDIF()

IF(PyTrilinos_ENABLE_Tpetra)
  IF(Tpetra_INST_INT_LONG_LONG)
    SET(HAVE_PYTRILINOS_TPETRA_INST_INT_LONG_LONG ON)
  ENDIF()
ENDIF()

#
# Build the PyTrilinos configuration file
TRIBITS_CONFIGURE_FILE(PyTrilinos_config.h)

#
# On Mac OS X Gnu compilers, add dynamic lookup for undefined symbols
# to the pytrilinos library and PyTrilinos extension modules
SET(EXTRA_LINK_ARGS "${CMAKE_SHARED_LINKER_FLAGS}")
IF(APPLE)
  IF((CMAKE_CXX_COMPILER_ID MATCHES "GNU"  ) OR
     (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    SET(EXTRA_LINK_ARGS "${EXTRA_LINK_ARGS} -undefined dynamic_lookup")
  ENDIF()
ENDIF()

#
# Add the current binary and source directories, the python include
# path, the NumPy include path, the Mpi4Py include path, and the
# PETSc4Py include path to the include directories
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${NumPy_INCLUDE_DIR})
IF(TPL_ENABLE_MPI)
  IF(Mpi4Py_FOUND)
    INCLUDE_DIRECTORIES(${Mpi4Py_INCLUDE_DIR})
  ENDIF(Mpi4Py_FOUND)
ENDIF(TPL_ENABLE_MPI)
IF(NOX_ENABLE_PETSC)
  INCLUDE_DIRECTORIES(${PETSC4PY_INCLUDES})
ENDIF(NOX_ENABLE_PETSC)

#
# Copy the pyfragments.swg file to the build directory
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/pyfragments.swg.in
  ${CMAKE_CURRENT_BINARY_DIR}/pyfragments.swg)

#
# Initialize the headers and sources
SET(HEADERS "")
SET(SOURCES "")
SET(HEADERS ${HEADERS}
  ${CMAKE_CURRENT_BINARY_DIR}/PyTrilinos_config.h
  )

#
# Core PyTrilinos headers and sources
LIST(APPEND HEADERS
  PyTrilinos_PythonException.hpp
  PyTrilinos_FILEstream.hpp
  PyTrilinos_Util.hpp
  numpy_include.hpp
  PyTrilinos_NumPyImporter.hpp
  PyTrilinos_NumPy_Util.hpp
  PyTrilinos_DAP.hpp
  )

LIST(APPEND SOURCES
  PyTrilinos_PythonException.cpp
  PyTrilinos_FILEstream.cpp
  PyTrilinos_Util.cpp
  PyTrilinos_NumPyImporter.cpp
  PyTrilinos_NumPy_Util.cpp
  PyTrilinos_DAP.cpp
  )

#
# PyTrilinos + Teuchos headers and sources
LIST(APPEND HEADERS
  PyTrilinos_Teuchos_Util.hpp
  )

LIST(APPEND SOURCES
  PyTrilinos_Teuchos_Util.cpp
  )

#
# PyTrilinos + Epetra headers and sources
IF(PyTrilinos_ENABLE_Epetra)

  LIST(APPEND HEADERS
    PyTrilinos_Epetra_Util.hpp
    PyTrilinos_LinearProblem.hpp
    )

  LIST(APPEND SOURCES
    PyTrilinos_Epetra_Util.cpp
    PyTrilinos_LinearProblem.cpp
    )

ENDIF(PyTrilinos_ENABLE_Epetra)

#
# PyTrilinos + Tpetra headers and sources
IF(PyTrilinos_ENABLE_Tpetra)

  LIST(APPEND HEADERS
    PyTrilinos_Tpetra_Util.hpp
    )

  LIST(APPEND SOURCES
    PyTrilinos_Tpetra_Util.cpp
    )

ENDIF(PyTrilinos_ENABLE_Tpetra)

#
# PyTrilinos + Domi headers and sources
IF(PyTrilinos_ENABLE_Domi)

  LIST(APPEND HEADERS
    PyTrilinos_Domi_Util.hpp
    )

  LIST(APPEND SOURCES
    PyTrilinos_Domi_Util.cpp
    )

ENDIF(PyTrilinos_ENABLE_Domi)

#
# PyTrilinos + EpetraExt headers and sources
IF(PyTrilinos_ENABLE_EpetraExt)

  LIST(APPEND HEADERS
    PyTrilinos_EpetraExt_Util.hpp
    )

  LIST(APPEND SOURCES
    PyTrilinos_EpetraExt_Util.cpp
    )

ENDIF(PyTrilinos_ENABLE_EpetraExt)

#
# PyTrilinos + ML headers and sources
IF(PyTrilinos_ENABLE_ML)

  LIST(APPEND HEADERS
    PyTrilinos_ML_Util.hpp
    )

  LIST(APPEND SOURCES
    PyTrilinos_ML_Util.cpp
    )

ENDIF(PyTrilinos_ENABLE_ML)

#
# Define the target for the swig runtime header
SET(SWIG_RUNTIME swigpyrun.h)
ADD_CUSTOM_COMMAND(
  OUTPUT ${SWIG_RUNTIME}
  COMMAND ${SWIG_EXECUTABLE} -python -external-runtime
  )
LIST(APPEND HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${SWIG_RUNTIME})

#
# Define the targets for the PyTrilinos library
TRIBITS_ADD_LIBRARY(pytrilinos
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  )

# # Link against the Python library to prevent undefined
# # symbols for shared builds with -Wl,--no-undefined.
# TARGET_LINK_LIBRARIES(pytrilinos
#   ${PYTHON_LIBRARIES}
#   )

SET_TARGET_PROPERTIES(pytrilinos
  PROPERTIES LINK_FLAGS "${EXTRA_LINK_ARGS}"
  )

#
# Get the python version
EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c
                        "import sys; print(sys.version[:3])"
  OUTPUT_VARIABLE PYTHON_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

#
# Define the PyTrilinos swig setup
SET(CMAKE_SWIG_FLAGS "-w202")
IF(${SWIG_VERSION} VERSION_LESS 4.0)
  GET_FILENAME_COMPONENT(PYTRILINOS_DOXDIR
    ${CMAKE_CURRENT_BINARY_DIR}/../doc/Doxygen ABSOLUTE
    )
  LIST(APPEND CMAKE_SWIG_FLAGS -I${PYTRILINOS_DOXDIR})
  LIST(APPEND CMAKE_SWIG_FLAGS -noproxydel)
ELSE()
  LIST(APPEND CMAKE_SWIG_FLAGS -doxygen)
ENDIF()
LIST(APPEND CMAKE_SWIG_FLAGS -relativeimport)

#
# Determine the install directory
SET(PyTrilinos_INSTALL_DIR
  ${PyTrilinos_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}/site-packages/PyTrilinos
  CACHE PATH "The path where PyTrilinos will be installed"
  )
MESSAGE(STATUS "PyTrilinos installation path: ${PyTrilinos_INSTALL_DIR}")

#
# Add the PyTrilinos subdirectory, which is the top-level package
# directory.  This needs to be done before the following loop over the
# PyTrilinos modules, because of the call to GET_DIRECTORY_PROPERTY().
ADD_SUBDIRECTORY(PyTrilinos)

#
# Add the scripts subdirectory
ADD_SUBDIRECTORY(scripts)

#
# Initialize the list of additional clean files
SET(ADDITIONAL_CLEAN_FILES "")

#
# Set the Domi library name
IF(PyTrilinos_ENABLE_Domi)
  SET(DOMI_LIB domi)
ENDIF(PyTrilinos_ENABLE_Domi)

#
# Teuchos.RCP.i has to be generated. For this reason, it has to be
# handled separately
SET(TEUCHOS_RCP_SCRIPT gen_teuchos_rcp.py    )
SET(TEUCHOS_RCP        Teuchos.RCP.i         )
SET(TEUCHOS_RCP_TARGET PyTrilinos_Teuchos_RCP)
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/${TEUCHOS_RCP_SCRIPT}.in
  ${CMAKE_CURRENT_BINARY_DIR}/${TEUCHOS_RCP_SCRIPT})
ADD_CUSTOM_COMMAND(
  OUTPUT  "${TEUCHOS_RCP}"
  COMMAND "${PYTHON_EXECUTABLE}"
  ARGS    ${CMAKE_CURRENT_BINARY_DIR}/${TEUCHOS_RCP_SCRIPT} >
          ${CMAKE_CURRENT_BINARY_DIR}/${TEUCHOS_RCP}
  )
SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_BINARY_DIR}/${TEUCHOS_RCP} PROPERTIES
  CPLUSPLUS ON
  GENERATED ON
  )
SWIG_ADD_MODULE(${TEUCHOS_RCP_TARGET}
  python
  ${CMAKE_CURRENT_BINARY_DIR}/${TEUCHOS_RCP}
  "PyTrilinos/Teuchos"
  "RCP"
  )
SET_PROPERTY(TARGET ${TEUCHOS_RCP_TARGET} PROPERTY NO_SONAME 1)
SWIG_LINK_LIBRARIES(${TEUCHOS_RCP_TARGET} ${Trilinos_LIBRARIES})
SET_TARGET_PROPERTIES(${TEUCHOS_RCP_TARGET} PROPERTIES
  LINK_FLAGS "${EXTRA_LINK_ARGS}"
  SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON
  )
INSTALL(TARGETS ${TEUCHOS_RCP_TARGET}
  DESTINATION ${PyTrilinos_INSTALL_DIR}/Teuchos)

#
# Loop over the PyTrilinos-enabled Trilinos modules and define the
# swig-generated PyTrilinos extension modules
FOREACH(MODULE ${PyTrilinos_MODULES})
  # Alternate versions of the module name
  STRING(TOUPPER ${MODULE} MODULE_UPPER)
  STRING(REPLACE "." "_" MODULE_NODOT ${MODULE})
  SET_SOURCE_FILES_PROPERTIES(${MODULE}.i PROPERTIES
    CPLUSPLUS ON
    )

  # Determine the swig module output directory, name and unique target name
  SWIG_MODULE_GET_OUTDIR_AND_MODULE(${MODULE}.i MODULE_OUTDIR MODULE_NAME)

  # Define the target for this module
  SET(PyTrilinos_TARGET_NAME "PyTrilinos_${MODULE_NODOT}")
  IF("${MODULE_NAME}" MATCHES "Teuchos")
    SWIG_ADD_MODULE(${PyTrilinos_TARGET_NAME}
      python
      "${MODULE}.i"
      "${MODULE_OUTDIR}/${MODULE_NAME}"
      "${MODULE_NAME}")
  ELSE("${MODULE_NAME}" MATCHES "Teuchos")
    SWIG_ADD_MODULE(${PyTrilinos_TARGET_NAME}
      python
      "${MODULE}.i"
      "${MODULE_OUTDIR}"
      "${MODULE_NAME}")
  ENDIF("${MODULE_NAME}" MATCHES "Teuchos")
  SET_PROPERTY(TARGET ${PyTrilinos_TARGET_NAME} PROPERTY NO_SONAME 1)
  SWIG_LINK_LIBRARIES(${PyTrilinos_TARGET_NAME} pytrilinos ${DOMI_LIB})
  SET_TARGET_PROPERTIES(${PyTrilinos_TARGET_NAME} PROPERTIES
    SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON
  )
  # Add the header file to the list of additional "make clean" files
  LIST(APPEND ADDITIONAL_CLEAN_FILES "${MODULE}PYTHON_wrap.h")

  #
  # Choosing the Trilinos libraries to link against, the python module
  # to byte-compile and the installation rules requires a little bit
  # of logic.

  # PyTrilinos.Teuchos: From Teuchos.i, generate
  # PyTrilinos/Teuchos/__init__.py, which violates the pattern.  This
  # causes other packages to "import Teuchos" properly.
  IF("${MODULE_OUTDIR}" MATCHES "PyTrilinos/Teuchos")
    INSTALL(TARGETS ${PyTrilinos_TARGET_NAME}
      DESTINATION ${PyTrilinos_INSTALL_DIR}/Teuchos)

  # PyTrilinos.NOX.Epetra
  ELSEIF("${MODULE_OUTDIR}" MATCHES "PyTrilinos/NOX/Epetra")
    SET(TRILINOS_LIBS ${NOXEPETRA_LIBRARIES})
    INSTALL(TARGETS ${PyTrilinos_TARGET_NAME}
      DESTINATION ${PyTrilinos_INSTALL_DIR}/NOX/Epetra)

  #
  # PyTrilinos.NOX.PETSc
  ELSEIF("${MODULE_OUTDIR}" MATCHES "PyTrilinos/NOX/PETSc")
    SET(TRILINOS_LIBS ${NOXPETSC_LIBRARIES})
    INSTALL(TARGETS ${PyTrilinos_TARGET_NAME}
      DESTINATION ${PyTrilinos_INSTALL_DIR}/NOX/PETSc)

  #
  # PyTrilinos.NOX
  ELSEIF("${MODULE_OUTDIR}" MATCHES "PyTrilinos/NOX")
    SET(TRILINOS_LIBS ${NOX_LIBRARIES})
    INSTALL(TARGETS ${PyTrilinos_TARGET_NAME}
      DESTINATION ${PyTrilinos_INSTALL_DIR}/NOX)

  #
  # PyTrilinos.LOCA.Hopf
  ELSEIF("${MODULE_OUTDIR}" MATCHES "PyTrilinos/LOCA/Hopf")
    SET(TRILINOS_LIBS ${LOCAEPETRA_LIBRARIES})
    INSTALL(TARGETS ${PyTrilinos_TARGET_NAME}
      DESTINATION ${PyTrilinos_INSTALL_DIR}/LOCA/Hopf)

  #
  # PyTrilinos.LOCA.Pitchfork
  ELSEIF("${MODULE_OUTDIR}" MATCHES "PyTrilinos/LOCA/Pitchfork")
    SET(TRILINOS_LIBS ${LOCAEPETRA_LIBRARIES})
    INSTALL(TARGETS ${PyTrilinos_TARGET_NAME}
      DESTINATION ${PyTrilinos_INSTALL_DIR}/LOCA/Pitchfork)

  #
  # PyTrilinos.LOCA.TurningPoint
  ELSEIF("${MODULE_OUTDIR}" MATCHES "PyTrilinos/LOCA/TurningPoint")
    SET(TRILINOS_LIBS ${LOCAEPETRA_LIBRARIES})
    INSTALL(TARGETS ${PyTrilinos_TARGET_NAME}
      DESTINATION ${PyTrilinos_INSTALL_DIR}/LOCA/TurningPoint)

  #
  # PyTrilinos.LOCA.Epetra
  ELSEIF("${MODULE_OUTDIR}" MATCHES "PyTrilinos/LOCA/Epetra")
    SET(TRILINOS_LIBS ${LOCAEPETRA_LIBRARIES})
    INSTALL(TARGETS ${PyTrilinos_TARGET_NAME}
      DESTINATION ${PyTrilinos_INSTALL_DIR}/LOCA/Epetra)

  #
  # PyTrilinos.LOCA
  ELSEIF("${MODULE_OUTDIR}" MATCHES "PyTrilinos/LOCA")
    SET(TRILINOS_LIBS ${LOCA_LIBRARIES})
    INSTALL(TARGETS ${PyTrilinos_TARGET_NAME}
      DESTINATION ${PyTrilinos_INSTALL_DIR}/LOCA)

  #
  # PyTrilinos.Isorropia
  ELSEIF("${MODULE_OUTDIR}" MATCHES "PyTrilinos/Isorropia")
    SET(TRILINOS_LIBS ${ISORROPIA_LIBRARIES})
    INSTALL(TARGETS ${PyTrilinos_TARGET_NAME}
      DESTINATION ${PyTrilinos_INSTALL_DIR}/Isorropia)

  #
  # PyTrilinos
  ELSE()
    SET(TRILINOS_LIBS ${${MODULE_UPPER}_LIBRARIES})
    INSTALL(TARGETS ${PyTrilinos_TARGET_NAME}
      DESTINATION ${PyTrilinos_INSTALL_DIR})
  ENDIF()

  #
  SWIG_LINK_LIBRARIES(${PyTrilinos_TARGET_NAME} ${TRILINOS_LIBS})
  SET_TARGET_PROPERTIES(${PyTrilinos_TARGET_NAME} PROPERTIES
    LINK_FLAGS "${EXTRA_LINK_ARGS}"
    SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON
    )
ENDFOREACH(MODULE)

#
# Add the additional "make clean" files
GET_DIRECTORY_PROPERTY(clean_files ADDITIONAL_MAKE_CLEAN_FILES)
LIST(APPEND            clean_files ${ADDITIONAL_CLEAN_FILES})
LIST(REMOVE_DUPLICATES clean_files)
LIST(REMOVE_ITEM       clean_files "")
SET_DIRECTORY_PROPERTIES(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${clean_files}")
