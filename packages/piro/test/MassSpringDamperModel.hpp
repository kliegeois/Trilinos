// @HEADER
// ************************************************************************
//
//        Piro: Strategy package for embedded analysis capabilitites
//                  Copyright (2010) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
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
// Questions? Contact Andy Salinger (agsalin@sandia.gov), Sandia
// National Laboratories.
//
// ************************************************************************
// @HEADER

#ifndef MASSSPRINGDAMPERMODEL_H
#define MASSSPRINGDAMPERMODEL_H

#include "Teuchos_Assert.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"
#include "Thyra_ModelEvaluator.hpp" // Interface
#include "Thyra_StateFuncModelEvaluatorBase.hpp" // Implementation

#include "Teuchos_ParameterListAcceptorDefaultBase.hpp"
#include "Teuchos_ParameterList.hpp"


/** \brief Mass-spring-damper model problem from Tempus.
  * This is a mass-spring-damper differential equation
  *   \f[
  *   m \ddot{x} + 2 \sqrt{m\,k} \dot{x} + k x - F = 0
  *   \f]
  * . 
*/

using LO = Tpetra::Map<>::local_ordinal_type;
using GO = Tpetra::Map<>::global_ordinal_type;
using Scalar = double; 
typedef Tpetra::Map<LO,GO>  Tpetra_Map;
typedef Tpetra::Vector<Scalar,LO,GO>  Tpetra_Vector;
typedef Tpetra::MultiVector<Scalar,LO,GO>  Tpetra_MultiVector;
typedef Tpetra::Operator<Scalar,LO,GO>  Tpetra_Operator;
typedef Tpetra::CrsGraph<LO,GO>  Tpetra_CrsGraph;
typedef Tpetra::CrsMatrix<Scalar,LO,GO>  Tpetra_CrsMatrix;
typedef Thyra::TpetraOperatorVectorExtraction<Scalar, LO, GO> ConverterT;



class MassSpringDamperModel
    : public Thyra::StateFuncModelEvaluatorBase<Scalar>,
      public Teuchos::ParameterListAcceptorDefaultBase

{
  public:

  // Constructor
  MassSpringDamperModel(Teuchos::RCP<Teuchos::ParameterList> pList = Teuchos::null);

  // Exact solution
  Thyra::ModelEvaluatorBase::InArgs<Scalar> getExactSolution(double t) const;

  // Exact sensitivity solution
  Thyra::ModelEvaluatorBase::InArgs<Scalar> getExactSensSolution(int j, double t) const;

  /** \name Public functions overridden from ModelEvaluator. */
  //@{

  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > get_x_space() const;
  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > get_f_space() const;
  Thyra::ModelEvaluatorBase::InArgs<Scalar> getNominalValues() const;
  Teuchos::RCP<Thyra::LinearOpWithSolveBase<Scalar> > create_W() const;
  Teuchos::RCP<Thyra::LinearOpBase<Scalar> > create_W_op() const;
  Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar> > get_W_factory() const;
  Thyra::ModelEvaluatorBase::InArgs<Scalar> createInArgs() const;

  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > get_p_space(int l) const;
  Teuchos::RCP<const Teuchos::Array<std::string> > get_p_names(int l) const;
  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > get_g_space(int j) const;

  //@}

  /** \name Public functions overridden from ParameterListAcceptor. */
  //@{
  void setParameterList(Teuchos::RCP<Teuchos::ParameterList> const& paramList);
  Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;
  //@}

private:

  void setupInOutArgs_() const;

  /** \name Private functions overridden from ModelEvaluatorDefaultBase. */
  //@{
  Thyra::ModelEvaluatorBase::OutArgs<Scalar> createOutArgsImpl() const;
  void evalModelImpl(
    const Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs_bar,
    const Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs_bar
    ) const;
  //@}

  void calculateCoeffFromIC_();

protected:
  int dim_;         ///< Number of state unknowns (1)
  int Np_;          ///< Number of parameter vectors (1)
  int np_;          ///< Number of parameters in this vector (2)
  int Ng_;          ///< Number of observation functions (1)
  int ng_;          ///< Number of elements in this observation function (1)
  bool haveIC_;     ///< false => no nominal values are provided (default=true)
  bool acceptModelParams_; ///< Changes inArgs to require parameters
  bool useDfDpAsTangent_; ///< Treat DfDp OutArg as tangent (df/dx*dx/dp+df/dp)
  mutable bool isInitialized_;
  mutable Thyra::ModelEvaluatorBase::InArgs<Scalar>  inArgs_;
  mutable Thyra::ModelEvaluatorBase::OutArgs<Scalar> outArgs_;
  mutable Thyra::ModelEvaluatorBase::InArgs<Scalar>  nominalValues_;
  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > x_space_;
  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > f_space_;
  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > p_space_;
  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > g_space_;
  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > DxDp_space_;

  double k_, m_, F_, target_time_, c1_, c2_, c3_, lambda_, target_x_, target_x_dot_, scaling_;
};


/// Non-member constructor
//Teuchos::RCP<MassSpringDamperModel> sineCosineModel(
//  Teuchos::RCP<Teuchos::ParameterList> pList_)
//{
//  Teuchos::RCP<MassSpringDamperModel> model = rcp(new MassSpringDamperModel(pList_));
//  return(model);
//}

//! Adjoint for MassSpringDamperModel
/*!
 * This model evaluator modifies evalModel() to compute the adjoint operator
 * instead of the forward operator.
 */
class MassSpringDamperModelAdjoint
  : public MassSpringDamperModel
{
  public:

  // Constructor
  MassSpringDamperModelAdjoint(Teuchos::RCP<Teuchos::ParameterList> pList = Teuchos::null) : MassSpringDamperModel(pList) {}

  /** \name Public functions overridden from ModelEvaluator. */
  //@{

  Thyra::ModelEvaluatorBase::InArgs<Scalar> createInArgs() const;
  Teuchos::RCP<Thyra::LinearOpWithSolveBase<Scalar> > create_W() const;
  Teuchos::RCP<Thyra::LinearOpBase<Scalar> > create_W_op() const;

  //@}

private:

  /** \name Private functions overridden from ModelEvaluatorDefaultBase. */
  //@{
  Thyra::ModelEvaluatorBase::OutArgs<Scalar> createOutArgsImpl() const;
  void evalModelImpl(
    const Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs_bar,
    const Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs_bar
    ) const;
  //@}
};


#endif // MASSSPRINGDAMPERMODEL
