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

#ifndef PIRO_TEMPUS_REDUCED_OBJECTIVE_HPP
#define PIRO_TEMPUS_REDUCED_OBJECTIVE_HPP

#include <string>

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Tempus_IntegratorBasic.hpp"
#include "Tempus_IntegratorForwardSensitivity.hpp"
#include "Tempus_IntegratorAdjointSensitivity.hpp"
#include "Tempus_IntegratorPseudoTransientForwardSensitivity.hpp"
#include "Tempus_IntegratorPseudoTransientAdjointSensitivity.hpp"

#include "Thyra_ModelEvaluator.hpp"
#include "Thyra_DefaultNominalBoundsOverrideModelEvaluator.hpp"
#include "Thyra_VectorStdOps.hpp"

#include "ROL_Objective.hpp"
#include "ROL_DynamicObjective.hpp"
#include "ROL_Vector.hpp"
#include "ROL_ThyraVector.hpp"
#include "Piro_ROL_ObserverBase.hpp"
#include "Piro_TempusIntegrator.hpp" 

namespace Piro {

template <typename Real>
class ThyraProductME_TempusFinalObjective : public virtual ROL::DynamicObjective<Real> {
public:

  ThyraProductME_TempusFinalObjective(
    const Teuchos::RCP<Piro::TempusIntegrator<Real> >& integrator,
    int g_index,
    const std::vector<int>& p_indices,
    Teuchos::ParameterList& piroParams,
    Teuchos::EVerbosityLevel verbLevel= Teuchos::VERB_HIGH,
    Teuchos::RCP<ROL_ObserverBase<Real>> observer = Teuchos::null);

  virtual ~ThyraProductME_TempusFinalObjective() {}

  //! Compute value of objective
  Real value( const ROL::Vector<Real> &u_old, const ROL::Vector<Real> &u_new, 
              const ROL::Vector<Real> &z, const ROL::TimeStamp<Real> &timeStamp ) const;

  //! Compute gradient of objective
  void gradient( ROL::Vector<Real> &g, const ROL::Vector<Real> &x, Real &tol );

  //! Helper function to create optimization vector
  Teuchos::RCP<ROL::Vector<Real> > create_design_vector() const;

  //! Helper function to create a response vector
  Teuchos::RCP<ROL::Vector<Real> > create_response_vector() const;

  //! Helper function to run tempus, computing responses and derivatives
  void run_tempus(ROL::Vector<Real>& r, const ROL::Vector<Real>& p) const;
  void run_tempus(const Thyra::ModelEvaluatorBase::InArgs<Real>&  inArgs,
                  const Thyra::ModelEvaluatorBase::OutArgs<Real>& outArgs) const;

private:

  struct ObjectiveStruct {
    Real value_;
    Teuchos::RCP<ROL::Vector<Real> > gradient1_ptr_;
    Teuchos::RCP<ROL::Vector<Real> > gradient2_ptr_;
    bool isValueValid_;
    bool isGradient1Valid_;
    bool isGradient2Valid_;
    bool areGradientsAllocated_;

    ObjectiveStruct() : 
      value_(0), gradient1_ptr_(Teuchos::null), gradient2_ptr_(Teuchos::null),
      isValueValid_(false), isGradient1Valid_(false), isGradient2Valid_(false),
      areGradientsAllocated_(false) {}

    void allocateGradients(const ROL::Vector<Real>& gradient1, const ROL::Vector<Real>& gradient2) {
      gradient1_ptr_ = gradient1.clone();
      gradient2_ptr_ = gradient2.clone();
      areGradientsAllocated_ = true;
      markAsNotValid();
    }

    bool shareGradients(const ObjectiveStruct& objectiveStruct) {
      return (objectiveStruct.gradient1_ptr_.ptr() == gradient1_ptr_.ptr()) ||
        (objectiveStruct.gradient2_ptr_.ptr() == gradient2_ptr_.ptr());
    }

    void markAsNotValid() {
    isValueValid_ = isGradient1Valid_ = isGradient2Valid_ = false;
    }
  };


  const Teuchos::RCP<Piro::TempusIntegrator<Real> > integrator_;
  const Teuchos::RCP<Thyra::ModelEvaluator<Real>> thyra_model_;
  const int g_index_;
  const std::vector<int> p_indices_;
  Real objectiveRecoveryValue_;
  bool useObjectiveRecoveryValue_;
  ROL::UpdateType updateType_;

  ObjectiveStruct objectiveStr_, cached_objectiveStr_,  tmp_objectiveStr_;

  Teuchos::ParameterList& optParams_;
  Teuchos::RCP<Teuchos::FancyOStream> out_;
  Teuchos::EVerbosityLevel verbosityLevel_;
  Teuchos::RCP<ROL_ObserverBase<Real>> observer_;

  Teuchos::RCP<Teuchos::ParameterList> tempus_params_;
  bool use_fd_gradient_;
  Real time_final_;

}; // class ThyraProductME_TempusFinalObjective

template <typename Real>
ThyraProductME_TempusFinalObjective<Real>::
ThyraProductME_TempusFinalObjective(
  const Teuchos::RCP<Piro::TempusIntegrator<Real> >& integrator,
  int g_index,
  const std::vector<int>& p_indices,
  Teuchos::ParameterList& piroParams,
  Teuchos::EVerbosityLevel verbLevel,
  Teuchos::RCP<ROL_ObserverBase<Real>> observer) :
  integrator_(integrator),
  thyra_model_(integrator->getModel()),
  g_index_(g_index),
  p_indices_(p_indices),
  optParams_(piroParams.sublist("Optimization Status")),
  out_(Teuchos::VerboseObjectBase::getDefaultOStream()),
  verbosityLevel_(verbLevel),
  observer_(observer),
  tempus_params_(Teuchos::rcp<Teuchos::ParameterList>(new Teuchos::ParameterList(piroParams.sublist("Tempus")))),
  use_fd_gradient_(true),
  time_final_(piroParams.get<Real>("Time final", 0.))
{
  
}

template <typename Real>
Real
ThyraProductME_TempusFinalObjective<Real>::
value( const ROL::Vector<Real> &u_old, const ROL::Vector<Real> &u_new, 
              const ROL::Vector<Real> &p, const ROL::TimeStamp<Real> &timeStamp ) const
{
  using Teuchos::RCP;
  typedef Thyra::ModelEvaluatorBase MEB;

  if(verbosityLevel_ >= Teuchos::VERB_MEDIUM)
    *out_ << "Piro::ThyraProductME_TempusFinalObjective::value" << std::endl;

  if(objectiveStr_.isValueValid_) {
    if(verbosityLevel_ >= Teuchos::VERB_HIGH)
      *out_ << "Piro::ThyraProductME_TempusFinalObjective::value, Skipping Computation of Value" << std::endl;
    return objectiveStr_.value_;
  }

  // Run tempus and compute response for specified parameter values
  MEB::InArgs<Real> inArgs = thyra_model_->getNominalValues();
  MEB::OutArgs<Real> outArgs = thyra_model_->createOutArgs();
  const ROL::ThyraVector<Real>& thyra_p =
    Teuchos::dyn_cast<const ROL::ThyraVector<Real> >(p);
  Teuchos::RCP<const Thyra::ProductVectorBase<Real> > thyra_prodvec_p =
    Teuchos::rcp_dynamic_cast<const Thyra::ProductVectorBase<Real>>(thyra_p.getVector());
  for(std::size_t i=0; i<p_indices_.size(); ++i)
    inArgs.set_p(p_indices_[i], thyra_prodvec_p->getVectorBlock(i));
  RCP<Thyra::VectorBase<Real> > g =
    Thyra::createMember<Real>(thyra_model_->get_g_space(g_index_));
  outArgs.set_g(g_index_, g);
  run_tempus(inArgs, outArgs);

  return ::Thyra::get_ele(*g,0);
}

template <typename Real>
void
ThyraProductME_TempusFinalObjective<Real>::
gradient(ROL::Vector<Real> &grad, const ROL::Vector<Real> &p, Real &tol)
{
  if (use_fd_gradient_) {
    ROL::Objective<Real>::gradient(grad, p, tol);
    return;
  }
}

template <typename Real>
Teuchos::RCP<ROL::Vector<Real> >
ThyraProductME_TempusFinalObjective<Real>::
create_design_vector() const {

  typedef Thyra::ModelEvaluatorBase MEB;

  Teuchos::Array<Teuchos::RCP<Thyra::VectorSpaceBase<Real> const>> p_spaces(p_indices_.size());
  Teuchos::Array<Teuchos::RCP<Thyra::VectorBase<Real>>> p_vecs(p_indices_.size());
  MEB::InArgs<Real> nominalValues = thyra_model_->getNominalValues();
  for (auto i = 0; i < p_indices_.size(); ++i) {
    p_spaces[i] = thyra_model_->get_p_space(p_indices_[i]);
    p_vecs[i] = Thyra::createMember(p_spaces[i]);
    if (nominalValues.get_p(p_indices_[i]) != Teuchos::null)
      Thyra::assign(p_vecs[i].ptr(), *(nominalValues.get_p(p_indices_[i])));
    else
      Thyra::assign(p_vecs[i].ptr(), Teuchos::ScalarTraits<Real>::zero());
  }
  Teuchos::RCP<Thyra::DefaultProductVectorSpace<Real> const> p_space = Thyra::productVectorSpace<double>(p_spaces);
  Teuchos::RCP<Thyra::DefaultProductVector<Real>> p_prod = Thyra::defaultProductVector<double>(p_space, p_vecs());

  return Teuchos::rcp(new ROL::ThyraVector<Real>(p_prod));
}

template <typename Real>
Teuchos::RCP<ROL::Vector<Real> >
ThyraProductME_TempusFinalObjective<Real>::
create_response_vector() const {
  Teuchos::RCP<Thyra::VectorBase<Real> > g =
    Thyra::createMember<Real>(thyra_model_->get_g_space(g_index_));
  Thyra::assign(g.ptr(), Teuchos::ScalarTraits<Real>::zero());
  return Teuchos::rcp(new ROL::ThyraVector<Real>(g));
}

template <typename Real>
void
ThyraProductME_TempusFinalObjective<Real>::
run_tempus(ROL::Vector<Real>& r, const ROL::Vector<Real>& p) const
{
  typedef Thyra::ModelEvaluatorBase MEB;

  MEB::InArgs<Real> inArgs = thyra_model_->getNominalValues();
  MEB::OutArgs<Real> outArgs = thyra_model_->createOutArgs();
  const ROL::ThyraVector<Real>& thyra_p =
    Teuchos::dyn_cast<const ROL::ThyraVector<Real> >(p);
  Teuchos::RCP<const Thyra::ProductVectorBase<Real> > thyra_prodvec_p =
    Teuchos::rcp_dynamic_cast<const Thyra::ProductVectorBase<Real>>(thyra_p.getVector());
  for(std::size_t i=0; i<p_indices_.size(); ++i)
    inArgs.set_p(p_indices_[i], thyra_prodvec_p->getVectorBlock(i));
  ROL::ThyraVector<Real>& thyra_r =
    Teuchos::dyn_cast<ROL::ThyraVector<Real> >(r);
  outArgs.set_g(g_index_, thyra_r.getVector());
  run_tempus(inArgs, outArgs);
}

template <typename Real>
void
ThyraProductME_TempusFinalObjective<Real>::
run_tempus(const Thyra::ModelEvaluatorBase::InArgs<Real>&  inArgs,
           const Thyra::ModelEvaluatorBase::OutArgs<Real>& outArgs) const
{
  using Teuchos::rcp;
  using Teuchos::RCP;
  using Teuchos::rcpFromRef;
  typedef Thyra::ModelEvaluatorBase MEB;
  typedef Thyra::DefaultNominalBoundsOverrideModelEvaluator<Real> DNBOME;

  // Override nominal values in model to supplied inArgs
  RCP<DNBOME> wrapped_model = rcp(new DNBOME(thyra_model_, rcpFromRef(inArgs)));

  Real t;
  RCP<const Thyra::VectorBase<Real> > x, x_dot;
  RCP<const Thyra::MultiVectorBase<double> > dxdp, dxdotdp;
  RCP<Thyra::VectorBase<Real> > g = outArgs.get_g(g_index_);

  // Create and run integrator
  SENS_METHOD sens_method = Piro::NONE; 
  Teuchos::RCP<Piro::TempusIntegrator<Real> > integrator 
    = Teuchos::rcp(new Piro::TempusIntegrator<Real>(tempus_params_, wrapped_model, sens_method));
  const bool integratorStatus = integrator->advanceTime(time_final_);
  TEUCHOS_TEST_FOR_EXCEPTION(
    !integratorStatus, std::logic_error, "Integrator failed!");

  // Get final state
  t = integrator->getTime();
  x = integrator->getX();
  x_dot = integrator->getXDot();


  // Evaluate response at final state
  const int num_g = thyra_model_->get_g_space(g_index_)->dim();
  MEB::InArgs<Real> modelInArgs   = inArgs;
  MEB::OutArgs<Real> modelOutArgs = outArgs;
  modelInArgs.set_x(x);
  if (modelInArgs.supports(MEB::IN_ARG_x_dot)) modelInArgs.set_x_dot(x_dot);
  if (modelInArgs.supports(MEB::IN_ARG_t)) modelInArgs.set_t(t);
  RCP<Thyra::MultiVectorBase<Real> > dgdx, dgdxdot;
  MEB::EDerivativeMultiVectorOrientation dgdx_orientation =
    MEB::DERIV_MV_JACOBIAN_FORM;
  MEB::EDerivativeMultiVectorOrientation dgdxdot_orientation =
    MEB::DERIV_MV_JACOBIAN_FORM;

  thyra_model_->evalModel(modelInArgs, modelOutArgs);
}

} // namespace Piro

#endif