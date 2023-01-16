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

#ifndef PIRO_PRODUCTMODELEVAL_HPP
#define PIRO_PRODUCTMODELEVAL_HPP

#include "Teuchos_RCP.hpp"
#include "Thyra_ModelEvaluator.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_ModelEvaluatorDelegatorBase.hpp"

namespace Piro {

/** \brief Product Model Evaluator
 *
 */

template<class Real>
class ProductModelEvaluator : public Thyra::ModelEvaluatorDelegatorBase<Real>
{
public:

    ProductModelEvaluator(const Teuchos::RCP<Thyra::ModelEvaluator<Real>> thyra_model,
                            int g_index,
                            const std::vector<int>& p_indices);

    ~ProductModelEvaluator();

    /** \brief . */
    int Np() const;
    /** \brief . */
    int Ng() const;

    Teuchos::RCP<const Thyra::VectorSpaceBase<Real>> get_x_space() const;
    Teuchos::RCP<const Thyra::VectorSpaceBase<Real>> get_f_space() const;
    Teuchos::RCP<const Thyra::VectorSpaceBase<Real>> get_g_space( int l) const;
    Teuchos::RCP<const Thyra::VectorSpaceBase<Real>> get_p_space( int l) const;

    Teuchos::RCP<const Teuchos::Array<std::string> > get_p_names(int l) const;

    Teuchos::RCP<Thyra::LinearOpBase<Real> > create_W_op() const;
    Teuchos::RCP<Thyra::PreconditionerBase<Real> > create_W_prec() const;
    Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Real> > get_W_factory() const;
    Teuchos::RCP<Thyra::LinearOpBase<Real> > create_hess_g_pp( int j, int l1, int l2 ) const;

    /** \brief . */
    //Teuchos::RCP<Thyra::LinearOpBase<Real> > create_DfDp_op(int l) const;
    /** \brief . */
    //Teuchos::RCP<Thyra::LinearOpBase<Real> > create_DgDx_dot_op(int j) const;
    /** \brief . */
    //Teuchos::RCP<Thyra::LinearOpBase<Real> > create_DgDx_op(int j) const;
    /** \brief . */
    //Teuchos::RCP<Thyra::LinearOpBase<Real> > create_DgDp_op(int j, int l) const;
    /** \brief . */
    //Teuchos::RCP<Thyra::LinearOpWithSolveBase<Real> > create_W() const;
    /** \brief . */
    //Thyra::ModelEvaluatorBase::OutArgs<Real> createOutArgs() const;
    /** \brief . */
    //void evalModel(
    //    const Thyra::ModelEvaluatorBase::InArgs<Real> &inArgs,
    //    const Thyra::ModelEvaluatorBase::OutArgs<Real> &outArgs
    //    ) const;

    const Teuchos::RCP<Thyra::ModelEvaluator<Real>> getModel() { return thyra_model_; }

    Thyra::ModelEvaluatorBase::InArgs<Real>  createInArgs() const;

    void reportFinalPoint(
        const Thyra::ModelEvaluatorBase::InArgs<Real>& finalPoint,
        const bool wasSolved);
        
    Teuchos::ArrayView<const std::string> get_g_names(int j) const;

    /** \brief . */
    ::Thyra::ModelEvaluatorBase::InArgs<Real> getNominalValues() const;
    /** \brief . */
    ::Thyra::ModelEvaluatorBase::InArgs<Real> getLowerBounds() const;
    /** \brief . */
    ::Thyra::ModelEvaluatorBase::InArgs<Real> getUpperBounds() const;

protected:

    /** \brief . */
    Thyra::ModelEvaluatorBase::OutArgs<Real>
    createOutArgsImpl() const;

    /** \brief . */
    void
    evalModelImpl(
        const Thyra::ModelEvaluatorBase::InArgs<Real>& inArgs,
        const Thyra::ModelEvaluatorBase::OutArgs<Real>& outArgs) const;
    //@}


private:

    void fromInternalInArgs(const Thyra::ModelEvaluatorBase::InArgs<Real>& inArgs1, Thyra::ModelEvaluatorBase::InArgsSetup<Real>& inArgs2) const;
    void toInternalInArgs(const Thyra::ModelEvaluatorBase::InArgs<Real>& inArgs1, Thyra::ModelEvaluatorBase::InArgsSetup<Real>& inArgs2) const;
    void fromInternalOutArgs(const Thyra::ModelEvaluatorBase::OutArgs<Real>& outArgs1, Thyra::ModelEvaluatorBase::OutArgsSetup<Real>& outArgs2) const;
    void toInternalOutArgs(const Thyra::ModelEvaluatorBase::OutArgs<Real>& outArgs1, Thyra::ModelEvaluatorBase::OutArgsSetup<Real>& outArgs2) const;

    /** \brief . */
    Thyra::ModelEvaluatorBase::InArgs<Real>  createInArgsImpl() const;

    const Teuchos::RCP<Thyra::ModelEvaluator<Real>> thyra_model_;
    const int g_index_;
    const std::vector<int> p_indices_;
}; // class ProductModelEvaluator


template <typename Real>
ProductModelEvaluator<Real>::
ProductModelEvaluator(
    const Teuchos::RCP<Thyra::ModelEvaluator<Real>> thyra_model,
    int g_index,
    const std::vector<int>& p_indices) :
    thyra_model_(thyra_model),
    g_index_(g_index),
    p_indices_(p_indices)
{
}

template <typename Real>
ProductModelEvaluator<Real>::~ProductModelEvaluator()
{
}

template <typename Real>
Teuchos::RCP<const Thyra::VectorSpaceBase<Real>>
ProductModelEvaluator<Real>::get_x_space() const
{
    return thyra_model_->get_x_space();
}

template <typename Real>
Teuchos::RCP<const Thyra::VectorSpaceBase<Real>>
ProductModelEvaluator<Real>::get_f_space() const
{
    return thyra_model_->get_f_space();
}

template <typename Real>
Teuchos::RCP<const Thyra::VectorSpaceBase<Real>>
ProductModelEvaluator<Real>::get_p_space(int l) const
{
    TEUCHOS_TEST_FOR_EXCEPTION(l != 0, std::logic_error,
                        std::endl <<
                        "Error!  ProductModelEvaluator<Real>::get_p_space() only " <<
                        " supports 1 parameter vector.  Supplied index l = " <<
                        l << std::endl);

    Teuchos::Array<Teuchos::RCP<Thyra::VectorSpaceBase<Real> const>> p_spaces(p_indices_.size());
    for (auto i = 0; i < p_indices_.size(); ++i) {
        p_spaces[i] = thyra_model_->get_p_space(p_indices_[i]);
    }
    Teuchos::RCP<Thyra::DefaultProductVectorSpace<Real> const> p_space = Thyra::productVectorSpace<Real>(p_spaces);

    return p_space;
}

template <typename Real>
Teuchos::RCP<const Thyra::VectorSpaceBase<Real>>
ProductModelEvaluator<Real>::get_g_space(int l) const
{
    TEUCHOS_TEST_FOR_EXCEPTION(l > thyra_model_->Ng(), std::logic_error,
                        std::endl <<
                        "Error!  ProductModelEvaluator::get_g_space() Supplied index l = " <<
                        l << " is greater or equal to the number of responses of the underlying model " << 
                        thyra_model_->Ng() << std::endl);
    Teuchos::RCP<const Thyra::VectorSpaceBase<Real>> g_space = thyra_model_->get_g_space(l);
    return g_space;
}

template <typename Real>
Teuchos::RCP<const  Teuchos::Array<std::string> >
ProductModelEvaluator<Real>::get_p_names(int l) const
{
    TEUCHOS_TEST_FOR_EXCEPTION(l != 0, std::logic_error,
                        std::endl <<
                        "Error!  ProductModelEvaluator<Real>::get_p_names() only " <<
                        " supports 1 parameter vector.  Supplied index l = " <<
                        l << std::endl);

    Teuchos::RCP<Teuchos::Array<std::string> > p_names =
        Teuchos::rcp(new Teuchos::Array<std::string>(p_indices_.size()) );
    for (auto i = 0; i < p_indices_.size(); ++i) {
    std::stringstream ss;
    ss << "Parameter " << i;
    const std::string name = ss.str();
    (*p_names)[i] = name;
    }
    return p_names;
}

template <typename Real>
Teuchos::RCP<Thyra::LinearOpBase<Real>>
ProductModelEvaluator<Real>::create_W_op() const
{
    return thyra_model_->create_W_op();
}

template <typename Real>
Teuchos::RCP<Thyra::PreconditionerBase<Real>>
ProductModelEvaluator<Real>::create_W_prec() const
{
    return thyra_model_->create_W_prec();
}

template <typename Real>
Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Real>>
ProductModelEvaluator<Real>::get_W_factory() const
{
    return thyra_model_->get_W_factory();
}

template <typename Real>
Teuchos::RCP<Thyra::LinearOpBase<Real>>
ProductModelEvaluator<Real>::create_hess_g_pp( int j, int l1, int l2 ) const
{
    // NO ?
    return thyra_model_->create_hess_g_pp(j, l1, l2);
}

template <typename Real>
Thyra::ModelEvaluatorBase::InArgs<Real>
ProductModelEvaluator<Real>::createInArgs() const
{
    Thyra::ModelEvaluatorBase::InArgs<Real> internal_inArgs = thyra_model_->createInArgs();
    Thyra::ModelEvaluatorBase::InArgsSetup<Real> result; 
    result.setModelEvalDescription(this->description());
    result.set_Np_Ng(1, thyra_model_->Ng());

    this->fromInternalInArgs(internal_inArgs, result);

    return result;
}

template <typename Real>
Thyra::ModelEvaluatorBase::OutArgs<Real>
ProductModelEvaluator<Real>::createOutArgsImpl() const
{
    Thyra::ModelEvaluatorBase::OutArgs<Real> internal_outArgs = thyra_model_->createOutArgs();
    Thyra::ModelEvaluatorBase::OutArgsSetup<Real> result; 
    result.setModelEvalDescription(this->description());
    result.set_Np_Ng(1, thyra_model_->Ng());

    this->fromInternalOutArgs(internal_outArgs, result);

    return result;
}

template <typename Real>
void 
ProductModelEvaluator<Real>::evalModelImpl(
    const Thyra::ModelEvaluatorBase::InArgs<Real>&  inArgs,
    const Thyra::ModelEvaluatorBase::OutArgs<Real>& outArgs) const
{
    Thyra::ModelEvaluatorBase::InArgs<Real> internal_inArgs = thyra_model_->createInArgs();
    Thyra::ModelEvaluatorBase::OutArgs<Real> internal_outArgs = thyra_model_->createOutArgs();

    internal_outArgs.setArgs(outArgs, true);
    internal_inArgs.setArgs(inArgs, true);

    Teuchos::RCP<const Thyra::ProductVectorBase<Real> > prodvec_p = Teuchos::rcp_dynamic_cast<const Thyra::ProductVectorBase<Real>>(inArgs.get_p(0));

    for (auto i = 0; i < p_indices_.size(); ++i) {
        internal_inArgs.set_p(p_indices_[i], prodvec_p->getVectorBlock(i));
    }

    thyra_model_->evalModel(internal_inArgs,internal_outArgs);
}

template <typename Real>
Thyra::ModelEvaluatorBase::InArgs<Real>
ProductModelEvaluator<Real>::createInArgsImpl() const
{
    Thyra::ModelEvaluatorBase::InArgs<Real> internal_inArgs = thyra_model_->createInArgs();
    Thyra::ModelEvaluatorBase::InArgsSetup<Real> result; 
    result.setModelEvalDescription(this->description());
    result.set_Np_Ng(1, thyra_model_->Ng());

    this->fromInternalInArgs(internal_inArgs, result);

    return result;
}

template <typename Real>
Teuchos::ArrayView<const std::string>
ProductModelEvaluator<Real>::get_g_names(int j) const
{
    return thyra_model_->get_g_names(j);
}

template <typename Real>
Thyra::ModelEvaluatorBase::InArgs<Real>
ProductModelEvaluator<Real>::getNominalValues() const
{
    Thyra::ModelEvaluatorBase::InArgs<Real> internal_inArgs = thyra_model_->getNominalValues();
    Thyra::ModelEvaluatorBase::InArgsSetup<Real> result; 
    result.setModelEvalDescription(this->description());
    result.set_Np_Ng(1, thyra_model_->Ng());

    this->fromInternalInArgs(internal_inArgs, result);
    result.setArgs(internal_inArgs, true);

    Teuchos::RCP<const Thyra::DefaultProductVectorSpace<Real>> p_space = Teuchos::rcp_dynamic_cast<const Thyra::DefaultProductVectorSpace<Real>>(this->get_p_space(0));

    Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<Real>>> p_vecs(p_indices_.size());
    for (auto i = 0; i < p_indices_.size(); ++i) {
        p_vecs[i] = internal_inArgs.get_p(p_indices_[i]);
    }
    Teuchos::RCP<Thyra::DefaultProductVector<Real>> p_prod = Thyra::defaultProductVector<double>(p_space, p_vecs());

    result.set_p(0, p_prod);

    return result;
}

template <typename Real>
Thyra::ModelEvaluatorBase::InArgs<Real>
ProductModelEvaluator<Real>::getLowerBounds() const
{
    Thyra::ModelEvaluatorBase::InArgs<Real> internal_inArgs = thyra_model_->getLowerBounds();
    Thyra::ModelEvaluatorBase::InArgsSetup<Real> result; 
    result.setModelEvalDescription(this->description());
    result.set_Np_Ng(1, thyra_model_->Ng());

    this->fromInternalInArgs(internal_inArgs, result);
    result.setArgs(internal_inArgs, true);

    Teuchos::RCP<const Thyra::DefaultProductVectorSpace<Real>> p_space = Teuchos::rcp_dynamic_cast<const Thyra::DefaultProductVectorSpace<Real>>(this->get_p_space(0));

    Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<Real>>> p_vecs(p_indices_.size());
    for (auto i = 0; i < p_indices_.size(); ++i) {
        p_vecs[i] = internal_inArgs.get_p(p_indices_[i]);
    }
    Teuchos::RCP<Thyra::DefaultProductVector<Real>> p_prod = Thyra::defaultProductVector<double>(p_space, p_vecs());

    result.set_p(0, p_prod);

    return result;
}

template <typename Real>
Thyra::ModelEvaluatorBase::InArgs<Real>
ProductModelEvaluator<Real>::getUpperBounds() const
{
    Thyra::ModelEvaluatorBase::InArgs<Real> internal_inArgs = thyra_model_->getUpperBounds();
    Thyra::ModelEvaluatorBase::InArgsSetup<Real> result; 
    result.setModelEvalDescription(this->description());
    result.set_Np_Ng(1, thyra_model_->Ng());

    this->fromInternalInArgs(internal_inArgs, result);
    result.setArgs(internal_inArgs, true);

    Teuchos::RCP<const Thyra::DefaultProductVectorSpace<Real>> p_space = Teuchos::rcp_dynamic_cast<const Thyra::DefaultProductVectorSpace<Real>>(this->get_p_space(0));

    Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<Real>>> p_vecs(p_indices_.size());
    for (auto i = 0; i < p_indices_.size(); ++i) {
        p_vecs[i] = internal_inArgs.get_p(p_indices_[i]);
    }
    Teuchos::RCP<Thyra::DefaultProductVector<Real>> p_prod = Thyra::defaultProductVector<double>(p_space, p_vecs());

    result.set_p(0, p_prod);

    return result;
}

template <typename Real>
int
ProductModelEvaluator<Real>::Np() const
{
    return 1;
}

template <typename Real>
int
ProductModelEvaluator<Real>::Ng() const
{
    return thyra_model_->Ng();
}

template <typename Real>
void
ProductModelEvaluator<Real>::reportFinalPoint(
    const Thyra::ModelEvaluatorBase::InArgs<Real>& finalPoint,
    const bool wasSolved)
{
    return thyra_model_->reportFinalPoint(finalPoint, wasSolved);
}

template <typename Real>
void
ProductModelEvaluator<Real>::fromInternalInArgs(const Thyra::ModelEvaluatorBase::InArgs<Real>& inArgs1, Thyra::ModelEvaluatorBase::InArgsSetup<Real>& inArgs2) const
{
    inArgs2.setSupports(Thyra::ModelEvaluator<Real>::IN_ARG_x_dot_dot, inArgs1.supports(Thyra::ModelEvaluator<Real>::IN_ARG_x_dot_dot));
    inArgs2.setSupports(Thyra::ModelEvaluator<Real>::IN_ARG_x_dot, inArgs1.supports(Thyra::ModelEvaluator<Real>::IN_ARG_x_dot));
    inArgs2.setSupports(Thyra::ModelEvaluator<Real>::IN_ARG_x, inArgs1.supports(Thyra::ModelEvaluator<Real>::IN_ARG_x));
    inArgs2.setSupports(Thyra::ModelEvaluator<Real>::IN_ARG_x_dot_poly, inArgs1.supports(Thyra::ModelEvaluator<Real>::IN_ARG_x_dot_poly));
    inArgs2.setSupports(Thyra::ModelEvaluator<Real>::IN_ARG_x_poly, inArgs1.supports(Thyra::ModelEvaluator<Real>::IN_ARG_x_poly));
    inArgs2.setSupports(Thyra::ModelEvaluator<Real>::IN_ARG_x_dot_mp, inArgs1.supports(Thyra::ModelEvaluator<Real>::IN_ARG_x_dot_mp));
    inArgs2.setSupports(Thyra::ModelEvaluator<Real>::IN_ARG_x_mp, inArgs1.supports(Thyra::ModelEvaluator<Real>::IN_ARG_x_mp)); 
    inArgs2.setSupports(Thyra::ModelEvaluator<Real>::IN_ARG_t, inArgs1.supports(Thyra::ModelEvaluator<Real>::IN_ARG_t)); 
    inArgs2.setSupports(Thyra::ModelEvaluator<Real>::IN_ARG_alpha, inArgs1.supports(Thyra::ModelEvaluator<Real>::IN_ARG_alpha)); 
    inArgs2.setSupports(Thyra::ModelEvaluator<Real>::IN_ARG_beta, inArgs1.supports(Thyra::ModelEvaluator<Real>::IN_ARG_beta)); 
    inArgs2.setSupports(Thyra::ModelEvaluator<Real>::IN_ARG_W_x_dot_dot_coeff, inArgs1.supports(Thyra::ModelEvaluator<Real>::IN_ARG_W_x_dot_dot_coeff)); 
    inArgs2.setSupports(Thyra::ModelEvaluator<Real>::IN_ARG_step_size, inArgs1.supports(Thyra::ModelEvaluator<Real>::IN_ARG_step_size)); 
    inArgs2.setSupports(Thyra::ModelEvaluator<Real>::IN_ARG_stage_number, inArgs1.supports(Thyra::ModelEvaluator<Real>::IN_ARG_stage_number)); 
}

template <typename Real>
void
ProductModelEvaluator<Real>::toInternalInArgs(const Thyra::ModelEvaluatorBase::InArgs<Real>& inArgs1, Thyra::ModelEvaluatorBase::InArgsSetup<Real>& inArgs2) const
{

}

template <typename Real>
void
ProductModelEvaluator<Real>::fromInternalOutArgs(const Thyra::ModelEvaluatorBase::OutArgs<Real>& outArgs1, Thyra::ModelEvaluatorBase::OutArgsSetup<Real>& outArgs2) const
{

    outArgs2.setSupports(Thyra::ModelEvaluator<Real>::OUT_ARG_f, outArgs1.supports(Thyra::ModelEvaluator<Real>::OUT_ARG_f));
    outArgs2.setSupports(Thyra::ModelEvaluator<Real>::OUT_ARG_W, outArgs1.supports(Thyra::ModelEvaluator<Real>::OUT_ARG_W));
    outArgs2.setSupports(Thyra::ModelEvaluator<Real>::OUT_ARG_f_mp, outArgs1.supports(Thyra::ModelEvaluator<Real>::OUT_ARG_f_mp));
    outArgs2.setSupports(Thyra::ModelEvaluator<Real>::OUT_ARG_W_mp, outArgs1.supports(Thyra::ModelEvaluator<Real>::OUT_ARG_W_mp));
    outArgs2.setSupports(Thyra::ModelEvaluator<Real>::OUT_ARG_W_op, outArgs1.supports(Thyra::ModelEvaluator<Real>::OUT_ARG_W_op));
    outArgs2.setSupports(Thyra::ModelEvaluator<Real>::OUT_ARG_W_prec, outArgs1.supports(Thyra::ModelEvaluator<Real>::OUT_ARG_W_prec));
    outArgs2.setSupports(Thyra::ModelEvaluator<Real>::OUT_ARG_f_poly, outArgs1.supports(Thyra::ModelEvaluator<Real>::OUT_ARG_f_poly));

    outArgs2.setSupports(Thyra::ModelEvaluator<Real>::OUT_ARG_DgDx, g_index_, outArgs1.supports(Thyra::ModelEvaluator<Real>::OUT_ARG_DgDx, g_index_));

    bool all_mv_gradient_form = false;
    bool all_mv_jacobian_form = false;
    for (auto i = 0; i < p_indices_.size(); ++i) {
      const Thyra::ModelEvaluatorBase::DerivativeSupport dgdp_support =
          outArgs1.supports(Thyra::ModelEvaluatorBase::OUT_ARG_DgDp, g_index_, p_indices_[i]);
      if (dgdp_support.supports(Thyra::ModelEvaluatorBase::DERIV_MV_GRADIENT_FORM)) {
        if (i == 0) all_mv_gradient_form = true;
        if (!all_mv_gradient_form)
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                        std::endl <<
                        "Piro::ThyraProductME_Objective::gradient_2, DgDp does support neither DERIV_MV_JACOBIAN_FORM nor DERIV_MV_GRADIENT_FORM forms" << std::endl);
      }
      else if(dgdp_support.supports(Thyra::ModelEvaluatorBase::DERIV_MV_JACOBIAN_FORM)) {
        if (i == 0) all_mv_jacobian_form = true;
        if (!all_mv_jacobian_form)
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                        std::endl <<
                        "Piro::ThyraProductME_Objective::gradient_2, DgDp does support neither DERIV_MV_JACOBIAN_FORM nor DERIV_MV_GRADIENT_FORM forms" << std::endl);
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                    std::endl <<
                    "Piro::ThyraProductME_Objective::gradient_2, DgDp does support neither DERIV_MV_JACOBIAN_FORM nor DERIV_MV_GRADIENT_FORM forms" << std::endl);
      }
    }
    outArgs2.setSupports(Thyra::ModelEvaluator<Real>::OUT_ARG_DgDp, g_index_, 0, outArgs1.supports(Thyra::ModelEvaluator<Real>::OUT_ARG_DgDp, g_index_, p_indices_[0]));

    outArgs2.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_DfDp, 0, outArgs1.supports(Thyra::ModelEvaluatorBase::OUT_ARG_DfDp, p_indices_[0]));
}

template <typename Real>
void
ProductModelEvaluator<Real>::toInternalOutArgs(const Thyra::ModelEvaluatorBase::OutArgs<Real>& outArgs1, Thyra::ModelEvaluatorBase::OutArgsSetup<Real>& outArgs2) const
{

}

} // namespace Piro

#endif