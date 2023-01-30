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
#include "Thyra_DefaultProductMultiVector.hpp"
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

    void block_diagonal_hessian_22(const Teuchos::RCP<Thyra::PhysicallyBlockedLinearOpBase<Real>> H,
                    const ROL::Vector<Real> &u,
                    const ROL::Vector<Real> &z,
                    const int g_idx) const;

    /** \brief . */
    Teuchos::RCP<Thyra::LinearOpBase<Real> > create_DfDp_op(int l) const;
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
    p_indices_(p_indices),
    Thyra::ModelEvaluatorDelegatorBase<Real>(thyra_model)
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
    Thyra::ModelEvaluatorBase::InArgsSetup<Real> internal_inArgs;
    Thyra::ModelEvaluatorBase::OutArgsSetup<Real> internal_outArgs;

    internal_inArgs.set_Np_Ng(thyra_model_->Np(), thyra_model_->Ng());
    internal_outArgs.set_Np_Ng(thyra_model_->Np(), thyra_model_->Ng());

    bool supports_vec_prod_g_xp = outArgs.supports(Thyra::ModelEvaluatorBase::OUT_ARG_hess_vec_prod_g_xp, g_index_, 0);
    bool supports_vec_prod_g_px = outArgs.supports(Thyra::ModelEvaluatorBase::OUT_ARG_hess_vec_prod_g_px, g_index_, 0);
    bool supports_vec_prod_g_pp = outArgs.supports(Thyra::ModelEvaluatorBase::OUT_ARG_hess_vec_prod_g_pp, g_index_, 0, 0);
    
    bool supports_vec_prod_f_xp = outArgs.supports(Thyra::ModelEvaluatorBase::OUT_ARG_hess_vec_prod_f_xp, 0);
    bool supports_vec_prod_f_px = outArgs.supports(Thyra::ModelEvaluatorBase::OUT_ARG_hess_vec_prod_f_px, 0);
    bool supports_vec_prod_f_pp = outArgs.supports(Thyra::ModelEvaluatorBase::OUT_ARG_hess_vec_prod_f_pp, 0, 0);

    this->toInternalInArgs(inArgs, internal_inArgs);
    this->toInternalOutArgs(outArgs, internal_outArgs);

    internal_outArgs.setArgs(outArgs, true);
    internal_inArgs.setArgs(inArgs, true);

    Teuchos::RCP<const Thyra::ProductVectorBase<Real> > prodvec_p = Teuchos::rcp_dynamic_cast<const Thyra::ProductVectorBase<Real>>(inArgs.get_p(0));
    Teuchos::RCP<const Thyra::ProductMultiVectorBase<Real> > prodvec_direction_p = Teuchos::rcp_dynamic_cast<const Thyra::ProductMultiVectorBase<Real>>(inArgs.get_p_direction(0));

    for (auto i = 0; i < p_indices_.size(); ++i) {
        auto tmp = prodvec_p->getVectorBlock(i);

        Teuchos::RCP<const Thyra::ProductVectorBase<Real> > prodvec_p_in
            = Teuchos::rcp_dynamic_cast<const Thyra::ProductVectorBase<Real>>(tmp);

        TEUCHOS_TEST_FOR_EXCEPTION(!prodvec_p_in.is_null(), std::logic_error,
            std::endl <<
            "Error!  ProductModelEvaluator<Real>::evalModelImpl() " <<
            " ProductVectorBase of ProductVectorBase is not supported.  Parameter index i = " <<
            i << std::endl);

        internal_inArgs.set_p(p_indices_[i], prodvec_p->getVectorBlock(i));
        if (!prodvec_direction_p.is_null())
            internal_inArgs.set_p_direction(p_indices_[i], prodvec_direction_p->getMultiVectorBlock(i));
    }

    for (auto g_index = 0; g_index < thyra_model_->Ng(); ++g_index) {
        auto dgdp = outArgs.get_DgDp(g_index, 0).getMultiVector();
        if (Teuchos::is_null(dgdp))
            continue;
        Teuchos::RCP<Thyra::ProductMultiVectorBase<Real> > prodvec_dgdp =
            Teuchos::rcp_dynamic_cast<Thyra::ProductMultiVectorBase<Real>>(dgdp);
        if (Teuchos::is_null(prodvec_dgdp))
            continue;
        for (auto i = 0; i < p_indices_.size(); ++i) {
            const Thyra::ModelEvaluatorBase::DerivativeSupport dgdp_support =
                internal_outArgs.supports(Thyra::ModelEvaluatorBase::OUT_ARG_DgDp, g_index, p_indices_[i]);
            Thyra::ModelEvaluatorBase::EDerivativeMultiVectorOrientation dgdp_orient;
            if (dgdp_support.supports(Thyra::ModelEvaluatorBase::DERIV_MV_GRADIENT_FORM))
                dgdp_orient = Thyra::ModelEvaluatorBase::DERIV_MV_GRADIENT_FORM;
            else if(dgdp_support.supports(Thyra::ModelEvaluatorBase::DERIV_MV_JACOBIAN_FORM))
                dgdp_orient = Thyra::ModelEvaluatorBase::DERIV_MV_JACOBIAN_FORM;
            else {
            ROL_TEST_FOR_EXCEPTION(true, std::logic_error,
                "Piro::ThyraProductME_Objective::gradient_2, DgDp does support neither DERIV_MV_JACOBIAN_FORM nor DERIV_MV_GRADIENT_FORM forms");
            }
            internal_outArgs.set_DgDp(g_index, 
                                    p_indices_[i], 
                                    Thyra::ModelEvaluatorBase::DerivativeMultiVector<Real>(prodvec_dgdp->getNonconstMultiVectorBlock(i), 
                                                                                            dgdp_orient));
        }
    }

    if (supports_vec_prod_g_xp) {
        std::vector<Teuchos::RCP< Thyra::MultiVectorBase<Real> > > hv_vec(p_indices_.size());

        hv_vec[0] = outArgs.get_hess_vec_prod_g_xp(g_index_,0);
        for(std::size_t j=1; j<p_indices_.size(); ++j) {
            hv_vec[j] = hv_vec[0]->clone_mv();
        }

        for(std::size_t j=0; j<p_indices_.size(); ++j) {
            internal_outArgs.set_hess_vec_prod_g_xp(g_index_,p_indices_[j], hv_vec[j]);
        }
    }

    if (supports_vec_prod_g_px) {
        Teuchos::RCP< Thyra::ProductMultiVectorBase<Real> > prodvec_hv =
            Teuchos::rcp_dynamic_cast<Thyra::ProductMultiVectorBase<Real>>(outArgs.get_hess_vec_prod_g_px(g_index_,0));

        for(std::size_t i=0; i<p_indices_.size(); ++i) {
            bool supports_deriv_j =   internal_outArgs.supports(Thyra::ModelEvaluatorBase::OUT_ARG_hess_vec_prod_g_px, g_index_, p_indices_[i]);
            ROL_TEST_FOR_EXCEPTION( !supports_deriv_j, std::logic_error, "Piro::ThyraProductME_Objective_SimOpt: H_px product vector is not supported");
            internal_outArgs.set_hess_vec_prod_g_px(g_index_,p_indices_[i], prodvec_hv->getNonconstMultiVectorBlock(i));
        }
    }

    if (supports_vec_prod_g_pp) {
        Teuchos::RCP< Thyra::ProductMultiVectorBase<Real> > prodvec_hv =
            Teuchos::rcp_dynamic_cast<Thyra::ProductMultiVectorBase<Real>>(outArgs.get_hess_vec_prod_g_pp(g_index_,0,0));
        std::vector<std::vector<Teuchos::RCP< Thyra::MultiVectorBase<Real> > > > hv_vec(p_indices_.size());

        for(std::size_t i=0; i<p_indices_.size(); ++i) {
            hv_vec[i].resize(p_indices_.size());
            hv_vec[i][0] = prodvec_hv->getNonconstMultiVectorBlock(i);
            for(std::size_t j=1; j<p_indices_.size(); ++j) {
                hv_vec[i][j] = hv_vec[i][0]->clone_mv();
            }
        }

        for(std::size_t i=0; i<p_indices_.size(); ++i) {
            for(std::size_t j=0; j<p_indices_.size(); ++j) {
                bool supports_deriv_j =   internal_outArgs.supports(Thyra::ModelEvaluatorBase::OUT_ARG_hess_vec_prod_g_pp, g_index_, p_indices_[i], p_indices_[j]);
                ROL_TEST_FOR_EXCEPTION( !supports_deriv_j, std::logic_error, "Piro::ThyraProductME_Objective_SimOpt: H_pp product vector is not supported");

                internal_outArgs.set_hess_vec_prod_g_pp(g_index_,p_indices_[i], p_indices_[j], hv_vec[i][j]);
            }
        }
    }

    if (supports_vec_prod_f_xp) {
        Teuchos::RCP< Thyra::MultiVectorBase<Real> > thyra_ahwv = outArgs.get_hess_vec_prod_f_xp(0);
        std::vector<Teuchos::RCP< Thyra::MultiVectorBase<Real> > > ahwv_vec(p_indices_.size());

        ahwv_vec[0] = thyra_ahwv;
        for(std::size_t j=1; j<p_indices_.size(); ++j) {
            ahwv_vec[j] = thyra_ahwv->clone_mv();
        }

        for(std::size_t j=0; j<p_indices_.size(); ++j) {
            bool supports_deriv_j =   internal_outArgs.supports(Thyra::ModelEvaluatorBase::OUT_ARG_hess_vec_prod_f_xp, p_indices_[j]);
            ROL_TEST_FOR_EXCEPTION( !supports_deriv_j, std::logic_error, "Piro::ThyraProductME_Constraint_SimOpt: H_xp product vector is not supported");
            internal_outArgs.set_hess_vec_prod_f_xp(p_indices_[j], ahwv_vec[j]);
        }
    }

    if (supports_vec_prod_f_px) {
        Teuchos::RCP< Thyra::ProductVectorBase<Real> > prodvec_ahwv =
            Teuchos::rcp_dynamic_cast<Thyra::ProductVectorBase<Real>>(outArgs.get_hess_vec_prod_f_px(0));
        for(std::size_t i=0; i<p_indices_.size(); ++i) {
            bool supports_deriv_i =   internal_outArgs.supports(Thyra::ModelEvaluatorBase::OUT_ARG_hess_vec_prod_f_px, p_indices_[i]);
            ROL_TEST_FOR_EXCEPTION( !supports_deriv_i, std::logic_error, "Piro::ThyraProductME_Constraint_SimOpt: H_px product vector is not supported");
            internal_outArgs.set_hess_vec_prod_f_px(p_indices_[i], prodvec_ahwv->getNonconstVectorBlock(i));
        }
    }

    if (supports_vec_prod_f_pp) {
        Teuchos::RCP< Thyra::ProductMultiVectorBase<Real> > prodvec_ahwv =
            Teuchos::rcp_dynamic_cast<Thyra::ProductMultiVectorBase<Real>>(outArgs.get_hess_vec_prod_f_pp(0,0));
        std::vector<std::vector<Teuchos::RCP< Thyra::MultiVectorBase<Real> > > > ahwv_vec(p_indices_.size());

        for(std::size_t i=0; i<p_indices_.size(); ++i) {
            ahwv_vec[i].resize(p_indices_.size());
            ahwv_vec[i][0] = prodvec_ahwv->getNonconstMultiVectorBlock(i);
            for(std::size_t j=1; j<p_indices_.size(); ++j) {
                ahwv_vec[i][j] = ahwv_vec[i][0]->clone_mv();
            }
        }

        for(std::size_t i=0; i<p_indices_.size(); ++i) {
            for(std::size_t j=0; j<p_indices_.size(); ++j) {
                bool supports_deriv_ij =   internal_outArgs.supports(Thyra::ModelEvaluatorBase::OUT_ARG_hess_vec_prod_f_pp, p_indices_[i], p_indices_[j]);
                ROL_TEST_FOR_EXCEPTION( !supports_deriv_ij, std::logic_error, "Piro::ThyraProductME_Constraint_SimOpt: H_pp product vector is not supported");

                internal_outArgs.set_hess_vec_prod_f_pp(p_indices_[i], p_indices_[j], ahwv_vec[i][j]);
            }
        }        
    }

    thyra_model_->evalModel(internal_inArgs,internal_outArgs);

    if (supports_vec_prod_g_xp) {
        Teuchos::RCP< Thyra::MultiVectorBase<Real> > hv_vec = internal_outArgs.get_hess_vec_prod_g_xp(g_index_,p_indices_[0]);
        for(std::size_t j=1; j<p_indices_.size(); ++j)
            hv_vec->update(1.0, *internal_outArgs.get_hess_vec_prod_g_xp(g_index_,p_indices_[j]));
        //outArgs.set_hess_vec_prod_g_xp(g_index_,0,hv_vec);
    }

    if (supports_vec_prod_g_pp) {
        for(std::size_t i=0; i<p_indices_.size(); ++i) {
            Teuchos::RCP< Thyra::MultiVectorBase<Real> > hv_vec = internal_outArgs.get_hess_vec_prod_g_pp(g_index_,p_indices_[i],p_indices_[0]);
            for(std::size_t j=1; j<p_indices_.size(); ++j)
                hv_vec->update(1.0, *internal_outArgs.get_hess_vec_prod_g_pp(g_index_,p_indices_[i],p_indices_[j]));
        }
    }

    if (supports_vec_prod_f_xp) {
        Teuchos::RCP< Thyra::MultiVectorBase<Real> > ahwv_vec = internal_outArgs.get_hess_vec_prod_f_xp(p_indices_[0]);
        for(std::size_t j=1; j<p_indices_.size(); ++j)
            ahwv_vec->update(1.0, *internal_outArgs.get_hess_vec_prod_f_xp(p_indices_[j]));    
    }

    if (supports_vec_prod_f_pp) {
        for(std::size_t i=0; i<p_indices_.size(); ++i) {
            Teuchos::RCP< Thyra::MultiVectorBase<Real> > ahwv_vec = internal_outArgs.get_hess_vec_prod_f_pp(p_indices_[i],p_indices_[0]);
            for(std::size_t j=1; j<p_indices_.size(); ++j)
                ahwv_vec->update(1.0, *internal_outArgs.get_hess_vec_prod_f_pp(p_indices_[i],p_indices_[j]));
        }
    }
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
Teuchos::RCP<Thyra::LinearOpBase<Real> > 
ProductModelEvaluator<Real>::create_DfDp_op(int l) const {
    return thyra_model_->create_DfDp_op(0);
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

    bool all_hess_g_pp = false;
    for (auto i = 0; i < p_indices_.size(); ++i) {
        if (outArgs1.supports(Thyra::ModelEvaluatorBase::OUT_ARG_hess_g_pp, g_index_, p_indices_[i], p_indices_[i])) {
            if (i == 0) all_hess_g_pp = true;
            if (!all_hess_g_pp)
                TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                        std::endl <<
                        "Piro::ThyraProductME_Objective::gradient_2, DgDp does support neither DERIV_MV_JACOBIAN_FORM nor DERIV_MV_GRADIENT_FORM forms" << std::endl);
        }
        else {
            if (all_hess_g_pp)
                TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                        std::endl <<
                        "Piro::ThyraProductME_Objective::gradient_2, DgDp does support neither DERIV_MV_JACOBIAN_FORM nor DERIV_MV_GRADIENT_FORM forms" << std::endl);
        }
    }
    outArgs2.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_hess_g_pp, g_index_, 0, 0, outArgs1.supports(Thyra::ModelEvaluatorBase::OUT_ARG_hess_g_pp, g_index_, p_indices_[0], p_indices_[0]));
}

template <typename Real>
void
ProductModelEvaluator<Real>::toInternalOutArgs(const Thyra::ModelEvaluatorBase::OutArgs<Real>& outArgs1, Thyra::ModelEvaluatorBase::OutArgsSetup<Real>& outArgs2) const
{
    outArgs2.setSupports(Thyra::ModelEvaluator<Real>::OUT_ARG_f, outArgs1.supports(Thyra::ModelEvaluator<Real>::OUT_ARG_f));
    outArgs2.setSupports(Thyra::ModelEvaluator<Real>::OUT_ARG_W, outArgs1.supports(Thyra::ModelEvaluator<Real>::OUT_ARG_W));
    outArgs2.setSupports(Thyra::ModelEvaluator<Real>::OUT_ARG_f_mp, outArgs1.supports(Thyra::ModelEvaluator<Real>::OUT_ARG_f_mp));
    outArgs2.setSupports(Thyra::ModelEvaluator<Real>::OUT_ARG_W_mp, outArgs1.supports(Thyra::ModelEvaluator<Real>::OUT_ARG_W_mp));
    outArgs2.setSupports(Thyra::ModelEvaluator<Real>::OUT_ARG_W_op, outArgs1.supports(Thyra::ModelEvaluator<Real>::OUT_ARG_W_op));
    outArgs2.setSupports(Thyra::ModelEvaluator<Real>::OUT_ARG_W_prec, outArgs1.supports(Thyra::ModelEvaluator<Real>::OUT_ARG_W_prec));
    outArgs2.setSupports(Thyra::ModelEvaluator<Real>::OUT_ARG_f_poly, outArgs1.supports(Thyra::ModelEvaluator<Real>::OUT_ARG_f_poly));

    outArgs2.setSupports(Thyra::ModelEvaluator<Real>::OUT_ARG_DgDx, g_index_, outArgs1.supports(Thyra::ModelEvaluator<Real>::OUT_ARG_DgDx, g_index_));

    for (auto i = 0; i < p_indices_.size(); ++i) {
        outArgs2.setSupports(Thyra::ModelEvaluator<Real>::OUT_ARG_DgDp, g_index_, p_indices_[i], outArgs1.supports(Thyra::ModelEvaluator<Real>::OUT_ARG_DgDp, g_index_, 0));

        outArgs2.setSupports(Thyra::ModelEvaluatorBase::OUT_ARG_DfDp, p_indices_[i], outArgs1.supports(Thyra::ModelEvaluatorBase::OUT_ARG_DfDp, 0));
    }
}

template <typename Real>
void
ProductModelEvaluator<Real>::block_diagonal_hessian_22(const Teuchos::RCP<Thyra::PhysicallyBlockedLinearOpBase<Real>> H,
                    const ROL::Vector<Real> &u,
                    const ROL::Vector<Real> &z,
                    const int g_idx) const
{
    Thyra::ModelEvaluatorBase::OutArgs<Real> outArgs = thyra_model_->createOutArgs();
    bool supports_deriv = true;
    for(std::size_t i=0; i<p_indices_.size(); ++i)
      supports_deriv = supports_deriv &&  outArgs.supports(Thyra::ModelEvaluatorBase::OUT_ARG_hess_g_pp, g_idx, p_indices_[i], p_indices_[i]);
    
    ROL_TEST_FOR_EXCEPTION( !supports_deriv, std::logic_error, "Piro::ProductModelEvaluator: H_pp is not supported");

    const ROL::ThyraVector<Real>  & thyra_p = dynamic_cast<const ROL::ThyraVector<Real>&>(z);
    ROL::Ptr<ROL::Vector<Real>> unew = u.clone();
    unew->set(u);
    const ROL::ThyraVector<Real>  & thyra_x = dynamic_cast<const ROL::ThyraVector<Real>&>(*unew);

    Teuchos::RCP<const  Thyra::ProductVectorBase<Real> > thyra_prodvec_p = Teuchos::rcp_dynamic_cast<const Thyra::ProductVectorBase<Real>>(thyra_p.getVector());

    Thyra::ModelEvaluatorBase::InArgs<Real> inArgs = thyra_model_->createInArgs();

    H->beginBlockFill(p_indices_.size(), p_indices_.size());

    for(std::size_t i=0; i<p_indices_.size(); ++i) {
      inArgs.set_p(p_indices_[i], thyra_prodvec_p->getVectorBlock(i));
    }
    inArgs.set_x(thyra_x.getVector());

    Teuchos::RCP< Thyra::VectorBase<Real> > multiplier_g = Thyra::createMember<Real>(thyra_model_->get_g_multiplier_space(g_idx));
    Thyra::put_scalar(1.0, multiplier_g.ptr());
    inArgs.set_g_multiplier(g_idx, multiplier_g);

    for(std::size_t i=0; i<p_indices_.size(); ++i) {
      Teuchos::RCP<Thyra::LinearOpBase<Real>> hess_g_pp = thyra_model_->create_hess_g_pp(g_idx, p_indices_[i], p_indices_[i]);
      outArgs.set_hess_g_pp(g_idx, p_indices_[i], p_indices_[i], hess_g_pp);
      H->setBlock(i, i, hess_g_pp);
    }
    H->endBlockFill();

    thyra_model_->evalModel(inArgs, outArgs);
  }

} // namespace Piro

#endif