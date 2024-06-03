#pragma once

#include <ceres/ceres.h>
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/utils/sophus_utils.hpp"
#include "utility.h"

#include "GaussianProcess.hpp"

class PoseGPFactor : public ceres::CostFunction
{
public:

    PoseGPFactor(const SE3d &pose_meas_, double wR_, double wP_, double Dt_, double s_)
    :   pose_meas   (pose_meas_      ),
        wR          (wR_             ),
        wP          (wP_             ),
        Dt          (Dt_             ),
        s           (s_              ),
        gpm         (Dt_             )
    {
        // 6-element residual: (3x1 omega, 3x1 a, 3x1 bw, 3x1 ba)
        set_num_residuals(6);

        // Rotation of the first knot
        mutable_parameter_block_sizes()->push_back(4);
        // Angular velocity of the first knot
        mutable_parameter_block_sizes()->push_back(3);
        // Position of the first knot
        mutable_parameter_block_sizes()->push_back(3);
        // Velocity of the first knot
        mutable_parameter_block_sizes()->push_back(3);
        // Acceleration of the first knot
        mutable_parameter_block_sizes()->push_back(3);

        // Rotation of the second knot
        mutable_parameter_block_sizes()->push_back(4);
        // Angular velocity of the second knot
        mutable_parameter_block_sizes()->push_back(3);
        // Position of the second knot
        mutable_parameter_block_sizes()->push_back(3);
        // Velocity of the second knot
        mutable_parameter_block_sizes()->push_back(3);
        // Acceleration of the second knot
        mutable_parameter_block_sizes()->push_back(3);
    }

    Matrix3d Jr(const Vector3d &phi) const
    {
        Matrix3d Jr;
        Sophus::rightJacobianSO3(phi, Jr);
        return Jr;
    }

    Matrix3d JrInv(const Vector3d &phi) const
    {
        Matrix3d JrInv;
        Sophus::rightJacobianInvSO3(phi, JrInv);
        return JrInv;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        SO3d Ra = Eigen::Map<SO3d const>(parameters[RaIdx]);
        Vec3 Oa = Eigen::Map<Vec3 const>(parameters[OaIdx]);
        Vec3 Pa = Eigen::Map<Vec3 const>(parameters[PaIdx]);
        Vec3 Va = Eigen::Map<Vec3 const>(parameters[VaIdx]);
        Vec3 Aa = Eigen::Map<Vec3 const>(parameters[AaIdx]);

        SO3d Rb = Eigen::Map<SO3d const>(parameters[RbIdx]);
        Vec3 Ob = Eigen::Map<Vec3 const>(parameters[ObIdx]);
        Vec3 Pb = Eigen::Map<Vec3 const>(parameters[PbIdx]);
        Vec3 Vb = Eigen::Map<Vec3 const>(parameters[VbIdx]);
        Vec3 Ab = Eigen::Map<Vec3 const>(parameters[AbIdx]);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the residual ---------------------------------------------------------------------------*/
        
        MatrixXd LAM_ROt = gpm.LAMBDA_RO(s*Dt);
        MatrixXd PSI_ROt = gpm.PSI_RO(s*Dt);

        MatrixXd LAM_PVAt = gpm.LAMBDA_PVA(s*Dt);
        MatrixXd PSI_PVAt = gpm.PSI_PVA(s*Dt);

        Matrix3d LAM_RO11 = LAM_ROt.block<3, 3>(0, 0);
        Matrix3d LAM_RO12 = LAM_ROt.block<3, 3>(0, 3);
        Matrix3d LAM_RO21 = LAM_ROt.block<3, 3>(3, 0);
        Matrix3d LAM_RO22 = LAM_ROt.block<3, 3>(3, 3);
        
        Matrix3d PSI_RO11 = PSI_ROt.block<3, 3>(0, 0);
        Matrix3d PSI_RO12 = PSI_ROt.block<3, 3>(0, 3);
        // Matrix3d PSI_RO21 = PSI_ROt.block<3, 3>(3, 0);
        // Matrix3d PSI_RO22 = PSI_ROt.block<3, 3>(3, 3);

        Matrix3d LAM_PVA11 = LAM_PVAt.block<3, 3>(0, 0);
        Matrix3d LAM_PVA12 = LAM_PVAt.block<3, 3>(0, 3);
        Matrix3d LAM_PVA13 = LAM_PVAt.block<3, 3>(0, 6);
        // Matrix3d LAM_PVA21 = LAM_PVAt.block<3, 3>(3, 0);
        // Matrix3d LAM_PVA22 = LAM_PVAt.block<3, 3>(3, 3);
        // Matrix3d LAM_PVA23 = LAM_PVAt.block<3, 3>(3, 6);
        // Matrix3d LAM_PVA31 = LAM_PVAt.block<3, 3>(6, 0);
        // Matrix3d LAM_PVA32 = LAM_PVAt.block<3, 3>(6, 3);
        // Matrix3d LAM_PVA33 = LAM_PVAt.block<3, 3>(6, 6);

        Matrix3d PSI_PVA11 = PSI_PVAt.block<3, 3>(0, 0);
        Matrix3d PSI_PVA12 = PSI_PVAt.block<3, 3>(0, 3);
        Matrix3d PSI_PVA13 = PSI_PVAt.block<3, 3>(0, 6);
        // Matrix3d PSI_PVA21 = PSI_PVAt.block<3, 3>(3, 0);
        // Matrix3d PSI_PVA22 = PSI_PVAt.block<3, 3>(3, 3);
        // Matrix3d PSI_PVA23 = PSI_PVAt.block<3, 3>(3, 6);
        // Matrix3d PSI_PVA31 = PSI_PVAt.block<3, 3>(6, 0);
        // Matrix3d PSI_PVA32 = PSI_PVAt.block<3, 3>(6, 3);
        // Matrix3d PSI_PVA33 = PSI_PVAt.block<3, 3>(6, 6);

        // Find the relative rotation 
        SO3d Rab = Ra.inverse()*Rb;

        // Calculate the interpolated rotation
        Vector3d thetaa    = Vector3d(0, 0, 0);
        Vector3d thetadota = Oa;
        Vector3d thetab    = Rab.log();
        Vector3d thetadotb = JrInv(thetab)*Ob;
        
        // Calculate the knot manifold states
        Matrix<double, 6, 1> gammaa; gammaa << thetaa, thetadota;
        Matrix<double, 6, 1> gammab; gammab << thetab, thetadotb;

        // Calculate the knot euclid states
        Matrix<double, 9, 1> pvaa; pvaa << Pa, Va, Aa;
        Matrix<double, 9, 1> pvab; pvab << Pb, Vb, Ab;

        Matrix<double, 6, 1> gammat = LAM_ROt*gammaa + PSI_ROt*gammab;
        Matrix<double, 9, 1> pvat = LAM_PVAt*pvaa + PSI_PVAt*pvab;

        Vector3d thetat    = gammat.block<3, 1>(0, 0);
        Vector3d thetadott = gammat.block<3, 1>(3, 0);

        SO3d Rt = Ra*SO3d::exp(thetat);
        // Vector3d Ot = JrInv(thetat)*thetadott;
        Vector3d Pt = pvat.block<3, 1>(0, 0);
        Vector3d Vt = pvat.block<3, 1>(3, 0);
        Vector3d At = pvat.block<3, 1>(6, 0);

        // Rotational residual
        Vector3d rR = (pose_meas.so3().inverse()*Rt).log();

        // Positional residual
        Vector3d rP = (Pt - pose_meas.translation());

        // Residual
        Eigen::Map<Matrix<double, 6, 1>> residual(residuals);
        residual.block<3, 1>(0, 0) = wR*rR;
        residual.block<3, 1>(3, 0) = wP*rP;

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        if (!jacobians)
            return true;

        Matrix3d DrR_DRt = JrInv(rR);
        // Matrix3d DrR_DPt = Matrix3d::Identity(3,3);
        
        Matrix3d DRt_DRa = SO3d::exp(-thetat).matrix() - Jr(thetat)*(PSI_RO11 - PSI_RO12*SO3d::hat(Ob))*JrInv(thetab)*Rab.inverse().matrix();
        Matrix3d DRt_DOa = Jr(thetat)*LAM_RO12;

        Matrix3d DRt_DRb = Jr(thetat)*(PSI_RO11 - PSI_RO12*SO3d::hat(Ob))*JrInv(thetab);
        Matrix3d DRt_DOb = Jr(thetat)*PSI_RO12*JrInv(thetab);

        // Matrix3d DrR_Dthetat = JrInv(rR)*Jr(thetat);

        size_t idx;

        // Jacobian on Ra
        idx = RaIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> Dr_DRa(jacobians[idx]);
            Dr_DRa.setZero();
            Dr_DRa.block<3, 3>(0, 0) = wR*DrR_DRt*DRt_DRa;
        }
        
        // Jacobian on Oa
        idx = OaIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DOa(jacobians[idx]);
            Dr_DOa.setZero();
            Dr_DOa.block<3, 3>(0, 0) = wR*DrR_DRt*DRt_DOa;
        }

        // Jacobian on Rb
        idx = RbIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> Dr_DRb(jacobians[idx]);
            Dr_DRb.setZero();
            Dr_DRb.block<3, 3>(0, 0) = wR*DrR_DRt*DRt_DRb;
        }

        // Jacobian on Ob
        idx = ObIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DOb(jacobians[idx]);
            Dr_DOb.setZero();
            Dr_DOb.block<3, 3>(0, 0) = wR*DrR_DRt*DRt_DOb;
        }

        // Jacobian on Pa
        idx = PaIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DPa(jacobians[idx]);
            Dr_DPa.setZero();
            Dr_DPa.block<3, 3>(3, 0) = wP*LAM_PVA11;
        }

        // Jacobian on Va
        idx = VaIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DVa(jacobians[idx]);
            Dr_DVa.setZero();
            Dr_DVa.block<3, 3>(3, 0) = wP*LAM_PVA12;
        }

        // Jacobian on Aa
        idx = AaIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DAa(jacobians[idx]);
            Dr_DAa.setZero();
            Dr_DAa.block<3, 3>(3, 0) = wP*LAM_PVA13;
        }

        // Jacobian on Pb
        idx = PbIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DPb(jacobians[idx]);
            Dr_DPb.setZero();
            Dr_DPb.block<3, 3>(3, 0) = wP*PSI_PVA11;
        }

        // Jacobian on Vb
        idx = VbIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DVb(jacobians[idx]);
            Dr_DVb.setZero();
            Dr_DVb.block<3, 3>(3, 0) = wP*PSI_PVA12;
        }

        // Jacobian on Ab
        idx = AbIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DAb(jacobians[idx]);
            Dr_DAb.setZero();
            Dr_DAb.block<3, 3>(3, 0) = wP*PSI_PVA13;
        }

        return true;
    }

private:

    SE3d pose_meas;

    double wR;
    double wP;

    // Gaussian process params

    const int RaIdx = 0;
    const int OaIdx = 1;
    const int PaIdx = 2;
    const int VaIdx = 3;
    const int AaIdx = 4;

    const int RbIdx = 5;
    const int ObIdx = 6;
    const int PbIdx = 7;
    const int VbIdx = 8;
    const int AbIdx = 9;

    double Dt;     // Knot length
    double s;      // Normalized time (t - t_i)/Dt
    GPMixer gpm;

    // Lambda
    Matrix<double, Dynamic, 1> lambda_R;
    // Lambda dot
    Matrix<double, Dynamic, 1> lambda_P;
};
