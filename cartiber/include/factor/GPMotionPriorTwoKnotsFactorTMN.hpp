#pragma once

#include <ceres/ceres.h>
// #include "basalt/spline/ceres_spline_helper.h"
// #include "basalt/utils/sophus_utils.hpp"
#include "../utility.h"
#include "GaussianProcess.hpp"

using SO3d = Sophus::SO3<double>;
using SE3d = Sophus::SE3<double>;


class GPMotionPriorTwoKnotsFactorTMN
{
typedef Eigen::Matrix<double, STATE_DIM, 2*STATE_DIM> MatJ;
public:

    GPMotionPriorTwoKnotsFactorTMN(double wR_, double wP_, double Dt_)
    :   wR          (wR_             ),
        wP          (wP_             ),
        Dt          (Dt_             ),
        dtsf        (Dt_             ),
        gpm         (Dt_             )
    {

        // // 6-element residual: (3x1 rotation, 3x1 position)
        // set_num_residuals(15); // Angular diff, angular vel, pos diff, vel diff, acc diff

        // for(int j = 0; j < 2; j++)
        // {
        //     mutable_parameter_block_sizes()->push_back(4);
        //     mutable_parameter_block_sizes()->push_back(3);
        //     mutable_parameter_block_sizes()->push_back(3);
        //     mutable_parameter_block_sizes()->push_back(3);
        //     mutable_parameter_block_sizes()->push_back(3);
        // }

        // Calculate the information matrix
        Matrix<double, STATE_DIM, STATE_DIM> Info;
        Info.setZero();

        double Dtpow[7];
        for(int j = 0; j < 7; j++)
            Dtpow[j] = pow(Dt, j);

        Matrix2d QR;
        QR << 1.0/20.0*Dtpow[5]*wP, 1.0/8.0*Dtpow[4]*wP, 1.0/6.0*Dtpow[3]*wP,
              1.0/08.0*Dtpow[4]*wP, 1.0/3.0*Dtpow[3]*wP, 1.0/2.0*Dtpow[2]*wP,
              1.0/06.0*Dtpow[3]*wP, 1.0/2.0*Dtpow[2]*wP, 1.0/1.0*Dtpow[1]*wP;
        Info.block<9, 9>(0, 0) = kron(QR, Matrix3d::Identity());

        Matrix3d QP;
        QP << 1.0/20.0*Dtpow[5]*wP, 1.0/8.0*Dtpow[4]*wP, 1.0/6.0*Dtpow[3]*wP,
              1.0/08.0*Dtpow[4]*wP, 1.0/3.0*Dtpow[3]*wP, 1.0/2.0*Dtpow[2]*wP,
              1.0/06.0*Dtpow[3]*wP, 1.0/2.0*Dtpow[2]*wP, 1.0/1.0*Dtpow[1]*wP;
        Info.block<9, 9>(6, 6) = kron(QP, Matrix3d::Identity());
        
        // Find the square root info
        // sqrtW = Matrix<double, STATE_DIM, STATE_DIM>::Identity(STATE_DIM, STATE_DIM);
        sqrtW = Eigen::LLT<Matrix<double, STATE_DIM, STATE_DIM>>(Info.inverse()).matrixL().transpose();

        residual.setZero();
        jacobian.setZero();
    }

    virtual bool Evaluate(const StateStamped<double> &Xa, const StateStamped<double> &Xb, bool computeJacobian=true)
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        // StateStamped Xa(0);  gpm.MapParamToState(parameters, RaIdx, Xa);
        // StateStamped Xb(Dt); gpm.MapParamToState(parameters, RbIdx, Xb);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the residual ---------------------------------------------------------------------------*/

        SO3d Rab = Xa.R.inverse()*Xb.R;
        Vec3 thetab = Rab.log();
        Mat3 JrInvthetab = gpm.JrInv(thetab);
        Vec3 thetadotb = JrInvthetab*Xb.O;
        Vec3 thetaddotb = JrInvthetab*Xb.O;

        // Rotational residual
        Vec3 rRot = thetab - Dt*Xa.O;

        // Rotational rate residual
        Vec3 rRdot = thetadotb - Xa.O;

        // Positional residual
        Vec3 rPos = Xb.P - (Xa.P + Dt*Xa.V + Dt*Dt/2*Xa.A);

        // Velocity residual
        Vec3 rVel = Xb.V - (Xa.V + Dt*Xa.A);

        // Acceleration residual
        Vec3 rAcc = Xb.A - Xa.A;

        // Residual
        residual << rRot, rRdot, rPos, rVel, rAcc;
        residual = sqrtW*residual;

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        if (!computeJacobian)
            return true;
        
        size_t idx;

        // Reusable Jacobians
        Mat3 Dthetab_DRa = -JrInvthetab*Rab.inverse().matrix();
        Mat3 Dthetab_DRb =  JrInvthetab;

        Mat3 Dthetadotb_Dthetab = gpm.DJrInvXV_DX(thetab, Xb.O);
        Mat3 Dthetadotb_DRa = Dthetadotb_Dthetab*Dthetab_DRa;
        Mat3 Dthetadotb_DRb = Dthetadotb_Dthetab*Dthetab_DRb;

        Mat3 DtI = Vec3(Dt, Dt, Dt).asDiagonal();
        Mat3 Eye = Mat3::Identity();

        // Work out the Jacobians for SO3 states first
        {
            // dr_dRa
            idx = RaIdx;
            Eigen::Block<MatJ, 15, 3> Dr_DRa(jacobian.block<15, 3>(0, idx));
            Dr_DRa.setZero();
            Dr_DRa.block<3, 3>(0, 0) = Dthetab_DRa;
            Dr_DRa.block<3, 3>(3, 0) = Dthetadotb_DRa;
            Dr_DRa = sqrtW*Dr_DRa;

            // dr_dOa
            idx = OaIdx;
            Eigen::Block<MatJ, 15, 3> Dr_DOa(jacobian.block<15, 3>(0, idx));
            Dr_DOa.setZero();
            Dr_DOa.block<3, 3>(0, 0) = -DtI;
            Dr_DOa.block<3, 3>(3, 0) = -Eye;
            Dr_DOa = sqrtW*Dr_DOa;

            // dr_dRb
            idx = RbIdx;
            Eigen::Block<MatJ, 15, 3> Dr_DRb(jacobian.block<15, 3>(0, idx));
            Dr_DRb.setZero();
            Dr_DRb.block<3, 3>(0, 0) = Dthetab_DRb;
            Dr_DRb.block<3, 3>(3, 0) = Dthetadotb_DRb;
            Dr_DRb = sqrtW*Dr_DRb;

            // dr_dOb
            idx = ObIdx;
            Eigen::Block<MatJ, 15, 3> Dr_DOb(jacobian.block<15, 3>(0, idx));
            Dr_DOb.setZero();
            Dr_DOb.block<3, 3>(3, 0) = JrInvthetab;
            Dr_DOb = sqrtW*Dr_DOb;
        }

        // Jacobians on PVAa states
        {
            idx = PaIdx;
            Eigen::Block<MatJ, 15, 3> Dr_DPa(jacobian.block<15, 3>(0, idx));
            Dr_DPa.setZero();
            Dr_DPa.block<3, 3>(6,  0) = -Eye;
            Dr_DPa = sqrtW*Dr_DPa;

            idx = VaIdx;
            Eigen::Block<MatJ, 15, 3> Dr_DVa(jacobian.block<15, 3>(0, idx));
            Dr_DVa.setZero();
            Dr_DVa.block<3, 3>(6,  0) = -DtI;
            Dr_DVa.block<3, 3>(9,  0) = -Eye;
            Dr_DVa = sqrtW*Dr_DVa;

            idx = AaIdx;
            Eigen::Block<MatJ, 15, 3> Dr_DAa(jacobian.block<15, 3>(0, idx));
            Dr_DAa.setZero();
            Dr_DAa.block<3, 3>(6,  0) = -0.5*Dt*DtI;
            Dr_DAa.block<3, 3>(9,  0) = -DtI;
            Dr_DAa.block<3, 3>(12, 0) = -Eye;
            Dr_DAa = sqrtW*Dr_DAa;
        }

        // Jacobians on PVAb states
        {
            idx = PbIdx;
            Eigen::Block<MatJ, 15, 3> Dr_DPb(jacobian.block<15, 3>(0, idx));
            Dr_DPb.setZero();
            Dr_DPb.block<3, 3>(6,  0) = Eye;
            Dr_DPb = sqrtW*Dr_DPb;

            idx = VbIdx;
            Eigen::Block<MatJ, 15, 3> Dr_DVb(jacobian.block<15, 3>(0, idx));
            Dr_DVb.setZero();
            Dr_DVb.block<3, 3>(9,  0) = Eye;
            Dr_DVb = sqrtW*Dr_DVb;

            idx = AbIdx;
            Eigen::Block<MatJ, 15, 3> Dr_DAb(jacobian.block<15, 3>(0, idx));
            Dr_DAb.setZero();
            Dr_DAb.block<3, 3>(12, 0) = Eye;
            Dr_DAb = sqrtW*Dr_DAb;
        }

        return true;
    }

    // Residual and Jacobian
    Matrix<double, STATE_DIM, 1> residual;
    Matrix<double, STATE_DIM, 2*STATE_DIM> jacobian;

private:

    // const int Ridx = 0;
    // const int Oidx = 1;
    // const int Pidx = 2;
    // const int Vidx = 3;
    // const int Aidx = 4;

    const int RaIdx = 0;
    const int OaIdx = 3;
    const int PaIdx = 6;
    const int VaIdx = 9;
    const int AaIdx = 12;

    const int RbIdx = 15;
    const int ObIdx = 18;
    const int PbIdx = 21;
    const int VbIdx = 24;
    const int AbIdx = 27;

    double wR;
    double wP;

    // Matrix<double, 15, 15> Info;
    Matrix<double, 15, 15> sqrtW;

    double Dt;     // Knot length
    // double ss;     // Normalized time (t - ts)/Dt
    // double sf;     // Normalized time (t - ts)/Dt
    double dtsf;   // Time difference ts - tf
    GPMixer gpm;
};