/**
 * This file is part of splio.
 *
 * Copyright (C) 2020 Thien-Minh Nguyen <thienminh.nguyen at ntu dot edu dot
 * sg>, School of EEE Nanyang Technological Univertsity, Singapore
 *
 * For more information please see <https://britsknguyen.github.io>.
 * or <https://github.com/brytsknguyen/splio>.
 * If you use this code, please cite the respective publications as
 * listed on the above websites.
 *
 * splio is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * splio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with splio.  If not, see <http://www.gnu.org/licenses/>.
 */

//
// Created by Thien-Minh Nguyen on 01/08/22.
//

#include <iostream> 
#include <ros/assert.h>
#include <ceres/ceres.h>

#include "../utility.h"
#include <basalt/spline/ceres_spline_helper.h>

class GPMotionPriorFactorAutodiff
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GPMotionPriorFactorAutodiff(double wR_, double wP_, double Dt_, double ss_, double sf_, double dtsf_)
    :   wR          (wR_             ),
        wP          (wP_             ),
        Dt          (Dt_             ),
        ss          (ss_             ),
        sf          (sf_             ),
        dtsf        (dtsf_           ),
        gpm         (Dt_             )
    {
        // Calculate the information matrix
        Matrix<double, 15, 15> Info;
        Info.setZero();

        double dtsfpow[7];
        for(int j = 0; j < 7; j++)
            dtsfpow[j] = pow(dtsf, j);

        Matrix2d QR;
        QR << 1.0/3.0*dtsfpow[3]*wR, 1.0/2.0*dtsfpow[2]*wR,
              1.0/2.0*dtsfpow[2]*wR,         dtsfpow[1]*wR;
        Info.block<6, 6>(0, 0) = kron(QR, Matrix3d::Identity());

        Matrix3d QP;
        QP << 1.0/20.0*dtsfpow[5]*wP, 1.0/8.0*dtsfpow[4]*wP, 1.0/6.0*dtsfpow[3]*wP,
              1.0/ 8.0*dtsfpow[4]*wP, 1.0/3.0*dtsfpow[3]*wP, 1.0/2.0*dtsfpow[2]*wP,
              1.0/ 6.0*dtsfpow[3]*wP, 1.0/2.0*dtsfpow[2]*wP, 1.0/1.0*dtsfpow[1]*wP;
        Info.block<9, 9>(6, 6) = kron(QP, Matrix3d::Identity());
        // sqrtW = Eigen::LLT<Matrix<double, 15, 15>>(Info).matrixL().transpose();
        sqrtW = Matrix<double, 15, 15>::Identity(15, 15);
    }

    template <class T>
    bool operator()(T const *const *parameters, T *residuals) const
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        // Map parameters to the control point states
        StateStamped<T> Xsa(0);  gpm.MapParamToState<T>(parameters, RsaIdx, Xsa);
        StateStamped<T> Xsb(Dt); gpm.MapParamToState<T>(parameters, RsbIdx, Xsb);
        StateStamped<T> Xfa(0);  gpm.MapParamToState<T>(parameters, RfaIdx, Xfa);
        StateStamped<T> Xfb(Dt); gpm.MapParamToState<T>(parameters, RfbIdx, Xfb);

        /* #region Calculate the residual ---------------------------------------------------------------------------*/

        StateStamped<T> Xs(ss*Dt); vector<vector<Mat3T>> DXs_DXsa; vector<vector<Mat3T>> DXs_DXsb;
        StateStamped<T> Xf(sf*Dt); vector<vector<Mat3T>> DXf_DXfa; vector<vector<Mat3T>> DXf_DXfb;

        Eigen::Matrix<T, 6, 1> gammasa;
        Eigen::Matrix<T, 6, 1> gammasb;
        Eigen::Matrix<T, 6, 1> gammas;
        Eigen::Matrix<T, 6, 1> gammafa;
        Eigen::Matrix<T, 6, 1> gammafb;
        Eigen::Matrix<T, 6, 1> gammaf;

        gpm.ComputeXtAndDerivs<T>(Xsa, Xsb, Xs, DXs_DXsa, DXs_DXsb, gammasa, gammasb, gammas);
        gpm.ComputeXtAndDerivs<T>(Xfa, Xfb, Xf, DXf_DXfa, DXf_DXfb, gammafa, gammafb, gammaf);

        // Relative rotation and its rate
        SO3T Rsf         = Xs.R.inverse()*Xf.R;
        Vec3T thetaf     = Rsf.log();
        Vec3T thetadotf  = gpm.JrInv(thetaf)*Xf.O;

        // Vec3T thetasb    = gammasb.block(0, 0, 3, 1);
        // Vec3T thetadotsb = gammasb.block(3, 0, 3, 1);
        // Vec3T thetas     = gammas.block(0, 0, 3, 1);
        // Vec3T thetadots  = gammas.block(3, 0, 3, 1);

        // Matrix<T, Dynamic, Dynamic> LAM_ROt = gpm.LAMBDA_RO(ss*Dt).cast<T>();
        // Matrix<T, Dynamic, Dynamic> PSI_ROt = gpm.PSI_RO(ss*Dt).cast<T>();
        // // Extract the blocks of SO3 states
        // Mat3T LAM_RO11 = LAM_ROt.block(0, 0, 3, 3);
        // Mat3T LAM_RO12 = LAM_ROt.block(0, 3, 3, 3);
        // Mat3T LAM_RO21 = LAM_ROt.block(3, 0, 3, 3);
        // Mat3T LAM_RO22 = LAM_ROt.block(3, 3, 3, 3);
        // Mat3T PSI_RO11 = PSI_ROt.block(0, 0, 3, 3);
        // Mat3T PSI_RO12 = PSI_ROt.block(0, 3, 3, 3);
        // Mat3T PSI_RO21 = PSI_ROt.block(3, 0, 3, 3);
        // Mat3T PSI_RO22 = PSI_ROt.block(3, 3, 3, 3);

        // Rotational residual
        Vec3T rRot = - dtsf*Xs.O;

        // Rotational rate residual
        Vec3T rRdot = thetadotf - Xs.O;

        // Positional residual
        Vec3T rPos = Xf.P - (Xs.P + dtsf*Xs.V + dtsf*dtsf/2*Xs.A);

        // Velocity residual
        Vec3T rVel = Xf.V - (Xs.V + dtsf*Xs.A);

        // Acceleration residual
        Vec3T rAcc = Xf.A - Xs.A;

        // Residual
        Eigen::Map<Matrix<T, 15, 1>> residual(residuals);
        residual << rRot, rRdot, rPos, rVel, rAcc;
        residual = sqrtW*residual;

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        return true;
    }

private:

    const int Ridx = 0;
    const int Oidx = 1;
    const int Pidx = 2;
    const int Vidx = 3;
    const int Aidx = 4;

    const int RsaIdx = 0;
    const int OsaIdx = 1;
    const int PsaIdx = 2;
    const int VsaIdx = 3;
    const int AsaIdx = 4;

    const int RsbIdx = 5;
    const int OsbIdx = 6;
    const int PsbIdx = 7;
    const int VsbIdx = 8;
    const int AsbIdx = 9;

    const int RfaIdx = 10;
    const int OfaIdx = 11;
    const int PfaIdx = 12;
    const int VfaIdx = 13;
    const int AfaIdx = 14;

    const int RfbIdx = 15;
    const int OfbIdx = 16;
    const int PfbIdx = 17;
    const int VfbIdx = 18;
    const int AfbIdx = 19;

    double wR;
    double wP;

    // Matrix<double, 15, 15> Info;
    Matrix<double, 15, 15> sqrtW;

    double Dt;     // Knot length
    double ss;     // Normalized time (t - ts)/Dt
    double sf;     // Normalized time (t - ts)/Dt
    double dtsf;   // Time difference ts - tf
    GPMixer gpm;
};