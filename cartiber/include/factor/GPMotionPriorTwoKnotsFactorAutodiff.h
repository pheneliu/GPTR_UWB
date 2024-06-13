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

class GPMotionPriorTwoKnotsFactorAutodiff
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GPMotionPriorTwoKnotsFactorAutodiff(double wR_, double wP_, double Dt_)
    :   wR          (wR_             ),
        wP          (wP_             ),
        Dt          (Dt_             ),
        dtsf        (Dt_             ),
        gpm         (Dt_             )
    {
        // Calculate the information matrix
        Matrix<double, 15, 15> Info;
        Info.setZero();

        double Dtpow[7];
        for(int j = 0; j < 7; j++)
            Dtpow[j] = pow(Dt, j);

        Matrix2d QR;
        QR << 1.0/3.0*Dtpow[3]*wR, 1.0/2.0*Dtpow[2]*wR,
              1.0/2.0*Dtpow[2]*wR,         Dtpow[1]*wR;
        Info.block<6, 6>(0, 0) = kron(QR, Matrix3d::Identity());

        Matrix3d QP;
        QP << 1.0/20.0*Dtpow[5]*wP, 1.0/8.0*Dtpow[4]*wP, 1.0/6.0*Dtpow[3]*wP,
              1.0/08.0*Dtpow[4]*wP, 1.0/3.0*Dtpow[3]*wP, 1.0/2.0*Dtpow[2]*wP,
              1.0/06.0*Dtpow[3]*wP, 1.0/2.0*Dtpow[2]*wP, 1.0/1.0*Dtpow[1]*wP;
        Info.block<9, 9>(6, 6) = kron(QP, Matrix3d::Identity());
        
        // Find the square root info
        // sqrtW = Matrix<double, 15, 15>::Identity(15, 15);
        sqrtW = Eigen::LLT<Matrix<double, 15, 15>>(Info.inverse()).matrixL().transpose();
    }

    template <class T>
    bool operator()(T const *const *parameters, T *residuals) const
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        StateStamped<T> Xa(0);  gpm.MapParamToState<T>(parameters, RaIdx, Xa);
        StateStamped<T> Xb(Dt); gpm.MapParamToState<T>(parameters, RbIdx, Xb);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the residual ---------------------------------------------------------------------------*/

        SO3T  Rab = Xa.R.inverse()*Xb.R;
        Vec3T thetab = Rab.log();
        Mat3T JrInvthetab = gpm.JrInv(thetab);
        Vec3T thetadotb = JrInvthetab*Xb.O;
        
        // Rotational residual
        Vec3T rRot = thetab - Dt*Xa.O;

        // Rotational rate residual
        Vec3T rRdot = thetadotb - Xa.O;

        // Positional residual
        Vec3T rPos = Xb.P - (Xa.P + Dt*Xa.V + Dt*Dt/2*Xa.A);

        // Velocity residual
        Vec3T rVel = Xb.V - (Xa.V + Dt*Xa.A);

        // Acceleration residual
        Vec3T rAcc = Xb.A - Xa.A;

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
