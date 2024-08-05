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

    GPMotionPriorTwoKnotsFactorAutodiff(GPMixerPtr gpm_)
    :   Dt          (gpm_->getDt()   ),
        gpm         (gpm_            )
    {
        // Calculate the information matrix
        Matrix<double, STATE_DIM, STATE_DIM> Info;
        Info.setZero();

        double Dtpow[7];
        for(int j = 0; j < 7; j++)
            Dtpow[j] = pow(Dt, j);

        Matrix3d Qtilde;
        Qtilde << 1.0/20.0*Dtpow[5], 1.0/8.0*Dtpow[4], 1.0/6.0*Dtpow[3],
                  1.0/08.0*Dtpow[4], 1.0/3.0*Dtpow[3], 1.0/2.0*Dtpow[2],
                  1.0/06.0*Dtpow[3], 1.0/2.0*Dtpow[2], 1.0/1.0*Dtpow[1];
        Info.block<9, 9>(0, 0) = gpm->kron(Qtilde, gpm->getSigGa());
        Info.block<9, 9>(9, 9) = gpm->kron(Qtilde, gpm->getSigNu());
        
        // Find the square root info
        // sqrtW = Matrix<double, STATE_DIM, STATE_DIM>::Identity(STATE_DIM, STATE_DIM);
        sqrtW = Eigen::LLT<Matrix<double, STATE_DIM, STATE_DIM>>(Info.inverse()).matrixL().transpose();
    }

    template <class T>
    bool operator()(T const *const *parameters, T *residuals) const
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        GPState<T> Xa(0);  gpm->MapParamToState<T>(parameters, RaIdx, Xa);
        GPState<T> Xb(Dt); gpm->MapParamToState<T>(parameters, RbIdx, Xb);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the residual ---------------------------------------------------------------------------*/

        SO3T Rab = Xa.R.inverse()*Xb.R;
        Vec3T Theb = Rab.log();

        Mat3T JrInvTheb = gpm->JrInv(Theb);
        Mat3T DJrInvThebOb_DTheb = gpm->DJrInvXV_DX(Theb, Xb.O);

        Vec3T Thedotb = JrInvTheb*Xb.O;
        Vec3T Theddotb = DJrInvThebOb_DTheb*Thedotb + JrInvTheb*Xb.S;

        double Dtsq = Dt*Dt;

        // Rotational residual
        Vec3T rRot = Theb - Dt*Xa.O - 0.5*Dtsq*Xa.S;

        // Rotational rate residual
        Vec3T rRdot = Thedotb - Xa.O - Dt*Xa.S;

        // Rotational acc residual
        Vec3T rRddot = Theddotb - Xa.S;

        // Positional residual
        Vec3T rPos = Xb.P - (Xa.P + Dt*Xa.V + 0.5*Dtsq*Xa.A);

        // Velocity residual
        Vec3T rVel = Xb.V - (Xa.V + Dt*Xa.A);

        // Acceleration residual
        Vec3T rAcc = Xb.A - Xa.A;

        // Residual
        Eigen::Map<Matrix<T, STATE_DIM, 1>> residual(residuals);
        residual << rRot, rRdot, rRddot, rPos, rVel, rAcc;
        residual = sqrtW*residual;

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        return true;
    }

private:

    const int Ridx = 0;
    const int Oidx = 1;
    const int Sidx = 2;
    const int Pidx = 3;
    const int Vidx = 4;
    const int Aidx = 5;

    const int RaIdx = 0;
    const int OaIdx = 1;
    const int SaIdx = 2;
    const int PaIdx = 3;
    const int VaIdx = 4;
    const int AaIdx = 5;

    const int RbIdx = 6;
    const int ObIdx = 7;
    const int SbIdx = 8;
    const int PbIdx = 9;
    const int VbIdx = 10;
    const int AbIdx = 11;

    double wR;
    double wP;

    // Square root information
    Matrix<double, STATE_DIM, STATE_DIM> sqrtW;
    
    // Knot length
    double Dt;
    
    // Mixer for gaussian process
    GPMixerPtr gpm;
};
