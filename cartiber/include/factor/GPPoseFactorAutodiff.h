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

class GPPoseFactorAutodiff
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GPPoseFactorAutodiff(const SE3d &pose_meas_, double wR_, double wP_, double Dt_, double s_)
    :   pose_meas   (pose_meas_      ),
        wR          (wR_             ),
        wP          (wP_             ),
        Dt          (Dt_             ),
        s           (s_              ),
        gpm         (Dt_             )
    {}

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

        StateStamped<T> Xt(s*Dt); vector<vector<Mat3T>> DXt_DXa; vector<vector<Mat3T>> DXt_DXb;

        Eigen::Matrix<T, 6, 1> gammaa;
        Eigen::Matrix<T, 6, 1> gammab;
        Eigen::Matrix<T, 6, 1> gammat;

        gpm.ComputeXtAndDerivs<T>(Xa, Xb, Xt, DXt_DXa, DXt_DXb, gammaa, gammab, gammat);

        // Rotational residual
        Vec3T rR = (pose_meas.so3().inverse()*Xt.R).log();

        // Positional residual
        Vec3T rP = (Xt.P - pose_meas.translation());

        // Residual
        Eigen::Map<Matrix<T, 6, 1>> residual(residuals);
        residual.block(0, 0, 3, 1) = wR*rR;
        residual.block(3, 0, 3, 1) = wP*rP;

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        return true;
    }

private:

    SE3d pose_meas;

    double wR;
    double wP;

    // Gaussian process params

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

    double Dt;     // Knot length
    double s;      // Normalized time (t - t_i)/Dt
    GPMixer gpm;
};