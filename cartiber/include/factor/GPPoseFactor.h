#pragma once

#include <ceres/ceres.h>
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/utils/sophus_utils.hpp"
#include "utility.h"

#include "GaussianProcess.hpp"

class GPPoseFactor : public ceres::CostFunction
{
public:

    GPPoseFactor(const SE3d &pose_meas_, double wR_, double wP_, double Dt_, double s_)
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

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        StateStamped Xa(0);  gpm.MapParamToState(parameters, RaIdx, Xa);
        StateStamped Xb(Dt); gpm.MapParamToState(parameters, RbIdx, Xb);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the residual ---------------------------------------------------------------------------*/
        
        StateStamped Xt(s*Dt); vector<vector<Matrix3d>> DXt_DXa; vector<vector<Matrix3d>> DXt_DXb;

        Eigen::Matrix<double, 6, 1> gammaa;
        Eigen::Matrix<double, 6, 1> gammab;
        Eigen::Matrix<double, 6, 1> gammat;

        gpm.ComputeXtAndDerivs(Xa, Xb, Xt, DXt_DXa, DXt_DXb, gammaa, gammab, gammat);

        // Rotational residual
        Vector3d rR = (pose_meas.so3().inverse()*Xt.R).log();

        // Positional residual
        Vector3d rP = (Xt.P - pose_meas.translation());

        // Residual
        Eigen::Map<Matrix<double, 6, 1>> residual(residuals);
        residual.block<3, 1>(0, 0) = wR*rR;
        residual.block<3, 1>(3, 0) = wP*rP;

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        if (!jacobians)
            return true;

        Matrix3d DrR_DRt = gpm.JrInv(rR);
        // Matrix3d DrP_DPt = Matrix3d::Identity(3,3);

        size_t idx;

        // DrR_Ra
        idx = RaIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> Dr_DRa(jacobians[idx]);
            Dr_DRa.setZero();
            Dr_DRa.block<3, 3>(0, 0) = wR*DrR_DRt*DXt_DXa[Ridx][Ridx];
        }
        
        // DrR_Oa
        idx = OaIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DOa(jacobians[idx]);
            Dr_DOa.setZero();
            Dr_DOa.block<3, 3>(0, 0) = wR*DrR_DRt*DXt_DXa[Ridx][Oidx];
        }

        // DrP_DPa
        idx = PaIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DPa(jacobians[idx]);
            Dr_DPa.setZero();
            Dr_DPa.block<3, 3>(3, 0) = wP*DXt_DXa[Pidx][Pidx];
        }

        // DrP_DVa
        idx = VaIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DVa(jacobians[idx]);
            Dr_DVa.setZero();
            Dr_DVa.block<3, 3>(3, 0) = wP*DXt_DXa[Pidx][Vidx];
        }

        // DrP_DAa
        idx = AaIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DAa(jacobians[idx]);
            Dr_DAa.setZero();
            Dr_DAa.block<3, 3>(3, 0) = wP*DXt_DXa[Pidx][Aidx];
        }

        // DrR_Rb
        idx = RbIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> Dr_DRb(jacobians[idx]);
            Dr_DRb.setZero();
            Dr_DRb.block<3, 3>(0, 0) = wR*DrR_DRt*DXt_DXb[Ridx][Ridx];
        }

        // DrR_Ob
        idx = ObIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DOb(jacobians[idx]);
            Dr_DOb.setZero();
            Dr_DOb.block<3, 3>(0, 0) = wR*DrR_DRt*DXt_DXb[Ridx][Oidx];
        }

        // DrP_DPb
        idx = PbIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DPb(jacobians[idx]);
            Dr_DPb.setZero();
            Dr_DPb.block<3, 3>(3, 0) = wP*DXt_DXb[Pidx][Pidx];
        }

        // DrP_DVb
        idx = VbIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DVb(jacobians[idx]);
            Dr_DVb.setZero();
            Dr_DVb.block<3, 3>(3, 0) = wP*DXt_DXb[Pidx][Vidx];
        }

        // DrP_DAb
        idx = AbIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DAb(jacobians[idx]);
            Dr_DAb.setZero();
            Dr_DAb.block<3, 3>(3, 0) = wP*DXt_DXb[Pidx][Aidx];
        }

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
