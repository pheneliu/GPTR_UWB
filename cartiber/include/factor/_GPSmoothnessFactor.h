#pragma once

#include <ceres/ceres.h>
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/utils/sophus_utils.hpp"
#include "GaussianProcess.hpp"
#include "utility.h"

using SO3d = Sophus::SO3<double>;
using SE3d = Sophus::SE3<double>;

class GPSmoothnessFactor : public ceres::CostFunction
{
public:

    GPSmoothnessFactor(double wR_, double wP_, double Dt_)
    :   wR          (wR_             ),
        wP          (wP_             ),
        Dt          (Dt_             ),
        gpm         (Dt_             )
    {
        // 6-element residual: (3x1 rotation, 3x1 position)
        set_num_residuals(6); // Angular diff, angular vel, pos diff, vel diff, acc diff

        // Factor involves three control points
        for(int j = 0; j < 3; j++)
        {
            mutable_parameter_block_sizes()->push_back(4);
            mutable_parameter_block_sizes()->push_back(3);
        }
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        SO3d Ra = Eigen::Map<SO3d const>(parameters[RaIdx]);
        Vec3 Pa = Eigen::Map<Vec3 const>(parameters[PaIdx]);
        SO3d Rb = Eigen::Map<SO3d const>(parameters[RbIdx]);
        Vec3 Pb = Eigen::Map<Vec3 const>(parameters[PbIdx]);
        SO3d Rc = Eigen::Map<SO3d const>(parameters[RcIdx]);
        Vec3 Pc = Eigen::Map<Vec3 const>(parameters[PcIdx]);
 
        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the residual ---------------------------------------------------------------------------*/

        SO3d Rainv = Ra.inverse();
        SO3d Rbinv = Rb.inverse(); 
        SO3d Rab = Rainv*Rb;
        SO3d Rbc = Rbinv*Rc;

        Vec3 phiab = Rab.log();
        Vec3 phibc = Rbc.log();

        Vec3 Pab = Pb - Pa;
        Vec3 Pbc = Pc - Pb;

        // Rotational residual
        Vec3 rRot = phiab - phibc;

        // Positional residual
        Vec3 rPos = Rainv*Pab - Rbinv*Pbc;

        // Residual
        Eigen::Map<Matrix<double, 6, 1>> residual(residuals);
        residual << wR*rRot, wP*rPos;

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        if (!jacobians)
            return true;
        
        size_t idx;

        // Jacobians on Ra, Rb, Rc
        {

            // dr_dRa
            idx = RaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> Dr_DRa(jacobians[idx]);
                Dr_DRa.setZero();
                Dr_DRa.block<3, 3>(0, 0) = -wR*gpm.JrInv(phiab)*Rab.inverse().matrix();
                Dr_DRa.block<3, 3>(3, 0) =  wP*SO3d::hat(Rainv*Pab);
            }

            // dr_dRb
            idx = RbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> Dr_DRb(jacobians[idx]);
                Dr_DRb.setZero();
                Dr_DRb.block<3, 3>(0, 0) =  wR*gpm.JrInv(phiab) + gpm.JrInv(phibc)*Rbc.inverse().matrix();
                Dr_DRb.block<3, 3>(3, 0) = -wP*SO3d::hat(Rbinv*Pbc);
            }

            // dr_dRc
            idx = RcIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> Dr_DRc(jacobians[idx]);
                Dr_DRc.setZero();
                Dr_DRc.block<3, 3>(0, 0) = -wR*gpm.JrInv(phibc);
            }

        }
        
        // Jacobian on Pa, Pb, Pc
        {

            // dr_dPa
            idx = PaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DPa(jacobians[idx]);
                Dr_DPa.setZero();
                Dr_DPa.block<3, 3>(3, 0) = -wP*Rainv.matrix();
            }

            // dr_dPb
            idx = PbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DPb(jacobians[idx]);
                Dr_DPb.setZero();
                Dr_DPb.block<3, 3>(3, 0) = wP*(Rainv.matrix() + Rbinv.matrix());
            }

            // dr_dPc
            idx = PcIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Dr_DPc(jacobians[idx]);
                Dr_DPc.setZero();
                Dr_DPc.block<3, 3>(3, 0) = -wP*Rbinv.matrix();
            }

        }

        return true;
    }

private:

    const int RaIdx = 0;
    const int PaIdx = 1;
    const int RbIdx = 2;
    const int PbIdx = 3;
    const int RcIdx = 4;
    const int PcIdx = 5;

    double wR;
    double wP;
    double Dt;     // Knot length
    GPMixer gpm;
};
