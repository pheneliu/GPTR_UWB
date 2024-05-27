/**
* This file is part of SLICT.
* 
* Copyright (C) 2020 Thien-Minh Nguyen <thienminh.nguyen at ntu dot edu dot sg>,
* School of EEE
* Nanyang Technological Univertsity, Singapore
* 
* For more information please see <https://britsknguyen.github.io>.
* or <https://github.com/brytsknguyen/slict>.
* If you use this code, please cite the respective publications as
* listed on the above websites.
* 
* SLICT is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* SLICT is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with SLICT.  If not, see <http://www.gnu.org/licenses/>.
*/

//
// Created by Thien-Minh Nguyen on 01/08/22.
//

#include <ceres/ceres.h>
#include <Eigen/Eigen>

#include "../utility.h"

using namespace Eigen;

class PoseFactorCT : public ceres::SizedCostFunction<6, 7, 7>
{
public:
    PoseFactorCT() = delete;
    PoseFactorCT(const Vector3d &pbar_, const Quaternd &qbar_,
                double s_, double dt_, double wp_ = 1.0, double wq_ = 1.0)
    : pbar(pbar_), qbar(qbar_), s(s_), dt(dt_), wp(wp_), wq(wq_)
    {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Quaternd Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Quaternd Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

        double sf = s;
        double sb = 1 - sf;

        Eigen::AngleAxis<double> Phif(Qi.inverse() * Qj);
        Eigen::AngleAxis<double> Phib(-Phif.angle(), Phif.axis());
        
        Eigen::AngleAxis<double> sfPhif(sf*Phif.angle(), Phif.axis());
        Eigen::AngleAxis<double> sbPhib(sb*Phib.angle(), Phib.axis());
        
        Eigen::AngleAxis<double> Phir(qbar.inverse() * Qi * sfPhif);

        Quaternd Qs(sfPhif);
        // Matrix3d Rs = Qs.toRotationMatrix();
        Vector3d Ps = sb*Pi + sf*Pj;

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);

        residual.block<3, 1>(0, 0) = wp*(Ps - pbar);
        residual.block<3, 1>(3, 0) = wq*Phir.axis()*Phir.angle();
        
        if (jacobians)
        {
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                jacobian_pose_i.block<3, 3>(0, 0) = wp*Vector3d(sb, sb, sb).asDiagonal();

                Eigen::Matrix<double, 3, 3> JrPhirInv = Util::SO3JrightInv(Phir);
                Eigen::Matrix<double, 3, 3> JrsbPhib  = Util::SO3Jright(sbPhib);
                Eigen::Matrix<double, 3, 3> JrPhibInv = Util::SO3JrightInv(Phib);

                jacobian_pose_i.block<3, 3>(3, 0) = wq * sb * JrPhirInv * JrsbPhib * JrPhibInv;
            }

            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
                jacobian_pose_j.setZero();

                jacobian_pose_j.block<3, 3>(0, 0) = wp*Vector3d(sf, sf, sf).asDiagonal();

                Eigen::Matrix<double, 3, 3> JrPhirInv = Util::SO3JrightInv(Phir);
                Eigen::Matrix<double, 3, 3> JrsfPhif  = Util::SO3Jright(sfPhif);
                Eigen::Matrix<double, 3, 3> JrPhifInv = Util::SO3JrightInv(Phif);

                jacobian_pose_j.block<3, 3>(3, 0) = wq * sf * JrPhirInv * JrsfPhif * JrPhifInv;
            }
        }

        return true;
    }
    // void Check(double **parameters);

private:
    Vector3d pbar;
    Quaternd qbar;

    double s;
    double dt;
    double wp;
    double wq;
};