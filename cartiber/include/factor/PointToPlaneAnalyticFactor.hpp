/**
* This file is part of splio.
* 
* Copyright (C) 2020 Thien-Minh Nguyen <thienminh.nguyen at ntu dot edu dot sg>,
* School of EEE
* Nanyang Technological Univertsity, Singapore
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

#include <ceres/ceres.h>
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/utils/sophus_utils.hpp"
#include "../utility.h"

using namespace Eigen;

class PointToPlaneAnalyticFactor: public ceres::CostFunction
{
public:

    // Destructor
    ~PointToPlaneAnalyticFactor() {};

    // Constructor
    PointToPlaneAnalyticFactor(const Vector3d &finW_, const Vector3d &f_, const Vector4d &coef, double w_,
                               int N_, double Dt_, double s_)
    :   finW       (finW_            ),
        f          (f_               ),
        n          (coef.head<3>()   ),
        m          (coef.tail<1>()(0)),
        w          (w_               ),
        N          (N_               ),
        Dt         (Dt_              ),
        s          (s_               )
    {
        // 1-element residual: n^T*(Rt*f + pt) + m
        set_num_residuals(1);

        for (size_t j = 0; j < N; j++)
            mutable_parameter_block_sizes()->push_back(4);

        for (size_t j = 0; j < N; j++)
            mutable_parameter_block_sizes()->push_back(3);

        // Calculate the spline-defining quantities:

        // Blending matrix for position, in standard form
        Matrix<double, Dynamic, Dynamic> B = basalt::computeBlendingMatrix<double, false>(N);

        // Blending matrix for rotation, in cummulative form
        Matrix<double, Dynamic, Dynamic> Btilde = basalt::computeBlendingMatrix<double, true>(N);

        // // Inverse of knot length
        // double Dt_inv = 1.0/Dt;

        // Time powers
        Matrix<double, Dynamic, 1> U(N);
        for(int j = 0; j < N; j++)
            U(j) = std::pow(s, j);

        // Lambda for p
        lambda_P = B*U;

        // Lambda for R
        lambda_R = Btilde*U;
    }

    Matrix3d rightJacobian(const Vector3d &phi) const
    {
        Matrix3d Jr;
        Sophus::rightJacobianSO3(phi, Jr);
        return Jr;
    }

    Matrix3d rightJacobianInv(const Vector3d &phi) const
    {
        Matrix3d Jr_inv;
        Sophus::rightJacobianInvSO3(phi, Jr_inv);
        return Jr_inv;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Indexing offsets for the states
        size_t R_offset = 0;            // for quaternion
        size_t P_offset = R_offset + N; // for position
        size_t B_offset = P_offset + N; // for bias

        // Map parameters to the control point states
        SO3d Rot[N];
        Vector3d Pos[N];
        for (int j = 0; j < N; j++)
        {
            Rot[j] = Map<SO3d const>(parameters[R_offset + j]);
            Pos[j] = Map<Vector3d const>(parameters[P_offset + j]);

            // printf("SPLIO: p%d. lambda_a: %f\n", j, lambda_P_ddot(j));
            // std::cout << pos[j].transpose() << std::endl;
        }

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the pose at sampling time --------------------------------------------------------------*/

        // The following use the formulations in the paper 2020 CVPR:
        // "Efficient derivative computation for cumulative b-splines on lie groups."
        // Sommer, Christiane, Vladyslav Usenko, David Schubert, Nikolaus Demmel, and Daniel Cremers.
        // Some errors in the paper is corrected

        // Calculate the delta terms: delta(1) ... delta(N-1), where delta(j) = Log( R(j-1)^-1 * R(j) ),
        // delta(0) is an extension in the paper as the index starts at 1
        Vector3d delta[N];
        delta[0] = Rot[0].log();
        for (int j = 1; j < N; j++)
        {
            delta[j] = (Rot[j - 1].inverse() * Rot[j]).log();
            // if (isnan(delta[j].norm()))
            // {
            //     printf(KRED "delta[%d] is nan!" RESET, j);
            //     delta[j] = Vector3d(0, 0, 0);
            // }
        }

        // Calculate the A terms: A(1) ... A(N-1), where A(j) = Log( lambda(j) * d(j) ), A(0) is an extension
        SO3d A[N];
        A[0] = Rot[0];
        for (int j = 1; j < N; j++)
            A[j] = SO3d::exp(lambda_R(j) * delta[j]).matrix();

        // Calculate the P terms: P(0) ... P(N-1) = I, where P(j) = A(N-1)^-1 A(N-2)^-1 ... A(j+1)^-1
        SO3d P[N];
        P[N - 1] = SO3d(Quaternd(1, 0, 0, 0));
        for (int j = N - 1; j >= 1; j--)
            P[j - 1] = P[j] * A[j].inverse();

        // Predicted orientation from Rt^-1 = P(N-1)R(0)^-1
        SO3d R_W_Bt = (P[0] * Rot[0].inverse()).inverse();

        // Predicted position
        Vector3d p_inW_Bt(0, 0, 0);
        for (int j = 0; j < N; j++)
            p_inW_Bt += lambda_P[j]*Pos[j];

        /* #endregion Calculate the pose at sampling time -----------------------------------------------------------*/
    
        /* #region Recalculate the normal, mean and noise -----------------------------------------------------------*/

        // Find the feature in the world frame
        Vector3d f_inW = R_W_Bt * f + p_inW_Bt;

        /* #endregion Recalculate the normal, mean and noise --------------------------------------------------------*/

        /* #region Calculate the residual ---------------------------------------------------------------------------*/

        // Residual
        residuals[0] = w*(n.transpose() * (R_W_Bt * f + p_inW_Bt) + m);

        // if (residuals[0] > 1.0e9 || std::isnan(residuals[0]))
        // {
        //     ROS_WARN("numerical unstability in lidar factor\n");
        //     cout << "f: " << f << endl;
        //     cout << "n: " << n << endl;
        //     cout << "m: " << m << endl;
        //     cout << "s: " << s << endl;
        // }

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        if (!jacobians)
            return true;

        /* #region Jacobian of dis on knot_R ------------------------------------------------------------------------*/

        // The inverse right Jacobian Jr(dj) = d(deltaj)/d(Rj). Note that d( d[j+1] )/d( R[j] ) = -Jr(-d[j+1])
        Matrix3d ddelta_dR[N];
        for (int j = 0; j < N; j++)
            ddelta_dR[j] = rightJacobianInv(delta[j]);

        Matrix3d JrLambaDelta[N];
        for (int j = 0; j < N; j++)
            JrLambaDelta[j] = rightJacobian(lambda_R[j] * delta[j]);

        // Jacobian d(Rt)/d(Rj). Derived from the chain rule:
        // d(Rt)/d(Rj) = d(Rt(rho))/d(rho) [ d(rho)/d(dj) . d(dj)/d(Rj) + d(rho)/d(d[j+1]) d(d[j+1]))/d(Rj) ]
        // by using equation (57) in the TUM CVPR paper and some manipulation, we obtain
        // d(Rt)/d(R[j]) = lambda[j] * P[j] * Jr( lambda[j] delta[j] ) * Jr^-1(delta[j])
        //                 - lambda[j+1] * P[j+1] * Jr( lambda[j+1] delta[j+1] ) * Jr^-1( -delta[j+1] )
        Matrix3d dRt_dR[N];
        for (int j = 0; j < N; j++)
        {
            if (j == N - 1)
                dRt_dR[j] = lambda_R[j] * P[j].matrix() * JrLambaDelta[j] * ddelta_dR[j];
            else
                dRt_dR[j] = lambda_R[j] * P[j].matrix() * JrLambaDelta[j] * ddelta_dR[j] - lambda_R[j + 1] * P[j + 1].matrix() * JrLambaDelta[j + 1] * ddelta_dR[j + 1].transpose();
        }

        // Jacobian on Rt
        Matrix<double, 1, 3> ddis_dRt = -n.transpose()*R_W_Bt.matrix()*SO3d::hat(f);

        // Jacobian on Rj
        Matrix<double, 1, 3> ddis_dR[N];
        for(int j = 0; j < N; j++)
            ddis_dR[j] = ddis_dRt*dRt_dR[j];

        /* #endregion Jacobian of dis on knot_R ---------------------------------------------------------------------*/

        /* #region Jacobian of dis on knot P ------------------------------------------------------------------------*/
        
        Matrix<double, 1, 3> ddis_dP[N];
        for (int j = 0; j < N; j++)
            ddis_dP[j] = n.transpose()*Vector3d(lambda_P[j], lambda_P[j], lambda_P[j]).asDiagonal();

        /* #endregion Jacobian of dis on knot P ---------------------------------------------------------------------*/

        size_t idx;

        /// Rotation control point
        for (size_t j = 0; j < N; j++)
        {
            idx = R_offset + j;
            if (jacobians[idx])
            {
                Map<Matrix<double, 1, 4, RowMajor>> J_knot_R(jacobians[idx]);
                J_knot_R.setZero();

                // for gyro residual
                J_knot_R.block<1, 3>(0, 0) = w*ddis_dR[j];
            }
        }

        /// Position control point
        for (size_t j = 0; j < N; j++)
        {
            idx = P_offset + j;
            if (jacobians[idx])
            {
                Map<Matrix<double, 1, 3, RowMajor>> J_knot_p(jacobians[idx]);
                J_knot_p.setZero();

                /// for accel residual
                J_knot_p.block<1, 3>(0, 0) = w*ddis_dP[j];
            }
        }

        return true;
    }

private:

    // Feature coordinates in world frame
    Vector3d finW;

    // Feature coordinates in body frame
    Vector3d f;

    // Plane normal
    Vector3d n;

    // Plane offset
    double m;

    // Weight
    double w = 0.1;

    // Spline param
    int    N;
    double Dt;
    double s;

    // Lambda
    Matrix<double, Dynamic, 1> lambda_R;
    
    // Lambda dot
    Matrix<double, Dynamic, 1> lambda_P;
};