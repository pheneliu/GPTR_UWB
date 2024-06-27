#pragma once

#include <ceres/ceres.h>
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/utils/sophus_utils.hpp"
#include "utility.h"

using SO3d = Sophus::SO3<double>;
using SE3d = Sophus::SE3<double>;

class ExtrinsicPoseFactor : public ceres::CostFunction
{
public:

    ExtrinsicPoseFactor(double wR_, double wP_, int N_, double Dt_, double sa_, double sb_)
    :   wR          (wR_             ),
        wP          (wP_             ),
        N           (N_              ),
        Dt          (Dt_             ),
        sa          (sa_             ),
        sb          (sb_             )
    {
        // 6-element residual: (3x1 rotation, 3x1 position)
        set_num_residuals(6);

        for (size_t j = 0; j < N; j++)
            mutable_parameter_block_sizes()->push_back(4);

        for (size_t j = 0; j < N; j++)
            mutable_parameter_block_sizes()->push_back(3);

        for (size_t j = 0; j < N; j++)
            mutable_parameter_block_sizes()->push_back(4);

        for (size_t j = 0; j < N; j++)
            mutable_parameter_block_sizes()->push_back(3);

        mutable_parameter_block_sizes()->push_back(4);
        mutable_parameter_block_sizes()->push_back(3);    

        // Calculate the spline-defining quantities:

        // Blending matrix for position, in standard form
        Matrix<double, Dynamic, Dynamic> B = basalt::computeBlendingMatrix<double, false>(N);

        // Blending matrix for rotation, in cummulative form
        Matrix<double, Dynamic, Dynamic> Btilde = basalt::computeBlendingMatrix<double, true>(N);

        // Inverse of knot length
        // double Dt_inv = 1.0/Dt;

        // Time powers
        Matrix<double, Dynamic, 1> Ua(N);
        for(int j = 0; j < N; j++)
            Ua(j) = std::pow(sa, j);

        Matrix<double, Dynamic, 1> Ub(N);
        for(int j = 0; j < N; j++)
            Ub(j) = std::pow(sb, j);

        // Lambda for p
        lambda_Pa = B*Ua;

        // Lambda for R
        lambda_Ra = Btilde*Ua;

        // Lambda for p
        lambda_Pb = B*Ub;

        // Lambda for R
        lambda_Rb = Btilde*Ub;
    }

    Matrix3d SO3Jr(const Vector3d &phi) const
    {
        Matrix3d Jr;
        Sophus::rightJacobianSO3(phi, Jr);
        return Jr;
    }

    Matrix3d SO3JrInv(const Vector3d &phi) const
    {
        Matrix3d Jr_inv;
        Sophus::rightJacobianInvSO3(phi, Jr_inv);
        return Jr_inv;
    }

    void ComputeRtAndDerivs(SO3d Rot[], const Matrix<double, Dynamic, 1> &lambda_R, SO3d &Rt, Matrix3d dRt_dR[]) const
    {
        // The following use the formulations in the paper 2020 CVPR:
        // "Efficient derivative computation for cumulative b-splines on lie groups."
        // Sommer, Christiane, Vladyslav Usenko, David Schubert, Nikolaus Demmel, and Daniel Cremers.
        // Some errors in the paper is corrected

        // Calculate the delta terms: delta(1) ... delta(N-1), where delta(j) = Log( R(j-1)^-1 * R(j) ),
        // delta(0) is an extension in the paper as the index starts at 1
        Vector3d delta[N];
        delta[0] = Rot[0].log();
        for (int j = 1; j < N; j++)
            delta[j] = (Rot[j - 1].inverse() * Rot[j]).log();

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
        Rt = (P[0] * Rot[0].inverse()).inverse();

        // The inverse right Jacobian Jr(dj) = d(deltaj)/d(Rj). Note that d( d[j+1] )/d( R[j] ) = -Jr(-d[j+1])
        Matrix3d ddelta_dR[N];
        for (int j = 0; j < N; j++)
            ddelta_dR[j] = SO3JrInv(delta[j]);

        Matrix3d JrLambaDelta[N];
        for (int j = 0; j < N; j++)
            JrLambaDelta[j] = SO3Jr(lambda_R[j] * delta[j]);

        // Jacobian d(Rt)/d(Rj). Derived from the chain rule:
        // d(Rt)/d(Rj) = d(Rt(rho))/d(rho) [ d(rho)/d(dj) . d(dj)/d(Rj) + d(rho)/d(d[j+1]) d(d[j+1]))/d(Rj) ]
        // by using equation (57) in the TUM CVPR paper and some manipulation, we obtain
        // d(Rt)/d(R[j]) = lambda[j] * P[j] * Jr( lambda[j] delta[j] ) * Jr^-1(delta[j])
        //                 - lambda[j+1] * P[j+1] * Jr( lambda[j+1] delta[j+1] ) * Jr^-1( -delta[j+1] )
        for (int j = 0; j < N; j++)
        {
            if (j == N - 1)
                dRt_dR[j] = lambda_R[j] * P[j].matrix() * JrLambaDelta[j] * ddelta_dR[j];
            else
                dRt_dR[j] = lambda_R[j] * P[j].matrix() * JrLambaDelta[j] * ddelta_dR[j]
                            - lambda_R[j + 1] * P[j + 1].matrix() * JrLambaDelta[j + 1] * ddelta_dR[j + 1].transpose();
        }
    }

    void ComputePtAndDerivs(Vector3d Pos[], const Matrix<double, Dynamic, 1> &lambda_P, Vector3d &Pt, Matrix3d dPt_P[]) const
    {
        // Predicted position
        Pt = Vector3d(0, 0, 0);
        for (int j = 0; j < N; j++)
        {
            Pt += lambda_P[j]*Pos[j];
            dPt_P[j] = Vector3d(lambda_P[j], lambda_P[j], lambda_P[j]).asDiagonal();
        }
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Indexing offsets for the states
        size_t Ra_offset = 0;             // for quaternion
        size_t Pa_offset = Ra_offset + N; // for position
        size_t Rb_offset = Pa_offset;     // for quaternion
        size_t Pb_offset = Rb_offset + N; // for position
        size_t Rx_offset = Pb_offset;     // for quaternion
        size_t Px_offset = Rx_offset + 1; // for position

        // Map parameters to the control point states
        SO3d Rota[N]; Vector3d Posa[N];
        SO3d Rotb[N]; Vector3d Posb[N];
        for (int j = 0; j < N; j++)
        {
            Rota[j] = Eigen::Map<SO3d const>(parameters[Ra_offset + j]);
            Posa[j] = Eigen::Map<Vector3d const>(parameters[Pa_offset + j]);
            Rotb[j] = Eigen::Map<SO3d const>(parameters[Rb_offset + j]);
            Posb[j] = Eigen::Map<Vector3d const>(parameters[Pb_offset + j]);
        }
        SO3d Rotx; Vector3d Posx;
        Rotx = Eigen::Map<SO3d const>(parameters[Rx_offset]);
        Posx = Eigen::Map<Vector3d const>(parameters[Px_offset]);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the residual ---------------------------------------------------------------------------*/

        SO3d Rat; Vector3d Pat; Matrix3d dRat_dRj[N]; Matrix3d dPat_dPj[N];
        SO3d Rbt; Vector3d Pbt; Matrix3d dRbt_dRj[N]; Matrix3d dPbt_dPj[N];

        ComputeRtAndDerivs(Rota, lambda_Ra, Rat, dRat_dRj);
        ComputeRtAndDerivs(Rotb, lambda_Rb, Rbt, dRbt_dRj);

        ComputePtAndDerivs(Posa, lambda_Pa, Pat, dPat_dPj);
        ComputePtAndDerivs(Posb, lambda_Pb, Pbt, dPbt_dPj);

        // Rotational residual
        Vector3d phi = (Rbt.inverse()*Rat*Rotx).log();

        // Positional residual
        Vector3d del = (Pat + Rat*Posx - Pbt);

        // Residual
        Eigen::Map<Matrix<double, 6, 1>> residual(residuals);
        residual.block<3, 1>(0, 0) = wR*phi;
        residual.block<3, 1>(3, 0) = wP*del;

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        if (!jacobians)
            return true;

        Matrix3d dphi_dRat =  SO3JrInv(phi)*Rotx.inverse().matrix();
        Matrix3d dphi_dRbt = -SO3JrInv(-phi);
        Matrix3d dphi_dRx  =  SO3JrInv(phi);

        Matrix3d ddel_dRat =  Rat.matrix()*SO3d::hat(Posx);

        /// Rotation control point
        for (size_t j = 0; j < N; j++)
        {
            int idx = Ra_offset + j;
            if (jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> J_knot_Ra(jacobians[idx]);
                J_knot_Ra.setZero();
                J_knot_Ra.block<3, 3>(0, 0) = wR*dphi_dRat*dRat_dRj[j];
            }
        }

        for (size_t j = 0; j < N; j++)
        {
            int idx = Rb_offset + j;
            if (jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> J_knot_Rb(jacobians[idx]);
                J_knot_Rb.setZero();
                J_knot_Rb.block<3, 3>(0, 0) = wR*dphi_dRbt*dRat_dRj[j];
            }
        }
        
        {
            size_t idx = Rx_offset;
            Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> J_knot_Rx(jacobians[idx]);
            J_knot_Rx.setZero();
            J_knot_Rx.block<3, 3>(0, 0) = wR*dphi_dRx;
        }

        /// Position control point
        for (size_t j = 0; j < N; j++)
        {
            int idx = Ra_offset + j;
            if (jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> J_knot_Ra(jacobians[idx]);
                J_knot_Ra.setZero();
                J_knot_Ra.block<3, 3>(3, 0) = wP*ddel_dRat*dRat_dRj[j];
            }
        }

        for (size_t j = 0; j < N; j++)
        {
            int idx = Pa_offset + j;
            if (jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_knot_Pa(jacobians[idx]);
                J_knot_Pa.setZero();
                J_knot_Pa.block<3, 3>(3, 0) = wP*dPat_dPj[j];
            }
        }

        for (size_t j = 0; j < N; j++)
        {
            int idx = Pb_offset + j;
            if (jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_knot_Pb(jacobians[idx]);
                J_knot_Pb.setZero();
                J_knot_Pb.block<3, 3>(3, 0) = -wP*dPbt_dPj[j];
            }
        }

        for (size_t j = 0; j < N; j++)
        {
            int idx = Px_offset + j;
            if (jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_knot_Pb(jacobians[idx]);
                J_knot_Pb.setZero();
                J_knot_Pb.block<3, 3>(3, 0) = wP*Rotx.matrix();
            }
        }

        return true;
    }

private:

    double wR;
    double wP;

    int    N;
    double Dt;     // Knot length
    double sa;     // Normalized time (t - t_i)/Dt
    double sb;     // Normalized time (t - t_i)/Dt

    // Lambda
    Matrix<double, Dynamic, 1> lambda_Ra;
    // Lambda dot
    Matrix<double, Dynamic, 1> lambda_Pa;
    // Lambda
    Matrix<double, Dynamic, 1> lambda_Rb;
    // Lambda dot
    Matrix<double, Dynamic, 1> lambda_Pb;
};
