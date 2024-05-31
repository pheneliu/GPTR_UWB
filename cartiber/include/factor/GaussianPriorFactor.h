#pragma once

#include <ceres/ceres.h>
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/utils/sophus_utils.hpp"
#include "utility.h"

using SO3d = Sophus::SO3<double>;
using SE3d = Sophus::SE3<double>;

template <typename MatrixType1, typename MatrixType2>
MatrixXd kroneckerProduct(const MatrixType1& A, const MatrixType2& B) {
    MatrixXd result(A.rows() * B.rows(), A.cols() * B.cols());
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            result.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
        }
    }
    return result;
}

class GaussianPriorFactor : public ceres::CostFunction
{

public:


    void ComputeLambdaAndDerivs(double s,
                                Matrix<double, Dynamic, 1> &lambda_R,
                                Matrix<double, Dynamic, 1> &lambda_R_dot,
                                Matrix<double, Dynamic, 1> &lambda_P,
                                Matrix<double, Dynamic, 1> &lambda_P_dot,
                                Matrix<double, Dynamic, 1> &lambda_P_ddot)
    {
        // Blending matrix for position, in standard form
        static Matrix<double, Dynamic, Dynamic> B = basalt::computeBlendingMatrix<double, false>(N);

        // Blending matrix for rotation, in cummulative form
        static Matrix<double, Dynamic, Dynamic> Btilde = basalt::computeBlendingMatrix<double, true>(N);

        // Inverse of knot length
        static double Dt_inv = 1.0/Dt;
        
        // Time powers for a
        Matrix<double, Dynamic, 1> U(N);
        for(int j = 0; j < N; j++)
            U(j) = std::pow(s, j);

        // Time power 1st derivative
        Matrix<double, Dynamic, 1> Udot = Matrix<double, Dynamic, 1>::Zero(N);
        for (int j = 1; j < N; j++)
            Udot(j) = j * std::pow(s, j - 1);

        // Time power 2nd derivative
        Matrix<double, Dynamic, 1> Uddot = Matrix<double, Dynamic, 1>::Zero(N);
        for (int j = 2; j < N; j++)
            Uddot(j) = j * (j - 1) * std::pow(s, j - 2);


        // Lambda for R
        lambda_R = Btilde * U;

        // Lambda dot
        lambda_R_dot = Dt_inv * Btilde * Udot;

        // Lambda for pa
        lambda_P = B * U;

        // Lambda derivative for velocity
        lambda_P_dot = Dt_inv * B * Udot;

        // Lambda derivative for acceleration
        lambda_P_ddot = Dt_inv * Dt_inv * B * Uddot;
    }

    GaussianPriorFactor(double wR_, double wP_, int N_, double Dt_, double sa_, double sb_, double dtba_, int DK_)
    :   wR          (wR_             ),
        wP          (wP_             ),
        N           (N_              ),
        Dt          (Dt_             ),
        sa          (sa_             ),
        sb          (sb_             ),
        dtba        (dtba_           ),
        DK          (DK_             )
    {

        // 6-element residual: (3x1 rotation, 3x1 position)
        set_num_residuals(15); // Angular diff, angular vel, pos diff, vel diff, acc diff

        for (size_t j = 0; j < N+DK; j++)
            mutable_parameter_block_sizes()->push_back(4);

        for (size_t j = 0; j < N+DK; j++)
            mutable_parameter_block_sizes()->push_back(3);

        // Calculate the spline-defining quantities:

        ComputeLambdaAndDerivs(sa, lambda_Ra, lambda_Ra_dot, lambda_Pa, lambda_Pa_dot, lambda_Pa_ddot);
        ComputeLambdaAndDerivs(sb, lambda_Rb, lambda_Rb_dot, lambda_Pb, lambda_Pb_dot, lambda_Pb_ddot);

        // Calculate the information matrix
        Matrix<double, 15, 15> Info;
        Info.setZero();

        double dtbapow[7];
        for(int j = 0; j < 7; j++)
            dtbapow[j] = pow(dtba, j);

        Matrix2d QR;
        QR << 1/3*dtbapow[3]*wR, 1/2*dtbapow[2]*wR,
              1/2*dtbapow[2]*wR,     dtbapow[1]*wR;
        Info.block<6, 6>(0, 0) = kroneckerProduct(QR, Matrix3d::Identity());

        Matrix3d QP;
        QP << 1/20*dtbapow[5]*wP, 1/8*dtbapow[4]*wP, 1/6*dtbapow[3]*wP,
              1/ 8*dtbapow[4]*wP, 1/3*dtbapow[3]*wP, 1/2*dtbapow[2]*wP,
              1/ 6*dtbapow[3]*wP, 1/2*dtbapow[2]*wP, 1/1*dtbapow[1]*wP;
        Info.block<9, 9>(6, 0) = kroneckerProduct(QP, Matrix3d::Identity());

        sqrtW = Eigen::LLT<Matrix<double, 15, 15>>(Info).matrixL().transpose();
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

    void ComputeRtAndDerivs(const SO3d Rot[],
                            const Matrix<double, Dynamic, 1> &lambda_R,
                            const Matrix<double, Dynamic, 1> &lambda_R_dot,
                            SO3d &Rt, Vector3d &Wt,
                            Matrix3d *dRt_dR,
                            Matrix3d *dWt_dR) const
    {
        // The following use the formulations in the paper 2020 CVPR:
        // "Efficient derivative computation for cumulative b-splines on lie groups."
        // Sommer, Christiane, Vladyslav Usenko, David Schubert, Nikolaus Demmel, and Daniel Cremers.
        // Some errors in the paper is corrected



        // CALCULATE Rt FROM THE ROTATIONAL CONTROL POINT

        // Calculate the delta terms: delta(1) ... delta(N-1), where delta(j) = Log( R(j-1)^-1 * R(j) ),
        // delta(0) is an extension in the paper as the index starts at 1
        Vector3d delta[N];
        delta[0] = Rot[0].log();
        for (int j = 1; j < N; j++)
            delta[j] = (Rot[j - 1].inverse() * Rot[j]).log();

        // Calculate the A terms: A(1) ... A(N-1), where A(j) = Exp( lambda(j) * d(j) ), A(0) is an extension
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
        


        // FIND THE JACOBIAN OF RT OVER THE CONTROL ROTATIONS:

        Matrix3d ddelta_dR[N];  // The inverse right Jacobian Jr(dj) = d(deltaj)/d(Rj). Note that d( d[j+1] )/d( R[j] ) = -Jr(-d[j+1])
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



        // CALCULATE THE OMEGA

        // Calculate the omega terms: omega(1) ... omega(N), using equation (38), omega(N) is the angular velocity
        Vector3d omega[N + 1];
        omega[0] = Vector3d(0, 0, 0);
        omega[1] = Vector3d(0, 0, 0);
        for (int j = 1; j < N + 1; j++)
            omega[j] = A[j - 1].inverse() * omega[j - 1] + lambda_R_dot(j - 1) * delta[j - 1];

        Wt = omega[N];



        // CALCULATE THE JACOBIAN OF OMEGA

        // Jacobian of d(omega)/d(deltaj)
        Matrix3d domega_ddelta[N];
        for (int j = 1; j < N; j++)
            domega_ddelta[j] = P[j].matrix() * (lambda_R(j) * A[j].matrix().transpose() * SO3d::hat(omega[j]) * JrLambaDelta[j].transpose()
                                                + lambda_R_dot[j] * Matrix3d::Identity());

        // Jacobian of d(omega)/d(Rj)
        Matrix3d *domega_dR = dWt_dR;
        for (int j = 0; j < N; j++)
        {
            domega_dR[j].setZero();

            if (j == 0)
                domega_dR[j] = -domega_ddelta[1] * ddelta_dR[1].transpose();
            else if (j == N - 1)
                domega_dR[j] = domega_ddelta[j] * ddelta_dR[j];
            else
                domega_dR[j] = domega_ddelta[j] * ddelta_dR[j] - domega_ddelta[j + 1] * ddelta_dR[j + 1].transpose();
        }

    }

    void ComputePtAndDerivs(const Vector3d Pos[],
                            const Matrix<double, Dynamic, 1> &lambda_P,
                            const Matrix<double, Dynamic, 1> &lambda_P_dot,
                            const Matrix<double, Dynamic, 1> &lambda_P_ddot,
                            Vector3d &Pt, Vector3d &Vt, Vector3d &At,
                            Matrix3d *dPt_dP,
                            Matrix3d *dVt_dP,
                            Matrix3d *dAt_dP) const
    {
        // Predicted position
        for (int j = 0; j < N; j++)
        {
            Pt += Pos[j]*lambda_P[j];
            Vt += Pos[j]*lambda_P_dot[j];
            At += Pos[j]*lambda_P_ddot[j];

            dPt_dP[j] = Vector3d(lambda_P[j], lambda_P[j], lambda_P[j]).asDiagonal();
            dVt_dP[j] = Vector3d(lambda_P_dot[j], lambda_P_dot[j], lambda_P_dot[j]).asDiagonal();
            dAt_dP[j] = Vector3d(lambda_P_ddot[j], lambda_P_ddot[j], lambda_P_ddot[j]).asDiagonal();
        }
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Check if the knot difference is larger than spline order
        int DK_ = min(N, DK);                   // If DK is smaller than N then there is no overlap

        // Indexing offsets for the states
        size_t Ra_offset = 0;                    // for quaternion
        size_t Rb_offset = Ra_offset + DK_;      // for quaternion

        size_t Pa_offset = Rb_offset + N;        // for position
        size_t Pb_offset = Pa_offset + DK_;      // for position

        // Map parameters to the control point states
        SO3d Rot[N + DK_]; Vector3d Pos[N + DK_];
        
        SO3d *Rota = &Rot[Ra_offset];
        SO3d *Rotb = &Rot[Rb_offset];
        
        Vector3d Posa[Pa_offset];
        Vector3d Posb[Pb_offset];

        for (int j = 0; j < N + DK_; j++)
        {
            Rot[j] = Eigen::Map<SO3d const>(parameters[Ra_offset + j]);
            Pos[j] = Eigen::Map<Vector3d const>(parameters[Pa_offset + j]);
        }

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the residual ---------------------------------------------------------------------------*/

        SO3d     Ra;
        Vector3d Wa(0, 0, 0);
        Vector3d Pa(0, 0, 0);
        Vector3d Va(0, 0, 0);
        Vector3d Aa(0, 0, 0);
        Matrix3d dRa_dRj[N + DK_] = {Matrix3d::Zero()};
        Matrix3d dWa_dRj[N + DK_] = {Matrix3d::Zero()};
        Matrix3d dPa_dPj[N + DK_] = {Matrix3d::Zero()};
        Matrix3d dVa_dPj[N + DK_] = {Matrix3d::Zero()};
        Matrix3d dAa_dPj[N + DK_] = {Matrix3d::Zero()};
        
        SO3d     Rb;
        Vector3d Wb(0, 0, 0);
        Vector3d Pb(0, 0, 0);
        Vector3d Vb(0, 0, 0);
        Vector3d Ab(0, 0, 0);
        Matrix3d dRb_dRj[N + DK_] = {Matrix3d::Zero()};
        Matrix3d dWb_dRj[N + DK_] = {Matrix3d::Zero()};
        Matrix3d dPb_dPj[N + DK_] = {Matrix3d::Zero()};
        Matrix3d dVb_dPj[N + DK_] = {Matrix3d::Zero()};
        Matrix3d dAb_dPj[N + DK_] = {Matrix3d::Zero()};

        ComputeRtAndDerivs(&Rot[Ra_offset], lambda_Ra, lambda_Ra_dot, Ra, Wa, &dRa_dRj[Ra_offset], &dWa_dRj[Ra_offset]);
        ComputeRtAndDerivs(&Rot[Rb_offset], lambda_Rb, lambda_Rb_dot, Rb, Wb, &dRb_dRj[Rb_offset], &dWb_dRj[Rb_offset]);

        ComputePtAndDerivs(&Pos[Pa_offset], lambda_Pa, lambda_Pa_dot, lambda_Pa_ddot, Pa, Va, Aa, &dPa_dPj[Pa_offset], &dVa_dPj[Pa_offset], &dAa_dPj[Pa_offset]);
        ComputePtAndDerivs(&Pos[Pb_offset], lambda_Pb, lambda_Pb_dot, lambda_Pb_ddot, Pb, Vb, Ab, &dPb_dPj[Pb_offset], &dVb_dPj[Pb_offset], &dAb_dPj[Pb_offset]);
        
        // Relative rotation
        SO3d Rab = Ra.inverse()*Rb;

        // Rotation vector
        Vector3d phib = Rab.log();

        // Rotational residual
        Vector3d rphi = phib - dtba*Wa;

        // Rotational rate residual
        Vector3d rphidot = SO3JrInv(phib)*Wb;

        // Positional residual
        Vector3d rpos = Pb - (Pa + dtba*Va + dtba*dtba/2*Aa);

        // Velocity rate residual
        Vector3d rvel = Vb - (Va + dtba*Aa);

        // Acceleration residual
        Vector3d racc = Ab - Aa;

        // Residual
        Eigen::Map<Matrix<double, 15, 1>> residual(residuals);
        residual << rphi, rphidot, rpos, rvel, racc;

        // cout << "Residual: " << residual << endl;

        residual =  sqrtW*residual;

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        if (!jacobians)
            return true;

        // Jacobian on the rphi part
        Matrix3d dphib_dRa =  SO3JrInv(phib)*Rab.inverse().matrix();
        Matrix3d dphib_dRb =  SO3JrInv(phib);

        for (size_t j = 0; j < N + DK_; j++)
        {
            int idx = j;
            if (jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> Jrphi_knot_R(jacobians[idx]);
                Jrphi_knot_R.setZero();

                // Jacobian on rotation
                Jrphi_knot_R.block<3, 3>(0, 0) = (dphib_dRa*dRa_dRj[j] + dphib_dRb*dRb_dRj[j] - dtba*dWa_dRj[j]);

                // cout << myprintf("Jrphi_knot_R: %d\n", j) << Jrphi_knot_R << endl;
                
                // Mulitply the noise
                Jrphi_knot_R = sqrtW*Jrphi_knot_R;
            }
        }

        // Jacobian on the rphidot part
        for (size_t j = 0; j < N + DK_; j++)
        {
            int idx = j;
            if (jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> Jrphidot_knot_R(jacobians[idx]);
                Jrphidot_knot_R.setZero();

                // Jacobian on rotation
                Jrphidot_knot_R.block<3, 3>(3, 0) = (dWb_dRj[j] - 0.5*SO3d::hat(Wb)*(dphib_dRa*dRa_dRj[j] + dphib_dRb*dRb_dRj[j])
                                                                + 0.5*SO3d::hat(phib)*dWb_dRj[j]
                                                    );

                // cout << myprintf("Jrphidot_knot_R: %d\n", j) << Jrphidot_knot_R << endl;

                // Mulitply the noise
                Jrphidot_knot_R = sqrtW*Jrphidot_knot_R;
            }
        }

        for (size_t j = 0; j < N + DK_; j++)
        {
            int idx = Pa_offset + j;
            if (jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Jpos_knot_P(jacobians[idx]);
                Jpos_knot_P.setZero();
                
                // Jacobian on position
                Jpos_knot_P.block<3, 3>(6, 0) = dPb_dPj[j] - (dPa_dPj[j] + dtba*dVa_dPj[j] + 0.5*dtba*dtba*dAa_dPj[j]);
                
                // cout << myprintf("Jpos_knot_P: %d\n", j) << Jpos_knot_P << endl;

                // Mulitply the noise
                Jpos_knot_P = sqrtW*Jpos_knot_P;
            }
        }
        
        for (size_t j = 0; j < N + DK_; j++)
        {
            int idx = Pa_offset + j;
            if (jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Jvel_knot_P(jacobians[idx]);
                Jvel_knot_P.setZero();

                // Jacobian on position
                Jvel_knot_P.block<3, 3>(9, 0) = dVb_dPj[j] - (dVa_dPj[j] + dtba*dAa_dPj[j]);

                // cout << myprintf("Jvel_knot_P: %d\n", j) << Jvel_knot_P << endl;

                // Mulitply the noise
                Jvel_knot_P = sqrtW*Jvel_knot_P;
            }
        }
        
        for (size_t j = 0; j < N + DK_; j++)
        {
            int idx = Pa_offset + j;
            if (jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Jacc_knot_P(jacobians[idx]);
                Jacc_knot_P.setZero();

                // Jacobian on position control points
                Jacc_knot_P.block<3, 3>(12, 0) = dAb_dPj[j] - dAa_dPj[j];

                // cout << myprintf("Jacc_knot_P: %d\n", j) << Jacc_knot_P << endl;

                // Mulitply the noise
                Jacc_knot_P = sqrtW*Jacc_knot_P;
            }
        }
        
        // exit(-1);

        return true;
    }

private:

    double wR;
    double wP;

    // Matrix<double, 15, 15> Info;
    Matrix<double, 15, 15> sqrtW;

    int    N;
    double Dt;     // Knot length
    double sa;     // Normalized time (t - ta)/Dt
    double sb;     // Normalized time (t - tb)/Dt
    double dtba;   // Time difference tb - ta

    int DK;        // Number of knots apart

    // Lambda for rot
    Matrix<double, Dynamic, 1> lambda_Ra;
    Matrix<double, Dynamic, 1> lambda_Ra_dot;
    // Lambda for pos
    Matrix<double, Dynamic, 1> lambda_Pa;
    Matrix<double, Dynamic, 1> lambda_Pa_dot;
    Matrix<double, Dynamic, 1> lambda_Pa_ddot;
    
    // Lambda for rot
    Matrix<double, Dynamic, 1> lambda_Rb;
    Matrix<double, Dynamic, 1> lambda_Rb_dot;
    // Lambda for pos
    Matrix<double, Dynamic, 1> lambda_Pb;
    Matrix<double, Dynamic, 1> lambda_Pb_dot;
    Matrix<double, Dynamic, 1> lambda_Pb_ddot;
};
