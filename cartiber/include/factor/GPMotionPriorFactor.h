#pragma once

#include <ceres/ceres.h>
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/utils/sophus_utils.hpp"
#include "GaussianProcess.hpp"
#include "utility.h"

using SO3d = Sophus::SO3<double>;
using SE3d = Sophus::SE3<double>;

// template <typename MatrixType1, typename MatrixType2>
// MatrixXd kroneckerProduct(const MatrixType1& A, const MatrixType2& B) {
//     MatrixXd result(A.rows() * B.rows(), A.cols() * B.cols());
//     for (int i = 0; i < A.rows(); ++i) {
//         for (int j = 0; j < A.cols(); ++j) {
//             result.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
//         }
//     }
//     return result;
// }

class GPMotionPriorFactor : public ceres::CostFunction
{

public:

    GPMotionPriorFactor(double wR_, double wP_, double Dt_, double ss_, double sf_, double dtsf_)
    :   wR          (wR_             ),
        wP          (wP_             ),
        Dt          (Dt_             ),
        ss          (ss_             ),
        sf          (sf_             ),
        dtsf        (dtsf_           ),
        gpm         (Dt_             )
    {

        // 6-element residual: (3x1 rotation, 3x1 position)
        set_num_residuals(15); // Angular diff, angular vel, pos diff, vel diff, acc diff

        for(int j = 0; j < 4; j++)
        {
            mutable_parameter_block_sizes()->push_back(4);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
        }

        // Calculate the information matrix
        Matrix<double, 15, 15> Info;
        Info.setZero();

        double dtsfpow[7];
        for(int j = 0; j < 7; j++)
            dtsfpow[j] = pow(dtsf, j);

        Matrix2d QR;
        QR << 1/3*dtsfpow[3]*wR, 1/2*dtsfpow[2]*wR,
              1/2*dtsfpow[2]*wR,     dtsfpow[1]*wR;
        Info.block<6, 6>(0, 0) = kroneckerProduct(QR, Matrix3d::Identity());

        Matrix3d QP;
        QP << 1/20*dtsfpow[5]*wP, 1/8*dtsfpow[4]*wP, 1/6*dtsfpow[3]*wP,
              1/ 8*dtsfpow[4]*wP, 1/3*dtsfpow[3]*wP, 1/2*dtsfpow[2]*wP,
              1/ 6*dtsfpow[3]*wP, 1/2*dtsfpow[2]*wP, 1/1*dtsfpow[1]*wP;
        Info.block<9, 9>(6, 0) = kroneckerProduct(QP, Matrix3d::Identity());
        sqrtW = Eigen::LLT<Matrix<double, 15, 15>>(Info).matrixL().transpose();
    }

    Matrix3d Jr(const Vector3d &phi) const
    {
        Matrix3d Jr;
        Sophus::rightJacobianSO3(phi, Jr);
        return Jr;
    }

    Matrix3d JrInv(const Vector3d &phi) const
    {
        Matrix3d Jr_inv;
        Sophus::rightJacobianInvSO3(phi, Jr_inv);
        return Jr_inv;
    }

    // void ComputeXtAndDerivs(const StateStamped &Xa, const StateStamped &Xb, StateStamped &Xt,
    //                         vector<vector<Matrix3d>> &DXt_DXa, vector<vector<Matrix3d>> &DXt_DXb) const
    // {
    //     // Map the variables of the state
    //     double tau = Xt.t; SO3d &Rt = Xt.R; Vec3 &Ot = Xt.O; Vec3 &Pt = Xt.P; Vec3 &Vt = Xt.V; Vec3 &At = Xt.A;
        
    //     // Calculate the the mixer matrixes
    //     MatrixXd LAM_ROt  = gpm.LAMBDA_RO(tau);
    //     MatrixXd PSI_ROt  = gpm.PSI_RO(tau);
    //     MatrixXd LAM_PVAt = gpm.LAMBDA_PVA(tau);
    //     MatrixXd PSI_PVAt = gpm.PSI_PVA(tau);

    //     // Extract the blocks of SO3 states
    //     Matrix3d LAM_RO11 = LAM_ROt.block<3, 3>(0, 0);
    //     Matrix3d LAM_RO12 = LAM_ROt.block<3, 3>(0, 3);

    //     Matrix3d LAM_RO21 = LAM_ROt.block<3, 3>(3, 0);
    //     Matrix3d LAM_RO22 = LAM_ROt.block<3, 3>(3, 3);

    //     Matrix3d PSI_RO11 = PSI_ROt.block<3, 3>(0, 0);
    //     Matrix3d PSI_RO12 = PSI_ROt.block<3, 3>(0, 3);

    //     Matrix3d PSI_RO21 = PSI_ROt.block<3, 3>(3, 0);
    //     Matrix3d PSI_RO22 = PSI_ROt.block<3, 3>(3, 3);

    //     // Extract the blocks of R3 states
    //     Matrix3d LAM_PVA11 = LAM_PVAt.block<3, 3>(0, 0);
    //     Matrix3d LAM_PVA12 = LAM_PVAt.block<3, 3>(0, 3);
    //     Matrix3d LAM_PVA13 = LAM_PVAt.block<3, 3>(0, 6);

    //     Matrix3d LAM_PVA21 = LAM_PVAt.block<3, 3>(3, 0);
    //     Matrix3d LAM_PVA22 = LAM_PVAt.block<3, 3>(3, 3);
    //     Matrix3d LAM_PVA23 = LAM_PVAt.block<3, 3>(3, 6);

    //     Matrix3d LAM_PVA31 = LAM_PVAt.block<3, 3>(6, 0);
    //     Matrix3d LAM_PVA32 = LAM_PVAt.block<3, 3>(6, 3);
    //     Matrix3d LAM_PVA33 = LAM_PVAt.block<3, 3>(6, 6);

    //     Matrix3d PSI_PVA11 = PSI_PVAt.block<3, 3>(0, 0);
    //     Matrix3d PSI_PVA12 = PSI_PVAt.block<3, 3>(0, 3);
    //     Matrix3d PSI_PVA13 = PSI_PVAt.block<3, 3>(0, 6);

    //     Matrix3d PSI_PVA21 = PSI_PVAt.block<3, 3>(3, 0);
    //     Matrix3d PSI_PVA22 = PSI_PVAt.block<3, 3>(3, 3);
    //     Matrix3d PSI_PVA23 = PSI_PVAt.block<3, 3>(3, 6);

    //     Matrix3d PSI_PVA31 = PSI_PVAt.block<3, 3>(6, 0);
    //     Matrix3d PSI_PVA32 = PSI_PVAt.block<3, 3>(6, 3);
    //     Matrix3d PSI_PVA33 = PSI_PVAt.block<3, 3>(6, 6);

    //     // Find the relative rotation 
    //     SO3d Rab = Xa.R.inverse()*Xb.R;

    //     // Calculate the SO3 knots in relative form
    //     Vec3 thetaa    = Vector3d(0, 0, 0);
    //     Vec3 thetadota = Xa.O;
    //     Vec3 thetab    = Rab.log();
    //     Vec3 thetadotb = JrInv(thetab)*Xb.O;
    //     // Put them in vector form
    //     Matrix<double, 6, 1> gammaa; gammaa << thetaa, thetadota;
    //     Matrix<double, 6, 1> gammab; gammab << thetab, thetadotb;

    //     // Calculate the knot euclid states and put them in vector form
    //     Matrix<double, 9, 1> pvaa; pvaa << Xa.P, Xa.V, Xa.A;
    //     Matrix<double, 9, 1> pvab; pvab << Xb.P, Xb.V, Xb.A;

    //     // Mix the knots to get the interpolated states
    //     Matrix<double, 6, 1> gammat = LAM_ROt*gammaa + PSI_ROt*gammab;
    //     Matrix<double, 9, 1> pvat   = LAM_PVAt*pvaa  + PSI_PVAt*pvab;

    //     // Retrive the interpolated SO3 in relative form
    //     Vector3d thetat    = gammat.block<3, 1>(0, 0);
    //     Vector3d thetadott = gammat.block<3, 1>(3, 0);

    //     // Assign the interpolated state
    //     Rt = Xa.R*SO3d::exp(thetat);
    //     Ot = JrInv(thetat)*thetadott;
    //     Pt = pvat.block<3, 1>(0, 0);
    //     Vt = pvat.block<3, 1>(3, 0);
    //     At = pvat.block<3, 1>(6, 0);

    //     // Calculate the Jacobian
    //     DXt_DXa = vector<vector<Matrix3d>>(5, vector<Matrix3d>(5, Matrix3d::Zero()));
    //     DXt_DXb = vector<vector<Matrix3d>>(5, vector<Matrix3d>(5, Matrix3d::Zero()));

    //     // Local index for the states in the state vector
    //     const int RIDX = 0;
    //     const int OIDX = 1;
    //     const int PIDX = 2;
    //     const int VIDX = 3;
    //     const int AIDX = 4;
        
    //     // DRt_DRa
    //     DXt_DXa[RIDX][RIDX] = SO3d::exp(-thetat).matrix() - Jr(thetat)*(PSI_RO11 - PSI_RO12*SO3d::hat(Xb.O))*JrInv(thetab)*Rab.inverse().matrix();
    //     // DRt_DOa
    //     DXt_DXa[RIDX][OIDX] = Jr(thetat)*LAM_RO12;
        
    //     // DRt_DPa DRt_DVa DRt_DAa are all zeros
        
    //     // TODO:
    //     // DOt_Ra still needs to be computed
    //     // DXt_DXa[OIDX][RIDX];
    //     // DOt_Oa still needs to be computed
    //     // DXt_DXa[OIDX][OIDX];
        
    //     // DOt_DPa DOt_DVa DOt_DAa are all zeros

    //     // DPt_DRa DPt_DOa are all all zeros
    //     // DPt_DPa
    //     DXt_DXa[PIDX][PIDX] = LAM_PVA11;
    //     // DPt_DVa
    //     DXt_DXa[PIDX][VIDX] = LAM_PVA12;
    //     // DPt_DAa
    //     DXt_DXa[PIDX][AIDX] = LAM_PVA13;
        
    //     // DVt_DPa
    //     DXt_DXa[VIDX][PIDX] = LAM_PVA21;
    //     // DVt_DVa
    //     DXt_DXa[VIDX][VIDX] = LAM_PVA22;
    //     // DVt_DAa
    //     DXt_DXa[VIDX][AIDX] = LAM_PVA23;

    //     // DAt_DPa
    //     DXt_DXa[AIDX][PIDX] = LAM_PVA31;
    //     // DAt_DVa
    //     DXt_DXa[AIDX][VIDX] = LAM_PVA32;
    //     // DAt_DAa
    //     DXt_DXa[AIDX][AIDX] = LAM_PVA33;




    //     // DRt_DRb
    //     DXt_DXb[RIDX][RIDX] = Jr(thetat)*(PSI_RO11 - PSI_RO12*SO3d::hat(Xb.O))*JrInv(thetab);
    //     // DRt_DOb
    //     DXt_DXb[RIDX][OIDX] = Jr(thetat)*PSI_RO12*JrInv(thetab);
    //     // DRt_DPb DRt_DVb DRt_DAb are all zeros

    //     // DRt_DPb DRt_DVb DRt_DAb are all zeros
        
    //     // TODO:
    //     // DOt_Rb still needs to be computed
    //     // DXt_DXb[OIDX][RIDX];
    //     // DOt_Ob still needs to be computed
    //     // DXt_DXb[OIDX][OIDX];
        
    //     // DOt_DPb DOt_DVb DOt_DAb are all zeros

    //     // DPt_DRb DPt_DOb are all all zeros
    //     // DPt_DPb
    //     DXt_DXb[PIDX][PIDX] = PSI_PVA11;
    //     // DRt_DPb
    //     DXt_DXb[PIDX][VIDX] = PSI_PVA12;
    //     // DRt_DAb
    //     DXt_DXb[PIDX][AIDX] = PSI_PVA13;

    //     // DVt_DPb
    //     DXt_DXb[VIDX][PIDX] = PSI_PVA21;
    //     // DVt_DVb
    //     DXt_DXb[VIDX][VIDX] = PSI_PVA22;
    //     // DVt_DAb
    //     DXt_DXb[VIDX][AIDX] = PSI_PVA23;
        
    //     // DAt_DPb
    //     DXt_DXb[AIDX][PIDX] = PSI_PVA21;
    //     // DAt_DVb
    //     DXt_DXb[AIDX][VIDX] = PSI_PVA22;
    //     // DAt_DAb
    //     DXt_DXb[AIDX][AIDX] = PSI_PVA23;
    // }

    void MapParamToState(double const *const *parameters, int base, StateStamped &X) const
    {
        X.R = Eigen::Map<SO3d const>(parameters[base + 0]);
        X.O = Eigen::Map<Vec3 const>(parameters[base + 1]);
        X.P = Eigen::Map<Vec3 const>(parameters[base + 2]);
        X.V = Eigen::Map<Vec3 const>(parameters[base + 3]);
        X.A = Eigen::Map<Vec3 const>(parameters[base + 4]);
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        StateStamped Xsa(0);  MapParamToState(parameters, RsaIdx, Xsa);
        StateStamped Xsb(Dt); MapParamToState(parameters, RsbIdx, Xsb);
        StateStamped Xfa(0);  MapParamToState(parameters, RfaIdx, Xfa);
        StateStamped Xfb(Dt); MapParamToState(parameters, RfbIdx, Xfb);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        // /* #region Calculate the residual ---------------------------------------------------------------------------*/

        StateStamped Xs(ss*Dt); vector<vector<Matrix3d>> DXs_DXsa; vector<vector<Matrix3d>> DXs_DXsb;
        StateStamped Xf(sf*Dt); vector<vector<Matrix3d>> DXf_DXfa; vector<vector<Matrix3d>> DXf_DXfb;

        gpm.ComputeXtAndDerivs(Xsa, Xsb, Xs, DXs_DXsa, DXs_DXsb);
        gpm.ComputeXtAndDerivs(Xfa, Xfb, Xf, DXf_DXfa, DXf_DXfb);

        // ComputePtAndDerivs(&Pos[Pa_offset], lambda_Pa, lambda_Pa_dot, lambda_Pa_ddot, Pa, Va, Aa, &dPa_dPj[Pa_offset], &dVa_dPj[Pa_offset], &dAa_dPj[Pa_offset]);
        // ComputePtAndDerivs(&Pos[Pb_offset], lambda_Pb, lambda_Pb_dot, lambda_Pb_ddot, Pb, Vb, Ab, &dPb_dPj[Pb_offset], &dVb_dPj[Pb_offset], &dAb_dPj[Pb_offset]);
        
        // // Relative rotation
        // SO3d Rab = Ra.inverse()*Rb;

        // // Rotation vector
        // Vector3d phib = Rab.log();

        // // Rotational residual
        // Vector3d rphi = phib - dtba*Wa;

        // // Rotational rate residual
        // Vector3d rphidot = JrInv(phib)*Wb;

        // // Positional residual
        // Vector3d rpos = Pb - (Pa + dtba*Va + dtba*dtba/2*Aa);

        // // Velocity rate residual
        // Vector3d rvel = Vb - (Va + dtba*Aa);

        // // Acceleration residual
        // Vector3d racc = Ab - Aa;

        // // Residual
        // Eigen::Map<Matrix<double, 15, 1>> residual(residuals);
        // residual << rphi, rphidot, rpos, rvel, racc;

        // // cout << "Residual: " << residual << endl;

        // residual =  sqrtW*residual;

        // /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        // if (!jacobians)
        //     return true;

        // // Jacobian on the rphi part
        // Matrix3d dphib_dRa =  JrInv(phib)*Rab.inverse().matrix();
        // Matrix3d dphib_dRb =  JrInv(phib);

        // for (size_t j = 0; j < N + DK_; j++)
        // {
        //     int idx = j;
        //     if (jacobians[idx])
        //     {
        //         Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> Jrphi_knot_R(jacobians[idx]);
        //         Jrphi_knot_R.setZero();

        //         // Jacobian on rotation
        //         Jrphi_knot_R.block<3, 3>(0, 0) = (dphib_dRa*dRa_dRj[j] + dphib_dRb*dRb_dRj[j] - dtba*dWa_dRj[j]);

        //         // cout << myprintf("Jrphi_knot_R: %d\n", j) << Jrphi_knot_R << endl;
                
        //         // Mulitply the noise
        //         Jrphi_knot_R = sqrtW*Jrphi_knot_R;
        //     }
        // }

        // // Jacobian on the rphidot part
        // for (size_t j = 0; j < N + DK_; j++)
        // {
        //     int idx = j;
        //     if (jacobians[idx])
        //     {
        //         Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> Jrphidot_knot_R(jacobians[idx]);
        //         Jrphidot_knot_R.setZero();

        //         // Jacobian on rotation
        //         Jrphidot_knot_R.block<3, 3>(3, 0) = (dWb_dRj[j] - 0.5*SO3d::hat(Wb)*(dphib_dRa*dRa_dRj[j] + dphib_dRb*dRb_dRj[j])
        //                                                         + 0.5*SO3d::hat(phib)*dWb_dRj[j]
        //                                             );

        //         // cout << myprintf("Jrphidot_knot_R: %d\n", j) << Jrphidot_knot_R << endl;

        //         // Mulitply the noise
        //         Jrphidot_knot_R = sqrtW*Jrphidot_knot_R;
        //     }
        // }

        // for (size_t j = 0; j < N + DK_; j++)
        // {
        //     int idx = Pa_offset + j;
        //     if (jacobians[idx])
        //     {
        //         Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Jpos_knot_P(jacobians[idx]);
        //         Jpos_knot_P.setZero();
                
        //         // Jacobian on position
        //         Jpos_knot_P.block<3, 3>(6, 0) = dPb_dPj[j] - (dPa_dPj[j] + dtba*dVa_dPj[j] + 0.5*dtba*dtba*dAa_dPj[j]);
                
        //         // cout << myprintf("Jpos_knot_P: %d\n", j) << Jpos_knot_P << endl;

        //         // Mulitply the noise
        //         Jpos_knot_P = sqrtW*Jpos_knot_P;
        //     }
        // }
        
        // for (size_t j = 0; j < N + DK_; j++)
        // {
        //     int idx = Pa_offset + j;
        //     if (jacobians[idx])
        //     {
        //         Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Jvel_knot_P(jacobians[idx]);
        //         Jvel_knot_P.setZero();

        //         // Jacobian on position
        //         Jvel_knot_P.block<3, 3>(9, 0) = dVb_dPj[j] - (dVa_dPj[j] + dtba*dAa_dPj[j]);

        //         // cout << myprintf("Jvel_knot_P: %d\n", j) << Jvel_knot_P << endl;

        //         // Mulitply the noise
        //         Jvel_knot_P = sqrtW*Jvel_knot_P;
        //     }
        // }
        
        // for (size_t j = 0; j < N + DK_; j++)
        // {
        //     int idx = Pa_offset + j;
        //     if (jacobians[idx])
        //     {
        //         Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Jacc_knot_P(jacobians[idx]);
        //         Jacc_knot_P.setZero();

        //         // Jacobian on position control points
        //         Jacc_knot_P.block<3, 3>(12, 0) = dAb_dPj[j] - dAa_dPj[j];

        //         // cout << myprintf("Jacc_knot_P: %d\n", j) << Jacc_knot_P << endl;

        //         // Mulitply the noise
        //         Jacc_knot_P = sqrtW*Jacc_knot_P;
        //     }
        // }
        
        // // exit(-1);

        // return true;
    }

private:

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
