#pragma once

#include <ceres/ceres.h>
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/utils/sophus_utils.hpp"
#include "GaussianProcess.hpp"
#include "utility.h"

using SO3d = Sophus::SO3<double>;
using SE3d = Sophus::SE3<double>;

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
        QR << 1.0/3.0*dtsfpow[3]*wR, 1.0/2.0*dtsfpow[2]*wR,
              1.0/2.0*dtsfpow[2]*wR,         dtsfpow[1]*wR;
        Info.block<6, 6>(0, 0) = gpm.kron(QR.inverse(), Matrix3d::Identity());

        Matrix3d QP;
        QP << 1.0/20.0*dtsfpow[5]*wP, 1.0/8.0*dtsfpow[4]*wP, 1.0/6.0*dtsfpow[3]*wP,
              1.0/ 8.0*dtsfpow[4]*wP, 1.0/3.0*dtsfpow[3]*wP, 1.0/2.0*dtsfpow[2]*wP,
              1.0/ 6.0*dtsfpow[3]*wP, 1.0/2.0*dtsfpow[2]*wP, 1.0/1.0*dtsfpow[1]*wP;
        Info.block<9, 9>(6, 6) = gpm.kron(QP.inverse(), Matrix3d::Identity());
        
        // Find the square root info
        // sqrtW = Matrix<double, 15, 15>::Identity(15, 15);
        sqrtW = Eigen::LLT<Matrix<double, 15, 15>>(Info.inverse()).matrixL().transpose();
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        GPState Xsa(0);  gpm.MapParamToState(parameters, RsaIdx, Xsa);
        GPState Xsb(Dt); gpm.MapParamToState(parameters, RsbIdx, Xsb);
        GPState Xfa(0);  gpm.MapParamToState(parameters, RfaIdx, Xfa);
        GPState Xfb(Dt); gpm.MapParamToState(parameters, RfbIdx, Xfb);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the residual ---------------------------------------------------------------------------*/

        GPState Xs(ss*Dt); vector<vector<Matrix3d>> DXs_DXsa; vector<vector<Matrix3d>> DXs_DXsb;
        GPState Xf(sf*Dt); vector<vector<Matrix3d>> DXf_DXfa; vector<vector<Matrix3d>> DXf_DXfb;

        Eigen::Matrix<double, 6, 1> gammasa;
        Eigen::Matrix<double, 6, 1> gammasb;
        Eigen::Matrix<double, 6, 1> gammas;
        Eigen::Matrix<double, 6, 1> gammafa;
        Eigen::Matrix<double, 6, 1> gammafb;
        Eigen::Matrix<double, 6, 1> gammaf;

        gpm.ComputeXtAndDerivs(Xsa, Xsb, Xs, DXs_DXsa, DXs_DXsb, gammasa, gammasb, gammas);
        gpm.ComputeXtAndDerivs(Xfa, Xfb, Xf, DXf_DXfa, DXf_DXfb, gammafa, gammafb, gammaf);

        // Relative rotation and its rate
        SO3d Rsf        = Xs.R.inverse()*Xf.R;
        Vec3 thetaf     = Rsf.log();
        Vec3 thetadotf  = gpm.JrInv(thetaf)*Xf.O;

        Vec3 thetasb    = gammasb.block(0, 0, 3, 1);
        Vec3 thetadotsb = gammasb.block(3, 0, 3, 1);
        Vec3 thetas     = gammas.block(0, 0, 3, 1);
        Vec3 thetadots  = gammas.block(3, 0, 3, 1);
        
        // Rotational residual
        Vec3 rRot = thetaf - dtsf*Xs.O;

        // Rotational rate residual
        Vec3 rRdot = thetadotf - Xs.O;

        // Positional residual
        Vec3 rPos = Xf.P - (Xs.P + dtsf*Xs.V + dtsf*dtsf/2*Xs.A);

        // Velocity residual
        Vec3 rVel = Xf.V - (Xs.V + dtsf*Xs.A);

        // Acceleration residual
        Vec3 rAcc = Xf.A - Xs.A;

        // Residual
        Eigen::Map<Matrix<double, 15, 1>> residual(residuals);
        residual << rRot, rRdot, rPos, rVel, rAcc;
        residual = sqrtW*residual;

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        if (!jacobians)
            return true;
        
        size_t idx;

        // Reusable Jacobians
        Mat3 JrInvthetaf =  gpm.JrInv(thetaf);
        Mat3 Dthetaf_DRs = -JrInvthetaf*Rsf.inverse().matrix();
        
        Mat3 Dthetadotf_Dthetaf = gpm.DJrInvXV_DX(thetaf, Xf.O);
        Mat3 Dthetadotf_DRs = Dthetadotf_Dthetaf*Dthetaf_DRs;
        Mat3 Dthetadotf_DRa = Dthetadotf_DRs*DXs_DXsa[Ridx][Ridx];
        Mat3 Dthetadotf_DRb = Dthetadotf_DRs*DXs_DXsb[Ridx][Ridx];
        
        Mat3 Dthetaf_DRf = JrInvthetaf;
        Mat3 Dthetadotf_DRf = Dthetadotf_Dthetaf*Dthetaf_DRf;

        // Work out the Jacobians for SO3 states first
        {
            // dr_dRsa
            idx = RsaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> Dr_DRsa(jacobians[idx]);
                Dr_DRsa.setZero();
                Dr_DRsa.block<3, 3>(0, 0) = Dthetaf_DRs*DXs_DXsa[Ridx][Ridx] - dtsf*DXs_DXsa[Oidx][Ridx];
                Dr_DRsa.block<3, 3>(3, 0) = Dthetadotf_DRa - DXs_DXsa[Oidx][Ridx];
                Dr_DRsa = sqrtW*Dr_DRsa;

                // cout << "Dr_DRsa\n" << Dr_DRsa << endl;
            }

            // dr_dOsa
            idx = OsaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Dr_DOsa(jacobians[idx]);
                Dr_DOsa.setZero();
                Dr_DOsa.block<3, 3>(0, 0) = Dthetaf_DRs*DXs_DXsa[Ridx][Oidx] - dtsf*DXs_DXsa[Oidx][Oidx];
                Dr_DOsa.block<3, 3>(3, 0) = gpm.DJrInvXV_DX(thetaf, Xf.O)*Dthetaf_DRs*DXs_DXsa[Ridx][Oidx] - DXs_DXsa[Oidx][Oidx];
                Dr_DOsa = sqrtW*Dr_DOsa;

                // cout << "Dr_DOsa\n" << Dr_DOsa << endl;
            }

            // dr_dRsb
            idx = RsbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> Dr_DRsb(jacobians[idx]);
                Dr_DRsb.setZero();
                Dr_DRsb.block<3, 3>(0, 0) = Dthetaf_DRs*DXs_DXsb[Ridx][Ridx] - dtsf*DXs_DXsb[Oidx][Ridx];
                Dr_DRsb.block<3, 3>(3, 0) = Dthetadotf_DRb - DXs_DXsb[Oidx][Ridx];
                Dr_DRsb = sqrtW*Dr_DRsb;

                // cout << "Dr_DRsb\n" << Dr_DRsb << endl;
            }
            
            // dr_dOsb
            idx = OsbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Dr_DOsb(jacobians[idx]);
                Dr_DOsb.setZero();
                Dr_DOsb.block<3, 3>(0, 0) = Dthetaf_DRs*DXs_DXsb[Ridx][Oidx] - dtsf*DXs_DXsb[Oidx][Oidx];
                Dr_DOsb.block<3, 3>(3, 0) = gpm.DJrInvXV_DX(thetaf, Xf.O)*Dthetaf_DRs*DXs_DXsb[Ridx][Oidx] - DXs_DXsb[Oidx][Oidx];
                Dr_DOsb = sqrtW*Dr_DOsb;

                // cout << "Dr_DOsb\n" << Dr_DOsb << endl;
            }

            // dr_dRfa
            idx = RfaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> Dr_DRfa(jacobians[idx]);
                Dr_DRfa.setZero();
                Dr_DRfa.block<3, 3>(0, 0) = Dthetaf_DRf*DXf_DXfa[Ridx][Ridx];
                Dr_DRfa.block<3, 3>(3, 0) = Dthetadotf_DRf*DXf_DXfa[Ridx][Ridx] + JrInvthetaf*DXf_DXfa[Oidx][Ridx];
                Dr_DRfa = sqrtW*Dr_DRfa;

                // cout << "Dr_DRfa\n" << Dr_DRfa << endl;
            }
            
            // dr_dOfa
            idx = OfaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Dr_DOfa(jacobians[idx]);
                Dr_DOfa.setZero();
                Dr_DOfa.block<3, 3>(0, 0) = Dthetaf_DRf*DXf_DXfa[Ridx][Oidx];
                Dr_DOfa.block<3, 3>(3, 0) = Dthetadotf_DRf*DXf_DXfa[Ridx][Oidx] + JrInvthetaf*DXf_DXfa[Oidx][Oidx];
                Dr_DOfa = sqrtW*Dr_DOfa;

                // cout << "Dr_DOfa\n" << Dr_DOfa << endl;
            }

            // dr_dRfb
            idx = RfbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> Dr_DRfb(jacobians[idx]);
                Dr_DRfb.setZero();
                Dr_DRfb.block<3, 3>(0, 0) = Dthetaf_DRf*DXf_DXfb[Ridx][Ridx];
                Dr_DRfb.block<3, 3>(3, 0) = Dthetadotf_DRf*DXf_DXfb[Ridx][Ridx] + JrInvthetaf*DXf_DXfb[Oidx][Ridx];
                Dr_DRfb = sqrtW*Dr_DRfb;

                // cout << "Dr_DRfb\n" << Dr_DRfb << endl;
            }

            // dr_dOfb
            idx = OfbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Dr_DOfb(jacobians[idx]);
                Dr_DOfb.setZero();
                Dr_DOfb.block<3, 3>(0, 0) = Dthetaf_DRf*DXf_DXfb[Ridx][Oidx];
                Dr_DOfb.block<3, 3>(3, 0) = Dthetadotf_DRf*DXf_DXfb[Ridx][Oidx] + JrInvthetaf*DXf_DXfb[Oidx][Oidx];
                Dr_DOfb = sqrtW*Dr_DOfb;

                // cout << "Dr_DOfb\n" << Dr_DOfb << endl;
            }
        }

        // Jacobians on sa PVA states
        {
            idx = PsaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Dr_DPsa(jacobians[idx]);
                Dr_DPsa.setZero();
                Dr_DPsa.block<3, 3>(6,  0) = -(DXs_DXsa[Pidx][Pidx] + dtsf*DXs_DXsa[Vidx][Pidx] + dtsf*dtsf/2*DXs_DXsa[Aidx][Pidx]);
                Dr_DPsa.block<3, 3>(9,  0) = -(DXs_DXsa[Vidx][Pidx] + dtsf*DXs_DXsa[Aidx][Pidx]);
                Dr_DPsa.block<3, 3>(12, 0) = -(DXs_DXsa[Aidx][Pidx]);
                Dr_DPsa = sqrtW*Dr_DPsa;

                // cout << "Dr_DPsa\n" << Dr_DPsa << endl;
            }
    
            idx = VsaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Dr_DVsa(jacobians[idx]);
                Dr_DVsa.setZero();
                Dr_DVsa.block<3, 3>(6,  0) = -(DXs_DXsa[Pidx][Vidx] + dtsf*DXs_DXsa[Vidx][Vidx] + dtsf*dtsf/2*DXs_DXsa[Aidx][Vidx]);
                Dr_DVsa.block<3, 3>(9,  0) = -(DXs_DXsa[Vidx][Vidx] + dtsf*DXs_DXsa[Aidx][Vidx]);
                Dr_DVsa.block<3, 3>(12, 0) = -(DXs_DXsa[Aidx][Vidx]);
                Dr_DVsa = sqrtW*Dr_DVsa;

                // cout << "Dr_DVsa\n" << Dr_DVsa << endl;
            }
    
            idx = AsaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Dr_DAsa(jacobians[idx]);
                Dr_DAsa.setZero();
                Dr_DAsa.block<3, 3>(6,  0) = -(DXs_DXsa[Pidx][Aidx] + dtsf*DXs_DXsa[Vidx][Aidx] + dtsf*dtsf/2*DXs_DXsa[Aidx][Aidx]);
                Dr_DAsa.block<3, 3>(9,  0) = -(DXs_DXsa[Vidx][Aidx] + dtsf*DXs_DXsa[Aidx][Aidx]);
                Dr_DAsa.block<3, 3>(12, 0) = -(DXs_DXsa[Aidx][Aidx]);
                Dr_DAsa = sqrtW*Dr_DAsa;

                // cout << "Dr_DAsa\n" << Dr_DAsa << endl;
            }
        }

        // Jacobians on sb PVA states
        {
            idx = PsbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Dr_DPsb(jacobians[idx]);
                Dr_DPsb.setZero();
                Dr_DPsb.block<3, 3>(6,  0) = -(DXs_DXsb[Pidx][Pidx] + dtsf*DXs_DXsb[Vidx][Pidx] + dtsf*dtsf/2*DXs_DXsb[Aidx][Pidx]);
                Dr_DPsb.block<3, 3>(9,  0) = -(DXs_DXsb[Vidx][Pidx] + dtsf*DXs_DXsb[Aidx][Pidx]);
                Dr_DPsb.block<3, 3>(12, 0) = -(DXs_DXsb[Aidx][Pidx]);
                Dr_DPsb = sqrtW*Dr_DPsb;

                // cout << "Dr_DPsb\n" << Dr_DPsb << endl;
            }
    
            idx = VsbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Dr_DVsb(jacobians[idx]);
                Dr_DVsb.setZero();
                Dr_DVsb.block<3, 3>(6,  0) = -(DXs_DXsb[Pidx][Vidx] + dtsf*DXs_DXsb[Vidx][Vidx] + dtsf*dtsf/2*DXs_DXsb[Aidx][Vidx]);
                Dr_DVsb.block<3, 3>(9,  0) = -(DXs_DXsb[Vidx][Vidx] + dtsf*DXs_DXsb[Aidx][Vidx]);
                Dr_DVsb.block<3, 3>(12, 0) = -(DXs_DXsb[Aidx][Vidx]);
                Dr_DVsb = sqrtW*Dr_DVsb;

                // cout << "Dr_DPsb\n" << Dr_DVsb << endl;
            }
    
            idx = AsbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Dr_DAsb(jacobians[idx]);
                Dr_DAsb.setZero();
                Dr_DAsb.block<3, 3>(6,  0) = -(DXs_DXsb[Pidx][Aidx] + dtsf*DXs_DXsb[Vidx][Aidx] + dtsf*dtsf/2*DXs_DXsb[Aidx][Aidx]);
                Dr_DAsb.block<3, 3>(9,  0) = -(DXs_DXsb[Vidx][Aidx] + dtsf*DXs_DXsb[Aidx][Aidx]);
                Dr_DAsb.block<3, 3>(12, 0) = -(DXs_DXsb[Aidx][Aidx]);
                Dr_DAsb = sqrtW*Dr_DAsb;

                // cout << "Dr_DAsb\n" << Dr_DAsb << endl;
            }
        }

        // Jacobians on fa PVA states
        {
            idx = PfaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Dr_DPfa(jacobians[idx]);
                Dr_DPfa.setZero();
                Dr_DPfa.block<3, 3>(6,  0) = DXf_DXfa[Pidx][Pidx];
                Dr_DPfa.block<3, 3>(9,  0) = DXf_DXfa[Vidx][Pidx];
                Dr_DPfa.block<3, 3>(12, 0) = DXf_DXfa[Aidx][Pidx];
                Dr_DPfa = sqrtW*Dr_DPfa;

                // cout << "Dr_DPfa\n" << Dr_DPfa << endl;
            }
    
            idx = VfaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Dr_DVfa(jacobians[idx]);
                Dr_DVfa.setZero();
                Dr_DVfa.block<3, 3>(6,  0) = DXf_DXfa[Pidx][Vidx];
                Dr_DVfa.block<3, 3>(9,  0) = DXf_DXfa[Vidx][Vidx];
                Dr_DVfa.block<3, 3>(12, 0) = DXf_DXfa[Aidx][Vidx];
                Dr_DVfa = sqrtW*Dr_DVfa;

                // cout << "Dr_DVfa\n" << Dr_DVfa << endl;
            }
    
            idx = AfaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Dr_DAfa(jacobians[idx]);
                Dr_DAfa.setZero();
                Dr_DAfa.block<3, 3>(6,  0) = DXf_DXfa[Pidx][Aidx];
                Dr_DAfa.block<3, 3>(9,  0) = DXf_DXfa[Vidx][Aidx];
                Dr_DAfa.block<3, 3>(12, 0) = DXf_DXfa[Aidx][Aidx];
                Dr_DAfa = sqrtW*Dr_DAfa;

                // cout << "Dr_DAfa\n" << Dr_DAfa << endl;
            }
        }

        // Jacobians on fb PVA states
        {
            idx = PfbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Dr_DPfb(jacobians[idx]);
                Dr_DPfb.setZero();
                Dr_DPfb.block<3, 3>(6,  0) = DXf_DXfb[Pidx][Pidx];
                Dr_DPfb.block<3, 3>(9,  0) = DXf_DXfb[Vidx][Pidx];
                Dr_DPfb.block<3, 3>(12, 0) = DXf_DXfb[Aidx][Pidx];
                Dr_DPfb = sqrtW*Dr_DPfb;

                // cout << "Dr_DPfb\n" << Dr_DPfb << endl;
            }
    
            idx = VfbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Dr_DVfb(jacobians[idx]);
                Dr_DVfb.setZero();
                Dr_DVfb.block<3, 3>(6,  0) = DXf_DXfb[Pidx][Vidx];
                Dr_DVfb.block<3, 3>(9,  0) = DXf_DXfb[Vidx][Vidx];
                Dr_DVfb.block<3, 3>(12, 0) = DXf_DXfb[Aidx][Vidx];
                Dr_DVfb = sqrtW*Dr_DVfb;

                // cout << "Dr_DVfb\n" << Dr_DVfb << endl;
            }
    
            idx = AfbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Dr_DAfb(jacobians[idx]);
                Dr_DAfb.setZero();
                Dr_DAfb.block<3, 3>(6,  0) = DXf_DXfb[Pidx][Aidx];
                Dr_DAfb.block<3, 3>(9,  0) = DXf_DXfb[Vidx][Aidx];
                Dr_DAfb.block<3, 3>(12, 0) = DXf_DXfb[Aidx][Aidx];
                Dr_DAfb = sqrtW*Dr_DAfb;

                // cout << "Dr_DAfb\n" << Dr_DAfb << endl;
            }
        }

        return true;
    }

private:

    const int Ridx = 0;
    const int Oidx = 1;
    const int Pidx = 2;
    const int Vidx = 3;
    const int Aidx = 4;

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
