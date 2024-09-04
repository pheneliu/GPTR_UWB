#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <ceres/ceres.h>

#include "utility.h"

// All about gaussian process
#include "GaussianProcess.hpp"

// Factors
#include "factor/GPExtrinsicFactor.h"
#include "factor/GPPointToPlaneFactor.h"
#include "factor/GPMotionPriorTwoKnotsFactor.h"

enum class ParamType
{
    SO3, RV3, UNKNOWN
};

enum class ParamRole
{
    GPSTATE, EXTRINSIC, UNKNOWN
};

class ParamInfo
{
public:
    double* address = NULL; // Actual address of the param block
    ParamType type;         // Type of the param block (SO3 or RV3)
    ParamRole role;         // What this param is used for state or extrinsics
    // int param_size;         // Size of the param block
    int pidx;               // Index of the parameter in the problem
    int tidx;               // Index of the trajectory
    int kidx;               // Index of the knot
    int sidx;               // Index of state in the knot

    int param_size;         // Size of the param in doubles
    int delta_size;         // Size of the dx in the "X \boxplus dx" of the param

    ParamInfo()
    {
        address = NULL;
        type = ParamType::UNKNOWN;
        role = ParamRole::UNKNOWN;
        // param_size = -1;
        pidx = -1;
        tidx = -1;
        kidx = -1;
        sidx = -1;
    }

    ParamInfo(double* address_, ParamType type_, ParamRole role_,
              int pidx_, int tidx_, int kidx_, int sidx_)
        : address(address_), type(type_), role(role_),
          pidx(pidx_), tidx(tidx_), kidx(kidx_), sidx(sidx_)
    {
        if(type == ParamType::SO3)
        {
            param_size = 4;
            delta_size = 3;
        }
        else if(type == ParamType::RV3)
        {
            param_size = 3;
            delta_size = 3;
        }
        else
        {
            printf("Unknown type! %d\n", type);
            exit(-1);
        }
    };

    bool operator<(const ParamInfo &other) const
    {
        if (tidx == -1 && other.tidx != -1)
            return false;

        if (tidx != -1 && other.tidx == -1)
            return true;

        if ((tidx != -1 && other.tidx != -1) && (tidx < other.tidx))
            return true;

        if ((tidx != -1 && other.tidx != -1) && (tidx > other.tidx))
            return false;

        // Including the situation that two knots are 01
        if (tidx == other.tidx)
        {
            if (kidx == -1 && other.kidx != -1)
                return false;

            if (kidx != -1 && other.kidx == -1)
                return true;

            if ((kidx != -1 && other.kidx != -1) && (kidx < other.kidx))
                return true;

            if ((kidx != -1 && other.kidx != -1) && (kidx > other.kidx))
                return false;

            if (kidx == other.kidx)
            {
                if (sidx == -1 && other.sidx != -1)
                    return false;

                if (sidx != -1 && other.sidx == -1)
                    return true;

                if ((sidx != -1 && other.sidx != -1) && (sidx < other.sidx))
                    return true;

                if ((sidx != -1 && other.sidx != -1) && (sidx > other.sidx))
                    return false;

                return false;    
            }    
        }                
    }
};

class FactorMeta
{
public:

    vector<double> stamp; // Time of the factor
    vector<ceres::ResidualBlockId> res;
    vector<vector<ParamInfo>> coupled_params;

    FactorMeta() {};
    FactorMeta(const FactorMeta &other)
        : res(other.res), coupled_params(other.coupled_params), stamp(other.stamp)
    {};

    // FactorMeta(int knots_coupled_)
    //     : knots_coupled(knots_coupled_)
    // {};

    // void ResizeTidx(int idx)
    // {
    //     tidx.resize(kidx.size(), vector<int>(knots_coupled, idx));
    // }

    FactorMeta operator+(const FactorMeta other)
    {
        FactorMeta added(*this);
        
        added.stamp.insert(added.stamp.end(), other.stamp.begin(), other.stamp.end());
        added.res.insert(added.res.end(), other.res.begin(), other.res.end());
        added.coupled_params.insert(added.coupled_params.end(), other.coupled_params.begin(), other.coupled_params.end());

        return added;
    }

    int size()
    {
        return res.size();
    }
};

class MarginalizationInfo
{
public:
    MatrixXd Hkeep;
    VectorXd bkeep;
    MatrixXd Jkeep;
    VectorXd rkeep;

    vector<ParamInfo> keptParamInfo;
    map<double*, vector<double>> keptParamPrior;

    MarginalizationInfo() {};

    template<typename T>
    vector<T> SO3ToDouble(Sophus::SO3<T> &rot)
    {
        Quaternion<T> q = rot.unit_quaternion();
        return {q.x(), q.y(), q.z(), q.w()};
    }

    template<typename T>
    vector<T> RV3ToDouble(Matrix<T, 3, 1> &vec)
    {
        return {vec.x(), vec.y(), vec.z()};
    }

    template<typename T>
    Sophus::SO3<T> DoubleToSO3(vector<T> &rot)
    {
        return Sophus::SO3<T>(Quaternion<T>(rot[3], rot[0], rot[1], rot[2]));
    }

    template<typename T>
    Matrix<T, 3, 1> DoubleToRV3(vector<T> &vec)
    {
        return Matrix<T, 3, 1>(vec[0], vec[1], vec[2]);
    }

    template<typename T>
    Sophus::SO3<T> DoubleToSO3(const double* rot)
    {
        return Sophus::SO3<T>(Quaternion<T>(rot[3], rot[0], rot[1], rot[2]));
    }

    template<typename T>
    Matrix<T, 3, 1> DoubleToRV3(const double* vec)
    {
        return Matrix<T, 3, 1>(vec[0], vec[1], vec[2]);
    }

    vector<double*> getAllParamBlocks()
    {
        vector<double*> params;
        for(auto &kpi : keptParamInfo)
            params.push_back(kpi.address);
        return params;    
    }

    // Convert the H, b matrices to J, r matrices
    void HbToJr(const MatrixXd &H, const VectorXd &b, MatrixXd &J, VectorXd &r)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(H);

        Eigen::VectorXd S = Eigen::VectorXd((saes.eigenvalues().array() > 0).select(saes.eigenvalues().array(), 0));
        Eigen::VectorXd S_inv = Eigen::VectorXd((saes.eigenvalues().array() > 0).select(saes.eigenvalues().array().inverse(), 0));
        
        Eigen::VectorXd S_sqrt = S.cwiseSqrt();
        Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

        J = S_sqrt.asDiagonal() * saes.eigenvectors().transpose();
        r = S_inv_sqrt.asDiagonal() * saes.eigenvectors().transpose() * b;
    }
};

typedef std::shared_ptr<MarginalizationInfo> MarginalizationInfoPtr;

class MarginalizationFactor : public ceres::CostFunction
{
private:
        MarginalizationInfoPtr margInfo;
        map<const double*, ParamInfo> keptParamMap;

public:

    MarginalizationFactor(MarginalizationInfoPtr margInfo_, map<double*, ParamInfo> &paramInfoMap)
    {
        margInfo = margInfo_;
        keptParamMap.clear();

        int res_size = 0;

        // Set the parameter blocks sizes
        for(auto &param : margInfo->keptParamInfo)
        {
            // Confirm that the param is in the new map
            // ROS_ASSERT(paramInfoMap.find(param.address) != paramInfoMap.end());

            if (param.type == ParamType::SO3)
                mutable_parameter_block_sizes()->push_back(4);
            else
                mutable_parameter_block_sizes()->push_back(3);

            res_size += param.delta_size;

            keptParamMap[param.address] = param;
        }

        // Set the residual sizes
        set_num_residuals(res_size);
    }

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        vector<Vector3d> rprior_;
        vector<Matrix3d> Jprior_;
        vector<ParamType> type;
        vector<int> RES_BASE;

        int PRIOR_SIZE = 0;

        // Iterate over groups of param blocks (9 for GP state)
        for(int idx = 0; idx < margInfo->keptParamInfo.size(); idx++)
        {
            ParamInfo &param = margInfo->keptParamInfo[idx];
            // ROS_ASSERT(keptParamMap.find(parameters[idx]) != keptParamMap.end());

            // Find the residual
            if (param.type == ParamType::SO3)
            {
                SO3d xso3_est = margInfo->DoubleToSO3<double>(parameters[idx]);
                SO3d xso3_pri = margInfo->DoubleToSO3<double>(margInfo->keptParamPrior[param.address]);
                Vector3d res = (xso3_pri.inverse()*xso3_est).log();
                rprior_.push_back(res);
                Jprior_.push_back(GPMixer::JrInv(res));
                type.push_back(ParamType::SO3);

                RES_BASE.push_back(PRIOR_SIZE);
                PRIOR_SIZE += param.delta_size;
            }
            else if (param.type == ParamType::RV3)
            {
                Vector3d xr3_est = margInfo->DoubleToRV3<double>(parameters[idx]);
                Vector3d xr3_pri = margInfo->DoubleToRV3<double>(margInfo->keptParamPrior[param.address]);
                Vector3d res = xr3_est - xr3_pri;
                rprior_.push_back(res);
                Jprior_.push_back(Matrix3d::Identity());
                type.push_back(ParamType::RV3);

                RES_BASE.push_back(PRIOR_SIZE);
                PRIOR_SIZE += param.delta_size;
            }
            else
            {
                yolos("Unknown param type! %d\n", param.type);
                exit(-1);
            }
        }

        int Nprior = rprior_.size();
        VectorXd rprior = VectorXd::Zero(PRIOR_SIZE, 1);
        MatrixXd Jprior = MatrixXd::Zero(PRIOR_SIZE, PRIOR_SIZE);
        for(int idx = 0; idx < Nprior; idx++)
        {
            rprior.block<3, 1>(RES_BASE[idx], 0) = rprior_[idx];
            Jprior.block<3, 3>(RES_BASE[idx], RES_BASE[idx]) = Jprior_[idx];
        }

        VectorXd &rkeep = margInfo->rkeep;
        MatrixXd &Jkeep = margInfo->Jkeep;

        const MatrixXd &bkeep = margInfo->bkeep;
        const MatrixXd &Hkeep = margInfo->Hkeep;
        MatrixXd bmarg = bkeep + Hkeep*Jprior*rprior;
        MatrixXd Hmarg = Jprior.transpose()*Hkeep*Jprior;
        
        VectorXd rmarg(PRIOR_SIZE, 1);
        MatrixXd Jmarg(PRIOR_SIZE, PRIOR_SIZE);
        // margInfo->HbToJr(Hmarg, bmarg, Jmarg, rmarg);
        rmarg = rkeep + Jkeep*rprior;
        Jmarg = Jkeep;

        // Export the residual
        Eigen::Map<VectorXd>(residuals, bkeep.rows()) = rmarg;

        // (*iteration)++;
        // printf("Iter: %d. rkeep: %.3f. rmarg: %.3f. Jkeep: %.3f. Jmarg: %.3f. Dif: %.3f. %.3f\n",
        //         (*iteration),
        //         rkeep.cwiseAbs().maxCoeff(), rmarg.cwiseAbs().maxCoeff(),
        //         Jkeep.cwiseAbs().maxCoeff(), Jmarg.cwiseAbs().maxCoeff(),
        //         (rkeep - rmarg).cwiseAbs().maxCoeff(),
        //         (Jkeep - Jmarg).cwiseAbs().maxCoeff());
        //         // cout << rkeep - rmarg << endl;

        // if (*iteration == 1 && (rmarg.hasNaN() || Jmarg.hasNaN()))
        // {
        //     printf("Marg has NaN\n");
        //     cout << "rmarg\n" << endl;
        //     cout << rmarg << endl;
        //     cout << "Jmarg\n" << endl;
        //     cout << Jmarg << endl;
        // }

        // Export the Jacobian
        if(jacobians)
        {
            for(int pidx = 0; pidx < Nprior; pidx++)
            {
                ParamInfo &param = margInfo->keptParamInfo[pidx];
                if(jacobians[pidx])
                {
                    Eigen::Map<Matrix<double, -1, -1, Eigen::RowMajor>> J(jacobians[pidx], rprior.rows(), param.param_size);
                    J.setZero();
                    J.leftCols(3) = Jmarg.middleCols(pidx*param.delta_size, param.delta_size);

                    MatrixXd Jtmp = MatrixXd::Zero(rprior.rows(), param.param_size);
                    Jtmp.leftCols(3) = J.leftCols(3);

                    // printf("Jprior %3d: %9.3f, %9.3f\n", pidx, Jtmp.cwiseAbs().maxCoeff(), Jkeep.middleCols(pidx*param.delta_size, param.delta_size).cwiseAbs().maxCoeff());
                    // cout << rprior << endl;
                }
            }
        }

        return true;
    }
};

class GPMLC
{
private:

    // Node handle to get information needed
    ros::NodeHandlePtr nh;

    int Nlidar;

    vector<SO3d> R_Lx_Ly;
    vector<Vec3> P_Lx_Ly;

protected:

    double fix_time_begin = -1;
    double fix_time_end = -1;

    double lidar_weight = 1.0;    
    double mp_loss_thres = -1.0;

    int max_lidarcoefs = 4000;

    deque<int> kidx_marg;
    deque<int> kidx_keep;

    // Map of traj-kidx and parameter id
    map<pair<int, int>, int> tk2p;
    map<double*, ParamInfo> paramInfoMap;
    MarginalizationInfoPtr margInfo;
    // MarginalizationFactor* margFactor = NULL;

public:

    // Destructor
   ~GPMLC();
   
    // Constructor
    GPMLC(ros::NodeHandlePtr &nh_, int Nlidar_);

    void AddTrajParams(
        ceres::Problem &problem, vector<GaussianProcessPtr> &trajs, int &tidx,
        map<double*, ParamInfo> &paramInfo, double tmin, double tmax, double tmid);

    void AddMP2KFactors(
        ceres::Problem &problem, GaussianProcessPtr &traj,
        map<double*, ParamInfo> &paramInfo, FactorMeta &factorMeta,
        double tmin, double tmax);

    void AddLidarFactors(
        ceres::Problem &problem, GaussianProcessPtr &traj,
        int ds_rate,
        map<double*, ParamInfo> &paramInfo, FactorMeta &factorMeta,
        const deque<vector<LidarCoef>> &cloudCoef,
        double tmin, double tmax);

    void AddGPExtrinsicFactors(
        ceres::Problem &problem, GaussianProcessPtr &trajx, GaussianProcessPtr &trajy, SO3d &R_Lx_Ly, Vec3 &P_Lx_Ly,
        map<double*, ParamInfo> &paramInfo, FactorMeta &factorMeta,
        double tmin, double tmax);

    void AddPriorFactor(
        ceres::Problem &problem, vector<GaussianProcessPtr> &trajs,
        FactorMeta &factorMeta, double tmin, double tmax);

    void Marginalize(
        ceres::Problem &problem, vector<GaussianProcessPtr> &trajs,
        double tmin, double tmax, double tmid,
        map<double*, ParamInfo> &paramInfo,
        FactorMeta &factorMetaMp2k, FactorMeta &factorMetaLidar, FactorMeta &factorMetaGpx, FactorMeta &factorMetaPrior);

    void Evaluate(
        int inner_iter, int outer_iter, vector<GaussianProcessPtr> &trajs,
        double tmin, double tmax, double tmid,
        const vector<deque<vector<LidarCoef>>> &cloudCoef,
        bool do_marginalization,
        Matrix<double, STATE_DIM, 1> &dX,
        vector<myTf<double>> &T_B_Li_gndtr);

    SE3d GetExtrinsics(int lidx);

    void Reset();  
};

typedef std::shared_ptr<GPMLC> GPMLCPtr;
