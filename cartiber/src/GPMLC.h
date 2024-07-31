#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <ceres/ceres.h>

#include "utility.h"

// /* All needed for filter of custom point type----------*/
// #include <pcl/pcl_base.h>
// #include <pcl/impl/pcl_base.hpp>
// #include <pcl/filters/filter.h>
// #include <pcl/filters/impl/filter.hpp>
// #include <pcl/filters/uniform_sampling.h>
// #include <pcl/filters/impl/uniform_sampling.hpp>
// #include <pcl/filters/impl/voxel_grid.hpp>
// #include <pcl/filters/crop_box.h>
// #include <pcl/filters/impl/crop_box.hpp>
// /* All needed for filter of custom point type----------*/

// All about gaussian process
#include "GaussianProcess.hpp"

// Custom solver
// #include "GNSolver.h"

// #include "factor/ExtrinsicFactor.h"
// #include "factor/FullExtrinsicFactor.h"
#include "factor/GPExtrinsicFactor.h"
#include "factor/GPPointToPlaneFactor.h"
#include "factor/GPMotionPriorTwoKnotsFactor.h"
// #include "factor/GPMotionPriorTwoKnotsFactorTMN.hpp"

struct TKPIDX
{
    TKPIDX(int tidx_, int kidx_, int pidx_)
        : tidx(tidx_), kidx(kidx_), pidx(pidx_)
    {};

    int tidx; // trajectory index
    int kidx; // knot index 
    int pidx; // parameter index
};

struct FactorMeta
{
    int knots_coupled = 2;
    vector<ceres::ResidualBlockId> res;
    vector<vector<int>> kidx;
    vector<vector<int>> tidx;
    vector<double> stamp;     // Time of the sample

    FactorMeta()
        : knots_coupled(2)
    {};

    FactorMeta(int knots_coupled_)
        : knots_coupled(knots_coupled_)
    {};

    void ResizeTidx(int idx)
    {
        tidx.resize(kidx.size(), vector<int>(knots_coupled, idx));
    }

    int size()
    {
        return res.size();
    }
};

enum class ParamType
{
    SO3, RV3
};

struct MarginalizationInfo
{
    MatrixXd Hkeep;
    VectorXd bkeep;

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
    Sophus::SO3<T> DoubleToSO3(double* &rot)
    {
        return Sophus::SO3<T>(Quaternion<T>(rot[3], rot[0], rot[1], rot[2]));
    }

    template<typename T>
    Matrix<T, 3, 1> DoubleToRV3(double* &vec)
    {
        return Matrix<T, 3, 1>(vec[0], vec[1], vec[2]);
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
    };

    vector<double*> getAllParamBlocks()
    {
        vector<double*> all_param_blocks;
        for(auto &pbg : param_block)
            for(auto &pb : pbg)
                all_param_blocks.push_back(pb);

        return all_param_blocks;
    }
    
    vector<vector<double*>> param_block;
    vector<vector<vector<double>>> param_prior;
    vector<vector<ParamType>> param_block_type;
    // vector<SE3d> xtrs_prior;
    vector<pair<int, int>> tk_idx;

    int so3_states = 0;
    int rv3_states = 0;
    vector<ParamType> type;
};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* margInfo_)
    {
        margInfo = margInfo_;

        int res_size = 0;

        // Set the parameter blocks sizes
        for(int gidx = 0; gidx < margInfo->param_block.size(); gidx++)
        {
            auto pbt_group = margInfo->param_block_type[gidx];
            for (auto &type : pbt_group)
            {
                margInfo->type.push_back(type);

                if (type == ParamType::SO3)
                {
                    mutable_parameter_block_sizes()->push_back(4);
                    margInfo->so3_states++;
                }                    
                else
                {
                    mutable_parameter_block_sizes()->push_back(3);
                    margInfo->rv3_states++;
                }

                res_size += 3;
            }
        }

        // Set the residual sizes
        set_num_residuals(res_size);
    }

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        vector<Vector3d> rprior_;
        vector<Matrix3d> Jprior_;
        vector<ParamType> type;

        // Iterate over groups of param blocks (9 for GP state)
        for(int gidx = 0; gidx < margInfo->param_block.size(); gidx++)
        {
            int Npg = margInfo->param_block[gidx].size();
            
            // Iterate over param blocks
            for(int pidx = 0; pidx < Npg; pidx++)
            {
                // Find the residual
                if (margInfo->param_block_type[gidx][pidx] == ParamType::SO3)
                {
                    SO3d xso3_est = margInfo->DoubleToSO3<double>(margInfo->param_block[gidx][pidx]);
                    SO3d xso3_pri = margInfo->DoubleToSO3<double>(margInfo->param_prior[gidx][pidx]);
                    Vector3d res = (xso3_pri.inverse()*xso3_est).log();
                    rprior_.push_back(res);
                    Jprior_.push_back(GPMixer::JrInv(res));
                    type.push_back(ParamType::SO3);
                }
                else
                {
                    Vector3d xr3_est = margInfo->DoubleToRV3<double>(margInfo->param_block[gidx][pidx]);
                    Vector3d xr3_pri = margInfo->DoubleToRV3<double>(margInfo->param_prior[gidx][pidx]);
                    Vector3d res = xr3_est - xr3_pri;
                    rprior_.push_back(res);
                    Jprior_.push_back(GPMixer::JrInv(res));
                    type.push_back(ParamType::RV3);
                }
            }
        }

        int Nprior = rprior_.size();
        VectorXd rprior(Nprior*3, 1);
        MatrixXd Jprior(Nprior*3, Nprior*3);
        for(int idx = 0; idx < Nprior; idx++)
        {
            rprior.block<3, 1>(idx*3, 0) = rprior_[idx];
            Jprior.block<3, 3>(idx*3, idx*3) = Jprior_[idx];
        }

        const MatrixXd &bkeep = margInfo->bkeep;
        const MatrixXd &Hkeep = margInfo->Hkeep;
        MatrixXd bprior = bkeep - Hkeep*Jprior*rprior;
        MatrixXd Hprior = Jprior.transpose()*Hkeep*Jprior;
        margInfo->HbToJr(Hprior, bprior, Jprior, rprior);

        // Export the residual
        Eigen::Map<VectorXd>(residuals, Nprior*3) = rprior;

        // Export the Jacobian
        if(jacobians)
        {
            for(int pidx = 0; pidx < Nprior; pidx++)
            {
                if(jacobians[pidx])
                {
                    if(type[pidx] == ParamType::SO3)
                    {
                        Eigen::Map<Matrix<double, -1, -1, Eigen::RowMajor>> J(jacobians[pidx], rprior.rows(), 4);
                        J.setZero();
                        J.leftCols(3) = Jprior.middleCols(pidx*3, 3);
                    }
                    else if(type[pidx] == ParamType::RV3)
                    {
                        Eigen::Map<Matrix<double, -1, -1, Eigen::RowMajor>> J(jacobians[pidx], rprior.rows(), 3);
                        J.setZero();
                        J.leftCols(3) = Jprior.middleCols(pidx*3, 3);
                    }
                }
            }
        }

        return true;
    }

    MarginalizationInfo* margInfo;
};

class GPMLC
{
private:

    // Node handle to get information needed
    ros::NodeHandlePtr nh;

    SO3d R_Lx_Ly;
    Vec3 P_Lx_Ly;

    deque<int> kidx_marg;
    deque<int> kidx_keep;

    // Map of traj-kidx and parameter id
    map<pair<int, int>, int> tk2p;
    MarginalizationInfo margInfo;

public:

    // Destructor
   ~GPMLC();
   
    // Constructor
    GPMLC(ros::NodeHandlePtr &nh_);

    void AddTrajParams(ceres::Problem &problem, GaussianProcessPtr &traj, double tmin, double tmax, double tmid, vector<int> &kpidx);
    void AddMP2KFactors(ceres::Problem &problem, GaussianProcessPtr &traj, FactorMeta &factorMeta, double tmin, double tmax);
    void AddLidarFactors(ceres::Problem &problem, GaussianProcessPtr &traj, const deque<vector<LidarCoef>> &cloudCoef, FactorMeta &factorMeta, double tmin, double tmax);
    void AddGPExtrinsicFactors(ceres::Problem &problem, GaussianProcessPtr &trajx, GaussianProcessPtr &trajy, FactorMeta &factorMeta, double tmin, double tmax);
    void AddPriorFactor(ceres::Problem &problem, GaussianProcessPtr &traj0, GaussianProcessPtr &traji, FactorMeta &factorMeta, double tmin, double tmax);

    void Marginalize(ceres::Problem &problem, GaussianProcessPtr &traj0, GaussianProcessPtr &traji,
                     double tmin, double tmax, double tmid,
                     FactorMeta &factorMetaMp2k, FactorMeta &factorMetaLidar, FactorMeta &factorMetaGpx,
                     const deque<vector<LidarCoef>> &cloudCoef0,
                     const deque<vector<LidarCoef>> &cloudCoefi);

    void Evaluate(int iter, GaussianProcessPtr &traj0, GaussianProcessPtr &traji,
                  double tmin, double tmax, double tmid,
                  const deque<vector<LidarCoef>> &cloudCoef0,
                  const deque<vector<LidarCoef>> &cloudCoefi,
                  myTf<double> &T_B_Li_gndtr);
};

typedef std::shared_ptr<GPMLC> GPMLCPtr;
