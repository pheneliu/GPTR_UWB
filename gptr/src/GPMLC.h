#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <ceres/ceres.h>

#include "utility.h"

// All about gaussian process
#include "GaussianProcess.hpp"

// Utilities to manage params and marginalization
#include "GaussNewtonUtilities.hpp"

// Factors
#include "factor/GPExtrinsicFactor.h"
#include "factor/GPPointToPlaneFactor.h"
#include "factor/GPMotionPriorTwoKnotsFactor.h"

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
    
    int max_ceres_iter = 50;

    double lidar_weight = 1.0;
    double ld_loss_thres = -1.0;
    double xt_loss_thres = -1.0;
    double mp_loss_thres = -1.0;
    
    double max_omg = 20.0;
    double max_alp = 10.0;
    
    double max_vel = 10.0;
    double max_acc = 2.0;

    double xtSigGa = 1.0;
    double xtSigNu = 1.0;

    int max_lidarcoefs = 4000;

    deque<int> kidx_marg;
    deque<int> kidx_keep;

    // Map of traj-kidx and parameter id
    map<pair<int, int>, int> tk2p;
    map<double*, ParamInfo> paramInfoMap;
    MarginalizationInfoPtr margInfo;
    // MarginalizationFactor* margFactor = NULL;

    bool compute_cost = false;
    bool fuse_marg = false;

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
        OptReport &report);

    SE3d GetExtrinsics(int lidx);

    void Reset();  
};

typedef std::shared_ptr<GPMLC> GPMLCPtr;
