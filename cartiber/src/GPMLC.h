#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>

#include <Eigen/Dense>
#include <Eigen/Sparse>
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

class GPMLC
{
private:

    // Node handle to get information needed
    ros::NodeHandlePtr nh;

    SO3d R_Lx_Ly;
    Vec3 P_Lx_Ly;

public:

    // Destructor
   ~GPMLC();
   
    // Constructor
    GPMLC(ros::NodeHandlePtr &nh_);

    void AddTrajParams(ceres::Problem &problem, GaussianProcessPtr &traj, double tmin, double tmax);
    void AddMP2kFactors(ceres::Problem &problem, GaussianProcessPtr &traj, vector<ceres::ResidualBlockId> &res_ids, double tmin, double tmax);
    void AddLidarFactors(ceres::Problem &problem, GaussianProcessPtr &traj, const deque<vector<LidarCoef>> &cloudCoef, vector<ceres::ResidualBlockId> &res_ids);
    void AddGPExtrinsicFactors(ceres::Problem &problem, GaussianProcessPtr &trajx, GaussianProcessPtr &trajy, vector<ceres::ResidualBlockId> &res_ids, double tmin, double tmax);

    void Evaluate(int iter, GaussianProcessPtr &traj0,
                  GaussianProcessPtr &traji,
                  double tmin, double tmax,
                  const deque<vector<LidarCoef>> &cloudCoef0,
                  const deque<vector<LidarCoef>> &cloudCoefi,
                  myTf<double> &T_B_Li_gndtr);
};

typedef std::shared_ptr<GPMLC> GPMLCPtr;
