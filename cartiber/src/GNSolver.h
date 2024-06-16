#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>

#include "utility.h"

#include "factor/GPPointToPlaneFactorTMN.hpp"
#include "factor/GPMotionPriorTwoKnotsFactorTMN.hpp"

class GNSolver
{
private:

    // Node handle to get information needed
    ros::NodeHandlePtr nh;

    // Lidar params
    double lidar_weight = 1.0;

    // Weight for the motion prior factors
    double mpSigmaR = 1.0;
    double mpSigmaP = 1.0;

    // Solver parameters
    int max_gniter = 3;
    double lambda = 1.0;
    double dxmax = 0.5;

public:

    // Destructor
   ~GNSolver();
   
    // Constructor
    GNSolver(ros::NodeHandlePtr &nh_);

    void EvaluateMotionPriorFactors
    (
        GaussianProcessPtr &traj,
        VectorXd &r, MatrixXd &J, double* cost
    );

    void EvaluateLidarFactors
    (
        const GaussianProcessPtr &traj,
        const deque<vector<LidarCoef>> &SwLidarCoef,
        VectorXd &r, MatrixXd &J, double* cost
    );

    bool Solve
    (
        GaussianProcessPtr &traj,
        deque<vector<LidarCoef>> &SwLidarCoef,
        const int &iter
    );
};