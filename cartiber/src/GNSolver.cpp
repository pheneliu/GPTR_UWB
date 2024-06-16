#include "GNSolver.h"

// Local size at individual factors
#define RES_LDR_LSIZE 1
#define RES_MP2_LSIZE 15

// Global size for the whole problem, to be changed in each solve
int RES_LDR_GSIZE;
int RES_MP2_GSIZE;
int RES_ALL_GSIZE;

int RES_LDR_GBASE;
int RES_MP2_GBASE;

// Size of each parameter block at individual level
#define XROT_LSIZE 0
#define XOMG_LSIZE 3
#define XPOS_LSIZE 6
#define XVEL_LSIZE 9
#define XACC_LSIZE 12
#define XALL_LSIZE STATE_DIM

// // Size of the parameter blocks on global scale
// int XROT_GSIZE;
// int XOMG_GSIZE;
// int XPOS_GSIZE;
// int XVEL_GSIZE;
// int XACC_GSIZE;
// // Size of the all params
int XALL_GSIZE;

// // Offset for each type of variable
// int XALL_GBASE;
// int XOMG_GBASE;
// int XPOS_GBASE;
// int XVEL_GBASE;
// int XACC_GBASE;

void UpdateDimensions(int &numldr, int &nummp2, int &numKnots)
{
    RES_LDR_GSIZE = RES_LDR_LSIZE*numldr;
    RES_MP2_GSIZE = RES_MP2_LSIZE*nummp2;
    RES_ALL_GSIZE = RES_LDR_GSIZE + RES_MP2_GSIZE;

    RES_LDR_GBASE = 0;
    RES_MP2_GBASE = RES_LDR_GBASE + RES_LDR_GSIZE;

    // XROT_GSIZE = XROT_LSIZE*numKnots;
    // XOMG_GSIZE = XOMG_LSIZE*numKnots;
    // XPOS_GSIZE = XPOS_LSIZE*numKnots;
    // XVEL_GSIZE = XVEL_LSIZE*numKnots;
    // XACC_GSIZE = XACC_LSIZE*numKnots;
    
    XALL_GSIZE = XALL_LSIZE*numKnots;

    // XROT_GBASE = 0;
    // XOMG_GBASE = XROT_GBASE + XROT_GSIZE;
    // XPOS_GBASE = XOMG_GBASE + XOMG_GSIZE;
    // XVEL_GBASE = XPOS_GBASE + XPOS_GSIZE;
    // XACC_GBASE = XVEL_GBASE + XVEL_GSIZE;
}

GNSolver::~GNSolver(){};

GNSolver::GNSolver(ros::NodeHandlePtr &nh_) : nh(nh_)
{
    nh->getParam("/lidar_weight", lidar_weight);

    nh->getParam("/max_gniter", max_gniter);
    nh->getParam("/lambda", lambda);
    nh->getParam("/dxmax", dxmax);
    
    // Weight for the motion prior
    nh->getParam("mpSigmaR", mpSigmaR);
    nh->getParam("mpSigmaP", mpSigmaP);
}

void GNSolver::EvaluateLidarFactors
(
    const GaussianProcessPtr &traj,
    const deque<vector<LidarCoef>> &SwLidarCoef,
    VectorXd &r, MatrixXd &J, double* cost
)
{
    // Index the points
    vector<int> lidarIdxBase(SwLidarCoef.size(), 0);
    for(int widx = 1; widx < SwLidarCoef.size(); widx++)
        lidarIdxBase[widx] += lidarIdxBase[widx-1] + SwLidarCoef[widx-1].size();

    // Create factors and calculate their residual and jacobian
    for(int widx = 0; widx < SwLidarCoef.size(); widx++)
    {
        auto Coefs = SwLidarCoef[widx];

        #pragma omp parallel for num_threads(MAX_THREADS)
        for(int cidx = 0; cidx < Coefs.size(); cidx++)
        {
            LidarCoef &coef = Coefs[cidx];
            ROS_ASSERT(cidx == coef.ptIdx);

            // Skip if lidar coef is not assigned
            if (coef.t < 0)
                continue;

            if (!traj->TimeInInterval(coef.t, 1e-6))
                continue;

            auto   us = traj->computeTimeIndex(coef.t);
            int    u  = us.first;
            double s  = us.second;

            typedef GPPointToPlaneFactorTMN ppFactor;
            ppFactor factor = ppFactor(coef.finW, coef.f, coef.n, coef.plnrty*lidar_weight, traj->getDt(), s);

            // Calculate the residual and jacobian
            factor.Evaluate(traj->getKnot(u), traj->getKnot(u+1));

            int row  = RES_LDR_GBASE + (lidarIdxBase[widx] + cidx)*RES_LDR_LSIZE;
            int cola = u*XALL_LSIZE;
            int colb = cola + XALL_LSIZE;

            r.block(row, 0, RES_LDR_LSIZE, 1).setZero();
            J.block(row, 0, RES_LDR_LSIZE, XALL_GSIZE).setZero();

            r.block<RES_LDR_LSIZE, 1>(row, 0) << factor.residual;
            J.block<RES_LDR_LSIZE, XALL_LSIZE>(row, cola) << factor.jacobian.block<RES_LDR_LSIZE, XALL_LSIZE>(0, 0);
            J.block<RES_LDR_LSIZE, XALL_LSIZE>(row, colb) << factor.jacobian.block<RES_LDR_LSIZE, XALL_LSIZE>(0, XALL_LSIZE);

        }
    }

    // Calculate the cost
    if (cost != NULL)
        *cost = pow(r.block(RES_LDR_GBASE, 0, RES_LDR_GSIZE, 1).norm(), 2);
}

void GNSolver::EvaluateMotionPriorFactors
(
    GaussianProcessPtr &traj,
    VectorXd &r, MatrixXd &J, double* cost
)
{
    // Add GP factors between consecutive knots
    #pragma omp parallel for num_threads(MAX_THREADS)
    for (int kidx = 0; kidx < traj->getNumKnots() - 1; kidx++)
    {
        // Create the factors
        typedef GPMotionPriorTwoKnotsFactorTMN mp2kFactor;
        mp2kFactor factor = mp2kFactor(mpSigmaR, mpSigmaP, traj->getDt());

        // Calculate the residual and jacobian
        factor.Evaluate(traj->getKnot(kidx), traj->getKnot(kidx + 1));

        int row  = RES_MP2_GBASE + kidx*RES_MP2_LSIZE;
        int cola = kidx*XALL_LSIZE;
        int colb = cola + XALL_LSIZE;

        r.block(row, 0, RES_MP2_LSIZE, 1).setZero();
        J.block(row, 0, RES_MP2_LSIZE, XALL_GSIZE).setZero();

        r.block<RES_MP2_LSIZE, 1>(row, 0) << factor.residual;
        J.block<RES_MP2_LSIZE, XALL_LSIZE>(row, cola) << factor.jacobian.block<RES_MP2_LSIZE, XALL_LSIZE>(0, 0);
        J.block<RES_MP2_LSIZE, XALL_LSIZE>(row, colb) << factor.jacobian.block<RES_MP2_LSIZE, XALL_LSIZE>(0, XALL_LSIZE);
    }

    // Calculate the cost
    if (cost != NULL)
        *cost = pow(r.block(RES_MP2_GBASE, 0, RES_MP2_GSIZE, 1).norm(), 2);
}

bool GNSolver::Solve
(
    GaussianProcessPtr &traj,
    deque<vector<LidarCoef>> &SwLidarCoef,
    const int &iter
)
{
    // Find the dimensions of the problem
    int numX = 0;
    int numLidarFactors = 0;
    int numMp2kFactors = 0;

    // Each knot is counted as one state
    numX = traj->getNumKnots();
    // Each coefficient makes one factor
    for(auto &c : SwLidarCoef)
        numLidarFactors += c.size();
    // One mp2k factor between each knot
    numMp2kFactors = traj->getNumKnots() - 1;

    // Determine the dimension of the Gauss-Newton Problem
    UpdateDimensions(numLidarFactors, numMp2kFactors, numX);

    // Create the big Matrices
    VectorXd RESIDUAL(RES_ALL_GSIZE, 1);
    MatrixXd JACOBIAN(RES_ALL_GSIZE, XALL_GSIZE);

    // Evaluate the lidar factors
    double J0ldr = -1;
    EvaluateLidarFactors(traj, SwLidarCoef, RESIDUAL, JACOBIAN, &J0ldr);

    // Evaluate the motion prior factors
    double J0mp2k = -1;
    EvaluateMotionPriorFactors(traj, RESIDUAL, JACOBIAN, &J0mp2k);

    // Build the problem and solve
    SparseMatrix<double> Jsparse = JACOBIAN.sparseView(); Jsparse.makeCompressed();
    SparseMatrix<double> Jtp = Jsparse.transpose();
    SparseMatrix<double> H = Jtp*Jsparse;
    MatrixXd b = -Jtp*RESIDUAL;

    MatrixXd dX = MatrixXd::Zero(XALL_GSIZE, 1);
    bool solver_failed = false;

    // Solving using dense QR
    // dX = S.toDense().colPivHouseholderQr().solve(b);

    // Solve using solver and LM method
    SparseMatrix<double> I(H.cols(), H.cols()); I.setIdentity();
    SparseMatrix<double> S = H + lambda/pow(2, iter)*I;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(S);
    solver.factorize(S);
    solver_failed = solver.info() != Eigen::Success;
    dX = solver.solve(b);

    // If solving is not successful, return false
    if (solver_failed || dX.hasNaN())
    {
        printf(KRED"Failed to solve!\n"RESET);
        // for(int idx = 0; idx < traj.numKnots(); idx++)
        //     printf("XPOSE%02d. %7.3f, %7.3f\n", idx,
        //            dX.block<3, 1>(idx*XSE3_SIZE + XROT_BASE, 0).norm(),
        //            dX.block<3, 1>(idx*XSE3_SIZE + XPOS_BASE, 0).norm());
        // printf("XBIAS%02d. %7.3f, %7.3f\n\n", 0, dX.block<3, 1>(XBIG_GBASE, 0).norm(), dX.block<3, 1>(XBIA_GBASE, 0).norm());
        cout << dX << endl;
        // return false;
    }

    if (dX.norm() > dxmax)
        dX = dX / dX.norm() * dxmax;

    // Update the states
    for(int kidx = 0; kidx < traj->getNumKnots(); kidx++)
        traj->updateKnot(kidx, dX.block<STATE_DIM, 1>(kidx*STATE_DIM, 0));

    // printf("dXsize: %d. NumKnots: %d\n", dX.rows(), traj->getNumKnots());
    // cout << dX << endl;

    return true;
}