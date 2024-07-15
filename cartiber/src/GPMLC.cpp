#include "GPMLC.h"

// Destructor
GPMLC::~GPMLC(){};

// Constructor
GPMLC::GPMLC(ros::NodeHandlePtr &nh_) : nh(nh_), R_Lx_Ly(Quaternd(1, 0, 0, 0)), P_Lx_Ly(Vec3(0, 0, 0)) {};

// Add parameters
void GPMLC::AddTrajParams(ceres::Problem &problem, GaussianProcessPtr &traj, double tmin, double tmax)
{
    ceres::LocalParameterization *so3parameterization = new GPSO3dLocalParameterization();

    for (int kidx = 0; kidx < traj->getNumKnots(); kidx++)
    {
        problem.AddParameterBlock(traj->getKnotSO3(kidx).data(), 4, so3parameterization);
        problem.AddParameterBlock(traj->getKnotOmg(kidx).data(), 3);
        problem.AddParameterBlock(traj->getKnotPos(kidx).data(), 3);
        problem.AddParameterBlock(traj->getKnotVel(kidx).data(), 3);
        problem.AddParameterBlock(traj->getKnotAcc(kidx).data(), 3);

        if (traj->getKnotTime(kidx) <= tmin)
        {
            problem.SetParameterBlockConstant(traj->getKnotSO3(kidx).data());
            problem.SetParameterBlockConstant(traj->getKnotOmg(kidx).data());
            problem.SetParameterBlockConstant(traj->getKnotPos(kidx).data());
            problem.SetParameterBlockConstant(traj->getKnotVel(kidx).data());
            problem.SetParameterBlockConstant(traj->getKnotAcc(kidx).data());
        }

        if (traj->getKnotTime(kidx) >= tmax)
        {
            problem.SetParameterBlockConstant(traj->getKnotSO3(kidx).data());
            problem.SetParameterBlockConstant(traj->getKnotOmg(kidx).data());
            problem.SetParameterBlockConstant(traj->getKnotPos(kidx).data());
            problem.SetParameterBlockConstant(traj->getKnotVel(kidx).data());
            problem.SetParameterBlockConstant(traj->getKnotAcc(kidx).data());
        }
    }
}

void GPMLC::AddMP2kFactors(ceres::Problem &problem, GaussianProcessPtr &traj, vector<ceres::ResidualBlockId> &res_ids, double tmin, double tmax)
{
    // Add the GP factors based on knot difference
    for (int kidx = 0; kidx < traj->getNumKnots() - 1; kidx++)
    {
        if (traj->getKnotTime(kidx+1) <= tmin || traj->getKnotTime(kidx) >= tmax)
            continue;

        vector<double *> factor_param_blocks;
        // Add the parameter blocks
        for (int kidx_ = kidx; kidx_ < kidx + 2; kidx_++)
        {
            factor_param_blocks.push_back(traj->getKnotSO3(kidx_).data());
            factor_param_blocks.push_back(traj->getKnotOmg(kidx_).data());
            factor_param_blocks.push_back(traj->getKnotAlp(kidx_).data());
            factor_param_blocks.push_back(traj->getKnotPos(kidx_).data());
            factor_param_blocks.push_back(traj->getKnotVel(kidx_).data());
            factor_param_blocks.push_back(traj->getKnotAcc(kidx_).data());
        }
        // Create the factors
        double mp_loss_thres = -1;
        // nh_ptr->getParam("mp_loss_thres", mp_loss_thres);
        ceres::LossFunction *mp_loss_function = mp_loss_thres <= 0 ? NULL : new ceres::HuberLoss(mp_loss_thres);
        ceres::CostFunction *cost_function = new GPMotionPriorTwoKnotsFactor(traj->getGPMixerPtr());
        auto res_block = problem.AddResidualBlock(cost_function, mp_loss_function, factor_param_blocks);
        res_ids.push_back(res_block);
    }
}

// Add lidar factors
void GPMLC::AddLidarFactors(ceres::Problem &problem, GaussianProcessPtr &traj, const deque<vector<LidarCoef>> &cloudCoef, vector<ceres::ResidualBlockId> &res_ids)
{
    for (auto &Coef : cloudCoef)
    {
        for (auto &coef : Coef)
        {
            // Skip if lidar coef is not assigned
            if (coef.t < 0)
                continue;
            if (!traj->TimeInInterval(coef.t, 1e-6))
                continue;
            // skip++;
            // if (skip % lidar_ds_rate != 0)
            //     continue;
            
            auto   us = traj->computeTimeIndex(coef.t);
            int    u  = us.first;
            double s  = us.second;
            vector<double *> factor_param_blocks;
            // Add the parameter blocks for rotation
            for (int knot_idx = u; knot_idx < u + 2; knot_idx++)
            {
                factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotAlp(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
            }
            
            double lidar_loss_thres = -1.0;
            ceres::LossFunction *lidar_loss_function = lidar_loss_thres == -1 ? NULL : new ceres::HuberLoss(lidar_loss_thres);
            ceres::CostFunction *cost_function = new GPPointToPlaneFactor(coef.f, coef.n, coef.plnrty, traj->getGPMixerPtr(), s);
            auto res = problem.AddResidualBlock(cost_function, lidar_loss_function, factor_param_blocks);
            res_ids.push_back(res);
        }
    }
}

void GPMLC::AddGPExtrinsicFactors(ceres::Problem &problem, GaussianProcessPtr &trajx, GaussianProcessPtr &trajy, vector<ceres::ResidualBlockId> &res_ids, double tmin, double tmax)
{
    GPMixerPtr gpmx = trajx->getGPMixerPtr();
    GPMixerPtr gpmy = trajy->getGPMixerPtr();

    int Nseg = 1;
    nh->getParam("Nseg", Nseg);

    double t_shift;
    nh->getParam("t_shift", t_shift);

    for (int kidx = 0; kidx < trajx->getNumKnots() - 2; kidx++)
    {
        if (trajx->getKnotTime(kidx+1) <= tmin || trajx->getKnotTime(kidx) >= tmax)
            continue;

        if (trajy->getKnotTime(kidx+1) <= tmin || trajy->getKnotTime(kidx) >= tmax)
            continue;

        for(int i = 0; i < Nseg; i++)
        {
            // Get the knot time
            double t = trajx->getKnotTime(kidx) + trajx->getDt()/Nseg*i + t_shift;

            // Skip if time is outside the range of the other trajectory
            if (!trajy->TimeInInterval(t))
                continue;

            pair<int, double> uss, usf;
            uss = trajx->computeTimeIndex(t);
            usf = trajy->computeTimeIndex(t);

            int umins = uss.first;
            int uminf = usf.first;
            double ss = uss.second;
            double sf = usf.second;

            // Add the parameter blocks
            vector<double *> factor_param_blocks;
            for (int idx = umins; idx < umins + 2; idx++)
            {
                ROS_ASSERT(idx < trajx->getNumKnots());
                // ROS_ASSERT(Util::SO3IsValid(trajx->getKnotSO3(idx)));
                factor_param_blocks.push_back(trajx->getKnotSO3(idx).data());
                factor_param_blocks.push_back(trajx->getKnotOmg(idx).data());
                factor_param_blocks.push_back(trajx->getKnotAlp(idx).data());
                factor_param_blocks.push_back(trajx->getKnotPos(idx).data());
                factor_param_blocks.push_back(trajx->getKnotVel(idx).data());
                factor_param_blocks.push_back(trajx->getKnotAcc(idx).data());
            }
            for (int idx = uminf; idx < uminf + 2; idx++)
            {
                ROS_ASSERT(idx < trajy->getNumKnots());
                // ROS_ASSERT(Util::SO3IsValid(trajy->getKnotSO3(idx)));
                factor_param_blocks.push_back(trajy->getKnotSO3(idx).data());
                factor_param_blocks.push_back(trajy->getKnotOmg(idx).data());
                factor_param_blocks.push_back(trajy->getKnotAlp(idx).data());
                factor_param_blocks.push_back(trajy->getKnotPos(idx).data());
                factor_param_blocks.push_back(trajy->getKnotVel(idx).data());
                factor_param_blocks.push_back(trajy->getKnotAcc(idx).data());
            }
            factor_param_blocks.push_back(R_Lx_Ly.data());
            factor_param_blocks.push_back(P_Lx_Ly.data());

            // Create the factors
            MatrixXd InvCov = (trajx->getKnotCov(umins) + trajy->getKnotCov(uminf))/1e6;
            // double mpSigmaR = 1.0;
            // double mpSigmaP = 1.0;
            double mp_loss_thres = -1;
            // nh_ptr->getParam("mp_loss_thres", mp_loss_thres);
            ceres::LossFunction *mp_loss_function = mp_loss_thres <= 0 ? NULL : new ceres::HuberLoss(mp_loss_thres);
            ceres::CostFunction *cost_function = new GPExtrinsicFactor(InvCov, gpmx, gpmy, ss, sf);
            auto res_block = problem.AddResidualBlock(cost_function, mp_loss_function, factor_param_blocks);
            res_ids.push_back(res_block);
        }
    }
}

// Prototype
void GPMLC::Evaluate(GaussianProcessPtr &traj0,
                     GaussianProcessPtr &traji,
                     double tmin, double tmax,
                     const deque<vector<LidarCoef>> &cloudCoef0,
                     const deque<vector<LidarCoef>> &cloudCoefi,
                     myTf<double> &T_B_Li_gndtr)
{
    // Ceres problem
    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

    // Set up the ceres problem
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = MAX_THREADS;
    options.max_num_iterations = 50;

    // Add the parameter blocks for rotation
    AddTrajParams(problem, traj0, tmin, tmax);
    AddTrajParams(problem, traji, tmin, tmax);
    // Add the extrinsic params
    problem.AddParameterBlock(R_Lx_Ly.data(), 4, new GPSO3dLocalParameterization());
    problem.AddParameterBlock(P_Lx_Ly.data(), 3);

    // Add the motion prior factor
    vector<ceres::ResidualBlockId> res_ids_mp2k;
    double cost_mp2k_init = -1; double cost_mp2k_final = -1;
    AddMP2kFactors(problem, traj0, res_ids_mp2k, tmin, tmax);
    AddMP2kFactors(problem, traji, res_ids_mp2k, tmin, tmax);

    // Add the lidar factors
    vector<ceres::ResidualBlockId> res_ids_lidar;
    double cost_lidar_init = -1; double cost_lidar_final = -1;
    AddLidarFactors(problem, traj0, cloudCoef0, res_ids_lidar);
    AddLidarFactors(problem, traji, cloudCoefi, res_ids_lidar);

    // Add the extrinsics factors
    vector<ceres::ResidualBlockId> res_ids_gpx;
    double cost_gpx_init = -1; double cost_gpx_final = -1;
    AddGPExtrinsicFactors(problem, traj0, traji, res_ids_gpx, tmin, tmax);
    // AddGPExtrinsicFactors(problem, traji, traj0, res_ids_gpx);

    // Find the initial cost
    Util::ComputeCeresCost(res_ids_mp2k,  cost_mp2k_init,  problem);
    Util::ComputeCeresCost(res_ids_gpx,   cost_gpx_init,   problem);
    Util::ComputeCeresCost(res_ids_lidar, cost_lidar_init, problem);

    TicToc tt_slv;

    ceres::Solve(options, &problem, &summary);
    
    tt_slv.Toc();

    Util::ComputeCeresCost(res_ids_mp2k,  cost_mp2k_final,  problem);
    Util::ComputeCeresCost(res_ids_gpx,   cost_gpx_final,   problem);
    Util::ComputeCeresCost(res_ids_lidar, cost_lidar_final, problem);

    myTf<double> T_L0_Li(R_Lx_Ly.unit_quaternion(), P_Lx_Ly);
    myTf T_err = T_B_Li_gndtr.inverse()*T_L0_Li;
    
    printf(KGRN
           "GPX Opt: Iter: %d. Time: %.0f.\n"
           "Factor: MP2K: %d, Cross: %d. Ldr: %d.\n"
           "J0: %12.3f. MP2k: %9.3f. Xtrs: %9.3f. LDR: %9.3f.\n"
           "Jk: %12.3f. MP2k: %9.3f. Xtrs: %9.3f. LDR: %9.3f.\n"
           "T_L0_Li. XYZ: %7.3f, %7.3f, %7.3f. YPR: %7.3f, %7.3f, %7.3f. Error: %f\n\n"
           RESET,
           summary.iterations.size(), tt_slv.GetLastStop(),
           res_ids_mp2k.size(), res_ids_gpx.size(), res_ids_lidar.size(),
           summary.initial_cost, cost_mp2k_init, cost_gpx_init, cost_lidar_init,
           summary.final_cost, cost_mp2k_final, cost_gpx_final, cost_lidar_final,
           T_L0_Li.pos.x(), T_L0_Li.pos.y(), T_L0_Li.pos.z(),
           T_L0_Li.yaw(), T_L0_Li.pitch(), T_L0_Li.roll(), T_err.pos.norm());

}