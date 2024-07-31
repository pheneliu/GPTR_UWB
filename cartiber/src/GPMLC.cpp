#include "GPMLC.h"

// Destructor
GPMLC::~GPMLC(){};

// Constructor
GPMLC::GPMLC(ros::NodeHandlePtr &nh_) : nh(nh_), R_Lx_Ly(Quaternd(1, 0, 0, 0)), P_Lx_Ly(Vec3(0, 0, 0)) {};

// Add parameters
void GPMLC::AddTrajParams(ceres::Problem &problem, GaussianProcessPtr &traj, double tmin, double tmax, double tmid, vector<int> &kpidx)
{
    ceres::LocalParameterization *so3parameterization = new GPSO3dLocalParameterization();

    auto usmin = traj->computeTimeIndex(tmin);
    auto usmax = traj->computeTimeIndex(tmax);

    int kidxmin = usmin.first;
    int kidxmax = usmax.first+1;

    // ROS_ASSERT(kidxmax <= traj->getNumKnots()-1);

    for (int kidx = 0; kidx < traj->getNumKnots(); kidx++)
    {
        if (kidx < kidxmin || kidx > kidxmax)
            continue;

        problem.AddParameterBlock(traj->getKnotSO3(kidx).data(), 4, so3parameterization);
        problem.AddParameterBlock(traj->getKnotOmg(kidx).data(), 3);
        problem.AddParameterBlock(traj->getKnotAlp(kidx).data(), 3);
        problem.AddParameterBlock(traj->getKnotPos(kidx).data(), 3);
        problem.AddParameterBlock(traj->getKnotVel(kidx).data(), 3);
        problem.AddParameterBlock(traj->getKnotAcc(kidx).data(), 3);

        // Log down the index of knot
        kpidx.push_back(kidx);

        if (kidx == kidxmin)
        {
            problem.SetParameterBlockConstant(traj->getKnotSO3(kidxmin).data());
            // problem.SetParameterBlockConstant(traj->getKnotOmg(kidxmin).data());
            // problem.SetParameterBlockConstant(traj->getKnotAlp(kidxmin).data());
            problem.SetParameterBlockConstant(traj->getKnotPos(kidxmin).data());
            // problem.SetParameterBlockConstant(traj->getKnotVel(kidxmin).data());
            // problem.SetParameterBlockConstant(traj->getKnotAcc(kidxmin).data());
        }

        if (kidx == kidxmax)
        {
            problem.SetParameterBlockConstant(traj->getKnotSO3(kidxmax).data());
            // problem.SetParameterBlockConstant(traj->getKnotOmg(kidxmax).data());
            // problem.SetParameterBlockConstant(traj->getKnotAlp(kidxmin).data());
            problem.SetParameterBlockConstant(traj->getKnotPos(kidxmax).data());
            // problem.SetParameterBlockConstant(traj->getKnotVel(kidxmax).data());
            // problem.SetParameterBlockConstant(traj->getKnotAcc(kidxmax).data());
        }
    }
}

void GPMLC::AddMP2KFactors(ceres::Problem &problem, GaussianProcessPtr &traj, FactorMeta &factorMeta, double tmin, double tmax)
{
    // Add the GP factors based on knot difference
    for (int kidx = 0; kidx < traj->getNumKnots() - 1; kidx++)
    {
        if (traj->getKnotTime(kidx + 1) <= tmin || traj->getKnotTime(kidx) >= tmax)
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
        
        // Record the residual block
        factorMeta.res.push_back(res_block);
        
        // Record the knot indices
        factorMeta.kidx.push_back(vector<int>({kidx, kidx + 1}));
        
        // Record the time stamp of the factor
        factorMeta.stamp.push_back(traj->getKnotTime(kidx+1));
    }
}

// Add lidar factors
void GPMLC::AddLidarFactors(ceres::Problem &problem, GaussianProcessPtr &traj, const deque<vector<LidarCoef>> &cloudCoef,
                            FactorMeta &factorMeta, double tmin, double tmax)
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

            if (traj->getKnotTime(u) <= tmin || traj->getKnotTime(u+1) >= tmax)
                continue;

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

            // Record the residual block
            factorMeta.res.push_back(res);

            // Record the knot indices
            factorMeta.kidx.push_back({u, u + 1});

            // Record the time stamp of the factor
            factorMeta.stamp.push_back(coef.t);
        }
    }
}

void GPMLC::AddGPExtrinsicFactors(ceres::Problem &problem, GaussianProcessPtr &trajx, GaussianProcessPtr &trajy,
                                  FactorMeta &factorMeta, double tmin, double tmax)
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
        {
            // printf("Skipping %f. %f, %f, %f\n", trajx->getKnotTime(kidx+1), tmin, trajx->getKnotTime(kidx), tmax);
            continue;
        }

        if (trajy->getKnotTime(kidx+1) <= tmin || trajy->getKnotTime(kidx) >= tmax)
        {
            // printf("Skipping %f. %f, %f, %f\n", trajy->getKnotTime(kidx+1), tmin, trajy->getKnotTime(kidx), tmax);
            continue;
        }

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
            
            // Save the residual block
            factorMeta.res.push_back(res_block);

            // Save the knot indices
            factorMeta.kidx.push_back({umins, umins + 1, uminf, uminf + 1});

            // Save the traj indices
            factorMeta.tidx.push_back({0, 0, 1, 1});

            // Record the time stamp of the factor
            factorMeta.stamp.push_back(t);
        }
    }
}

void GPMLC::AddPriorFactor(ceres::Problem &problem, GaussianProcessPtr &traj0, GaussianProcessPtr &traji,
                           FactorMeta &factorMeta, double tmin, double tmax)
{
    MarginalizationFactor* margFactor = new MarginalizationFactor(&margInfo);

    // Add the involved parameters blocks
    problem.AddResidualBlock(margFactor, NULL, margInfo.getAllParamBlocks());
}

void GPMLC::Marginalize(ceres::Problem &problem, GaussianProcessPtr &traj0, GaussianProcessPtr &traji,
                        double tmin, double tmax, double tmid,
                        FactorMeta &factorMetaMp2k, FactorMeta &factorMetaLidar, FactorMeta &factorMetaGpx,
                        const deque<vector<LidarCoef>> &cloudCoef0,
                        const deque<vector<LidarCoef>> &cloudCoefi)
{
    auto us0 = traj0->computeTimeIndex(tmid);
    auto usi = traji->computeTimeIndex(tmid);

    // Find the states that are kept and removed after marginalization
    vector<pair<int, int>> tk_kept;
    if (us0.second == 0)
        tk_kept.push_back(make_pair(0, us0.first));
    else
    {
        tk_kept.push_back(make_pair(0, us0.first));
        tk_kept.push_back(make_pair(0, us0.first+1));
    }
    if (usi.second == 0)
        tk_kept.push_back(make_pair(1, usi.first));
    else
    {
        tk_kept.push_back(make_pair(1, usi.first));
        tk_kept.push_back(make_pair(1, usi.first+1));
    }

    // ROS_ASSERT_MSG(traj0->computeTimeIndex(tmid).second == 0, "%f", traj0->computeTimeIndex(tmid).second);
    // ROS_ASSERT_MSG(traji->computeTimeIndex(tmid).second == 0, "%f", traji->computeTimeIndex(tmid).second);

    vector<pair<int, int>> tk_removed_res;

    // Deskew, Transform and Associate
    auto FindRemovedFactors = [&tk_removed_res, &tmid](FactorMeta &factorMeta, FactorMeta &factorMetaRemoved) -> void
    {
        for(int ridx = 0; ridx < factorMeta.res.size(); ridx++)
        {
            ceres::ResidualBlockId &res = factorMeta.res[ridx];
            int KC = factorMeta.kidx[ridx].size();
            bool removed = factorMeta.stamp[ridx] < tmid;

            if (removed)
            {
                factorMetaRemoved.res.push_back(res);
                factorMetaRemoved.kidx.push_back(factorMeta.kidx[ridx]);
                factorMetaRemoved.tidx.push_back(factorMeta.tidx[ridx]);
                factorMetaRemoved.stamp.push_back(factorMeta.stamp[ridx]);

                for(int coupling_idx = 0; coupling_idx < KC; coupling_idx++)
                {
                    int kidx = factorMeta.kidx[ridx][coupling_idx];
                    int tidx = factorMeta.tidx[ridx][coupling_idx];
                    tk_removed_res.push_back(make_pair(tidx, kidx));
                    // printf("Removing knot %d of traj %d, param %d.\n", kidx, tidx, tk2p[tk_removed_res.back()]);
                }
            }
        }
    };

    auto GetJacobian = [](ceres::CRSMatrix &J) -> MatrixXd
    {
        MatrixXd eJ(J.num_rows, J.num_cols);
        eJ.setZero();
        for (int r = 0; r < J.num_rows; ++r)
        {
            for (int idx = J.rows[r]; idx < J.rows[r + 1]; ++idx)
            {
                const int c = J.cols[idx];
                eJ(r, c) = J.values[idx];
            }
        }
        return eJ;
    };

    vector<GaussianProcessPtr> trajs = {traj0, traji};

    // Find the MP2k factors that will be removed
    FactorMeta factorMetaMp2kRemoved(2);
    FindRemovedFactors(factorMetaMp2k, factorMetaMp2kRemoved);
    printf("factorMetaMp2k: %d. Removed: %d\n", factorMetaMp2k.size(), factorMetaMp2kRemoved.size());

    // Find the lidar factors that will be removed
    FactorMeta factorMetaLidarRemoved(2);
    FindRemovedFactors(factorMetaLidar, factorMetaLidarRemoved);
    printf("factorMetaLidar: %d. Removed: %d\n", factorMetaLidar.size(), factorMetaLidarRemoved.size());

    // Find the extrinsic factors that will be marginalized
    FactorMeta factorMetaGpxRemoved(4);
    FindRemovedFactors(factorMetaGpx, factorMetaGpxRemoved);
    printf("factorMetaGpx: %d. Removed: %d\n", factorMetaGpx.size(), factorMetaGpxRemoved.size());

    // Identify param index of the kept knots
    vector<GPState<double>> KeptStatesPrior;;
    map<pair<int, int>, int> tkp_kept;
    for(auto &tk : tk_kept)
    {
        ROS_ASSERT(tk2p.find(tk) != tk2p.end());
        tkp_kept[tk] = tk2p[tk];
        KeptStatesPrior.push_back(trajs[tk.first]->getKnot(tk.second));
    }
    // Find param index of removed knots
    map<pair<int, int>, int> tkp_marg;
    for(auto &tk : tk_removed_res)
    {
        ROS_ASSERT(tk2p.find(tk) != tk2p.end());
        if (tkp_kept.find(tk) == tkp_kept.end())
            tkp_marg[tk] = tk2p[tk];
    }

    // Just make sure that all of the column index will increase
    {
        int prev_idx = -1;
        for(auto &tkp : tkp_marg)
        {
            ROS_ASSERT(tkp.second > prev_idx);
            prev_idx = tkp.second;
        }
    }

    // Make all parameter block variables
    std::vector<double*> parameter_blocks;
    problem.GetParameterBlocks(&parameter_blocks);
    for (auto &paramblock : parameter_blocks)
        problem.SetParameterBlockVariable(paramblock);

    // Find the jacobians of factors that will be removed
    ceres::Problem::EvaluateOptions e_option;
    for(auto &res : factorMetaMp2kRemoved.res)
        e_option.residual_blocks.push_back(res);
    for(auto &res : factorMetaLidarRemoved.res)
        e_option.residual_blocks.push_back(res);
    for(auto &res : factorMetaGpxRemoved.res)
        e_option.residual_blocks.push_back(res);
    e_option.num_threads = MAX_THREADS;

    vector<double> residual_;
    ceres::CRSMatrix Jacobian_;
    problem.Evaluate(e_option, NULL, &residual_, NULL, &Jacobian_);
    VectorXd residual = Eigen::Map<VectorXd>(residual_.data(), residual_.size());
    MatrixXd Jacobian = GetJacobian(Jacobian_);

    // Extract all collumns corresponding to the marginalized states
    int RMVD_SIZE = tkp_marg.size()*STATE_DIM;
    int KEPT_SIZE = tkp_kept.size()*STATE_DIM;
    int XTRS_SIZE = 6;

    int RMVD_BASE = 0;
    int KEPT_BASE = RMVD_SIZE;
    int XTRS_BASE = RMVD_SIZE + KEPT_SIZE;

    MatrixXd Jmkx = MatrixXd::Zero(Jacobian.rows(), RMVD_SIZE + KEPT_SIZE + XTRS_SIZE);

    auto CopyColM = [](MatrixXd &Jend, MatrixXd &Jsrc, map<pair<int, int>, int> &tkps, int BASE_END, int XSIZE) -> void
    {
        int cidx = 0;
        for(auto &tkp : tkps)
        {
            int XBASE = tkp.second*XSIZE;
            for (int c = 0; c < XSIZE; c++)
            {
                Jend.col(BASE_END + cidx) << Jsrc.col(XBASE + c);
                Jsrc.col(XBASE + c).setZero();
                printf(KMAG "Copying col %3d of param %3d in marg states. MaxCoef: %f\n" RESET,
                        c, tkp.second, Jend.col(BASE_END + cidx).cwiseAbs().maxCoeff());
                cidx++;
            }
        }
    };

    auto CopyColK = [](MatrixXd &Jend, MatrixXd &Jsrc, map<pair<int, int>, int> &tkps, int BASE_END, int XSIZE) -> void
    {
        int cidx = 0;
        for(auto &tkp : tkps)
        {
            int XBASE = tkp.second*XSIZE;
            for (int c = 0; c < XSIZE; c++)
            {
                Jend.col(BASE_END + cidx) << Jsrc.col(XBASE + c);
                Jsrc.col(XBASE + c).setZero();
                printf(KYEL "Copying col %3d of param %3d in keep states. MaxCoef: %f\n" RESET,
                        c, tkp.second, Jend.col(BASE_END + cidx).cwiseAbs().maxCoeff());
                cidx++;
            }
        }
    };

    // Copy the Jacobians of marginalized states
    CopyColM(Jmkx, Jacobian, tkp_marg, RMVD_BASE, STATE_DIM);

    // Copy the Jacobians of kept states
    CopyColK(Jmkx, Jacobian, tkp_kept, KEPT_BASE, STATE_DIM);

    // Copy the Jacobians of the extrinsic states
    Jmkx.rightCols(XTRS_SIZE) = Jacobian.rightCols(XTRS_SIZE);
    Jacobian.rightCols(XTRS_SIZE).setZero();

    // Calculate the Hessian
    typedef SparseMatrix<double> SMd;
    SMd r = residual.sparseView(); r.makeCompressed();
    SMd J = Jmkx.sparseView(); J.makeCompressed();
    SMd H = J.transpose()*J;
    SMd b = J.transpose()*r;

    // Divide the Hessian into corner blocks
    int MARG_SIZE = RMVD_SIZE;
    int KEEP_SIZE = KEPT_SIZE + XTRS_SIZE;

    SMd Hmm = H.block(0, 0, MARG_SIZE, MARG_SIZE);
    SMd Hmk = H.block(0, MARG_SIZE, MARG_SIZE, KEEP_SIZE);
    SMd Hkm = H.block(MARG_SIZE, 0, KEEP_SIZE, MARG_SIZE);
    SMd Hkk = H.block(MARG_SIZE, MARG_SIZE, KEEP_SIZE, KEEP_SIZE);

    SMd bm = b.block(0, 0, MARG_SIZE, 1);
    SMd bk = b.block(MARG_SIZE, 0, KEEP_SIZE, 1);

    // Create the Schur Complement
    MatrixXd Hmminv = Hmm.toDense().inverse();
    MatrixXd HkmHmminv = Hkm*Hmminv;
    MatrixXd Hkeep = Hkk - HkmHmminv*Hmk;
    MatrixXd bkeep = bk  - HkmHmminv*bm;

    MatrixXd Jkeep; VectorXd rkeep;
    margInfo.HbToJr(Hkeep, bkeep, Jkeep, rkeep);

    printf("Jacobian %d x %d. Jmkx: %d x %d. Params: %d.\n"
           "Jkeep: %d x %d. rkeep: %d x %d. Keep size: %d. Hb max: %f, %f\n",
            Jacobian.rows(), Jacobian.cols(), Jmkx.rows(), Jmkx.cols(), tk2p.size(),
            Jkeep.rows(), Jkeep.cols(), rkeep.rows(), rkeep.cols(), KEEP_SIZE,
            Hkeep.cwiseAbs().maxCoeff(), bkeep.cwiseAbs().maxCoeff());
    
    // Show the marginalization matrices
    // cout << "Jkeep\n" << Hkeep << endl;
    // cout << "rkeep\n" << bkeep << endl;

    // Making sanity checks
    map<ceres::ResidualBlockId, int> wierdRes;
    for(auto &tkp : tk2p)
    {
        int tidx = tkp.first.first;
        int kidx = tkp.first.second;
        int pidx = tkp.second;
        MatrixXd Jparam = Jacobian.block(0, pidx*STATE_DIM, Jacobian.rows(), STATE_DIM);
        // printf("KnotParam: %2d. Traj %2d, Knot %2d. Max: %f\n",
        //         pidx, tkp.first.first, tkp.first.second, Jparam.cwiseAbs().maxCoeff());

        if(Jparam.cwiseAbs().maxCoeff() != 0)
        {
            vector<ceres::ResidualBlockId> resBlocks;
            problem.GetResidualBlocksForParameterBlock(trajs[tidx]->getKnotSO3(kidx).data(), &resBlocks);
            // printf("Found %2d res blocks for param %2d. Knot %2d. Traj %2d.\n", resBlocks.size(), pidx, kidx, tidx);

            for(auto &res : resBlocks)
                wierdRes[res] = 1;
        }
    }
    cout << endl;

    // Check to see if removed res are among the wierd res
    for(auto &res : factorMetaMp2kRemoved.res)
        ROS_ASSERT(wierdRes.find(res) == wierdRes.end());
    // printf("Wierd res: %d. No overlap with mp2k\n");

    for(auto &res : factorMetaLidarRemoved.res)
        ROS_ASSERT(wierdRes.find(res) == wierdRes.end());
    // printf("Wierd res: %d. No overlap with lidar\n");

    for(auto &res : factorMetaGpxRemoved.res)
        ROS_ASSERT(wierdRes.find(res) == wierdRes.end());
    // printf("Wierd res: %d. No overlap with Gpx\n", wierdRes.size());

    // Save the marginalization factors and states
    margInfo.Hkeep = Hkeep;
    margInfo.bkeep = bkeep;

    // Add the param of control points
    margInfo.param_block.clear();
    margInfo.param_prior.clear();
    margInfo.param_block_type.clear();
    margInfo.tk_idx.clear();
    for(auto &tk : tkp_kept)
    {
        int tidx = tk.first.first;
        int kidx = tk.first.second;
        GaussianProcessPtr &traj = trajs[tidx];
        margInfo.param_block.push_back({traj->getKnotSO3(kidx).data(), traj->getKnotOmg(kidx).data(), traj->getKnotAlp(kidx).data(),
                                        traj->getKnotPos(kidx).data(), traj->getKnotVel(kidx).data(), traj->getKnotAcc(kidx).data()});
        margInfo.param_prior.push_back({margInfo.SO3ToDouble(traj->getKnotSO3(kidx)),
                                        margInfo.RV3ToDouble(traj->getKnotOmg(kidx)),
                                        margInfo.RV3ToDouble(traj->getKnotAlp(kidx)),
                                        margInfo.RV3ToDouble(traj->getKnotPos(kidx)),
                                        margInfo.RV3ToDouble(traj->getKnotVel(kidx)),
                                        margInfo.RV3ToDouble(traj->getKnotAcc(kidx))});
        margInfo.param_block_type.push_back({ParamType::SO3, ParamType::RV3, ParamType::RV3, ParamType::RV3, ParamType::RV3, ParamType::RV3});
        margInfo.tk_idx.push_back(tk.first);
    }
    // Add the extrinsic pose
    margInfo.param_block.push_back({R_Lx_Ly.data(), P_Lx_Ly.data()});
    // Add the prior
    margInfo.param_prior.push_back({margInfo.SO3ToDouble(R_Lx_Ly), margInfo.RV3ToDouble(P_Lx_Ly)});
    // Add the param type
    margInfo.param_block_type.push_back({ParamType::SO3, ParamType::RV3});
}

// Prototype
void GPMLC::Evaluate(int iter, GaussianProcessPtr &traj0, GaussianProcessPtr &traji,
                     double tmin, double tmax, double tmid,
                     const deque<vector<LidarCoef>> &cloudCoef0,
                     const deque<vector<LidarCoef>> &cloudCoefi,
                     myTf<double> &T_B_Li_gndtr)
{
    TicToc tt_build;

    // Ceres problem
    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

    // Set up the ceres problem
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = MAX_THREADS;
    options.max_num_iterations = 50;

    // Storing the knot indices added to the problem as param
    vector<vector<int>> tklist(2);

    // Add the parameter blocks for rotation
    AddTrajParams(problem, traj0, tmin, tmax, tmid, tklist[0]);
    AddTrajParams(problem, traji, tmin, tmax, tmid, tklist[1]);
    
    // Add the extrinsic params
    problem.AddParameterBlock(R_Lx_Ly.data(), 4, new GPSO3dLocalParameterization());
    problem.AddParameterBlock(P_Lx_Ly.data(), 3);

    // Make the map between traj-kidx with param index
    tk2p.clear();
    for (int tidx = 0; tidx < tklist.size(); tidx++)
        for(auto kidx : tklist[tidx])
            tk2p[make_pair(tidx, kidx)] = tk2p.size();

    // Add the motion prior factor
    FactorMeta factorMetaMp2k(2);
    double cost_mp2k_init = -1, cost_mp2k_final = -1;
    AddMP2KFactors(problem, traj0, factorMetaMp2k, tmin, tmax); factorMetaMp2k.ResizeTidx(0);
    AddMP2KFactors(problem, traji, factorMetaMp2k, tmin, tmax); factorMetaMp2k.ResizeTidx(1);

    // Add the lidar factors
    FactorMeta factorMetaLidar(2);
    double cost_lidar_init = -1; double cost_lidar_final = -1;
    AddLidarFactors(problem, traj0, cloudCoef0, factorMetaLidar, tmin, tmax); factorMetaLidar.ResizeTidx(0);
    AddLidarFactors(problem, traji, cloudCoefi, factorMetaLidar, tmin, tmax); factorMetaLidar.ResizeTidx(1);

    // Add the extrinsics factors
    FactorMeta factorMetaGpx(4);
    double cost_gpx_init = -1; double cost_gpx_final = -1;
    AddGPExtrinsicFactors(problem, traj0, traji, factorMetaGpx, tmin, tmax);

    // Add the prior factor
    // FactorMeta factorMetaPrior(4);
    // double cost_prior_init = -1; double cost_prior_final = -1;
    // AddPriorFactor(problem, traj0, traji, factorMetaPrior, tmin, tmax);

    tt_build.Toc();

    TicToc tt_slv;

    // Find the initial cost
    Util::ComputeCeresCost(factorMetaMp2k.res,  cost_mp2k_init,  problem);
    Util::ComputeCeresCost(factorMetaLidar.res, cost_lidar_init, problem);
    Util::ComputeCeresCost(factorMetaGpx.res,   cost_gpx_init,   problem);

    ceres::Solve(options, &problem, &summary);

    Util::ComputeCeresCost(factorMetaMp2k.res,  cost_mp2k_final,  problem);
    Util::ComputeCeresCost(factorMetaLidar.res, cost_lidar_final, problem);
    Util::ComputeCeresCost(factorMetaGpx.res,   cost_gpx_final,   problem);

    // Determine the factors to remove
    Marginalize(problem, traj0, traji, tmin, tmax, tmid, factorMetaMp2k, factorMetaLidar, factorMetaGpx, cloudCoef0, cloudCoefi);

    tt_slv.Toc();

    myTf<double> T_L0_Li(R_Lx_Ly.unit_quaternion(), P_Lx_Ly);
    myTf T_err_1 = T_B_Li_gndtr.inverse()*T_L0_Li;
    myTf T_err_2 = T_L0_Li.inverse()*T_B_Li_gndtr;
    MatrixXd T_err(6, 1); T_err << T_err_1.pos, T_err_2.pos;

    printf(KGRN
           "GPX Opt #%d: Iter: %d. Tbd: %.0f. Tslv: %.0f. Tmin-Tmax: %.3f, %.3f\n"
           "Factor: MP2K: %d, Cross: %d. Ldr: %d.\n"
           "J0: %12.3f. MP2k: %9.3f. Xtrs: %9.3f. LDR: %9.3f.\n"
           "Jk: %12.3f. MP2k: %9.3f. Xtrs: %9.3f. LDR: %9.3f.\n"
           "T_L0_Li. XYZ: %7.3f, %7.3f, %7.3f. YPR: %7.3f, %7.3f, %7.3f. Error: %.3f, %.3f, %.3f\n\n"
           RESET,
           iter,
           summary.iterations.size(), tt_build.GetLastStop(), tt_slv.GetLastStop(), tmin, tmax,
           factorMetaMp2k.size(), factorMetaGpx.size(), factorMetaLidar.size(),
           summary.initial_cost, cost_mp2k_init, cost_gpx_init, cost_lidar_init,
           summary.final_cost, cost_mp2k_final, cost_gpx_final, cost_lidar_final,
           T_L0_Li.pos.x(), T_L0_Li.pos.y(), T_L0_Li.pos.z(),
           T_L0_Li.yaw(), T_L0_Li.pitch(), T_L0_Li.roll(), T_err_1.pos.norm(), T_err_2.pos.norm(), T_err.norm());

}