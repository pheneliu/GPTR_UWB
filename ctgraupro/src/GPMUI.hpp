#pragma once

#include "GPMLC.h"
#include "factor/GPTDOAFactor.h"
#include "factor/GPMotionPriorTwoKnotsFactorUI.h"

class GPMUI : public GPMLC
{

public:

    // Destructor
   ~GPMUI() {};
   
    // Constructor
    GPMUI(ros::NodeHandlePtr &nh_) : GPMLC(nh_)
    {

    }

    void AddTrajParams(ceres::Problem &problem,
                              GaussianProcessPtr &traj, int tidx,
                              map<double*, ParamInfo> &paramInfoMap,
                              double tmin, double tmax, double tmid)
    {
        // GaussianProcessPtr &traj = trajs[tidx];

        auto usmin = traj->computeTimeIndex(tmin);
        auto usmax = traj->computeTimeIndex(tmax);

        int kidxmin = usmin.first;
        int kidxmax = usmax.first+1;

        // Create local parameterization for so3
        ceres::LocalParameterization *so3parameterization = new GPSO3dLocalParameterization();

        int pidx = -1;

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

            // Log down the information of the params
            paramInfoMap.insert(make_pair(traj->getKnotSO3(kidx).data(), ParamInfo(traj->getKnotSO3(kidx).data(), ParamType::SO3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 0)));
            paramInfoMap.insert(make_pair(traj->getKnotOmg(kidx).data(), ParamInfo(traj->getKnotOmg(kidx).data(), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 1)));
            paramInfoMap.insert(make_pair(traj->getKnotAlp(kidx).data(), ParamInfo(traj->getKnotAlp(kidx).data(), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 2)));
            paramInfoMap.insert(make_pair(traj->getKnotPos(kidx).data(), ParamInfo(traj->getKnotPos(kidx).data(), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 3)));
            paramInfoMap.insert(make_pair(traj->getKnotVel(kidx).data(), ParamInfo(traj->getKnotVel(kidx).data(), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 4)));
            paramInfoMap.insert(make_pair(traj->getKnotAcc(kidx).data(), ParamInfo(traj->getKnotAcc(kidx).data(), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 5)));
            
            if (kidx == kidxmin && fix_kidxmin)
            {
                problem.SetParameterBlockConstant(traj->getKnotSO3(kidxmin).data());
                // problem.SetParameterBlockConstant(traj->getKnotOmg(kidxmin).data());
                // problem.SetParameterBlockConstant(traj->getKnotAlp(kidxmin).data());
                problem.SetParameterBlockConstant(traj->getKnotPos(kidxmin).data());
                // problem.SetParameterBlockConstant(traj->getKnotVel(kidxmin).data());
                // problem.SetParameterBlockConstant(traj->getKnotAcc(kidxmin).data());
            }

            if (kidx == kidxmax && fix_kidxmax)
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

    void AddMP2KFactorsUI(
            ceres::Problem &problem, GaussianProcessPtr &traj,
            map<double*, ParamInfo> &paramInfoMap, FactorMeta &factorMeta,
            double tmin, double tmax)
    {
        // Add the GP factors based on knot difference
        for (int kidx = 0; kidx < traj->getNumKnots() - 1; kidx++)
        {
            if (traj->getKnotTime(kidx + 1) <= tmin || traj->getKnotTime(kidx) >= tmax)
                continue;

            vector<double *> factor_param_blocks;
            factorMeta.coupled_params.push_back(vector<ParamInfo>());
            
            // Add the parameter blocks
            for (int kidx_ = kidx; kidx_ < kidx + 2; kidx_++)
            {
                factor_param_blocks.push_back(traj->getKnotSO3(kidx_).data());
                factor_param_blocks.push_back(traj->getKnotOmg(kidx_).data());
                factor_param_blocks.push_back(traj->getKnotAlp(kidx_).data());
                factor_param_blocks.push_back(traj->getKnotPos(kidx_).data());
                factor_param_blocks.push_back(traj->getKnotVel(kidx_).data());
                factor_param_blocks.push_back(traj->getKnotAcc(kidx_).data());

                // Record the param info
                factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotSO3(kidx_).data()]);
                factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotOmg(kidx_).data()]);
                factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotAlp(kidx_).data()]);
                factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotPos(kidx_).data()]);
                factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotVel(kidx_).data()]);
                factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotAcc(kidx_).data()]);
            }

            // Create the factors
            double mp_loss_thres = -1;
            // nh_ptr->getParam("mp_loss_thres", mp_loss_thres);
            ceres::LossFunction *mp_loss_function = mp_loss_thres <= 0 ? NULL : new ceres::HuberLoss(mp_loss_thres);
            ceres::CostFunction *cost_function = new GPMotionPriorTwoKnotsFactorUI(traj->getGPMixerPtr());
            auto res_block = problem.AddResidualBlock(cost_function, mp_loss_function, factor_param_blocks);
            
            // Record the residual block
            factorMeta.res.push_back(res_block);
            
            // Record the time stamp of the factor
            factorMeta.stamp.push_back(traj->getKnotTime(kidx+1));
        }
    }    

    void AddPriorFactor(ceres::Problem &problem, GaussianProcessPtr &traj, FactorMeta &factorMeta, double tmin, double tmax)
    {
        // Check if kept states are still in the param list
        bool kept_state_present = true;
        vector<int> missing_param_idx;
        int removed_dims = 0;
        for(int idx = 0; idx < margInfo->keptParamInfo.size(); idx++)
        {
            ParamInfo &param = margInfo->keptParamInfo[idx];
            bool state_found = (paramInfoMap.find(param.address) != paramInfoMap.end());
            // printf("param 0x%8x of tidx %2d kidx %4d of sidx %4d is %sfound in paramInfoMap\n",
            //         param.address, param.tidx, param.kidx, param.sidx, state_found ? "" : "NOT ");
            
            if (!state_found)
            {
                kept_state_present = false;
                missing_param_idx.push_back(idx);
                removed_dims += param.delta_size;
            }
        }

        // If all marginalized states are found, add the marginalized factor
        if (kept_state_present)
        {
            MarginalizationFactor* margFactor = new MarginalizationFactor(margInfo, paramInfoMap);

            // Add the involved parameters blocks
            auto res_block = problem.AddResidualBlock(margFactor, NULL, margInfo->getAllParamBlocks());

            // Save the residual block
            factorMeta.res.push_back(res_block);

            // Add the coupled param
            factorMeta.coupled_params.push_back(margInfo->keptParamInfo);

            // Record the time stamp of the factor
            factorMeta.stamp.push_back(tmin);
        }
        else if (missing_param_idx.size() <= margInfo->keptParamInfo.size()) // If some marginalization states are missing, delete the missing states
        {
            // printf("Remove %d params missing from %d keptParamInfos\n", missing_param_idx.size(), margInfo->keptParamInfo.size());
            auto removeElementsByIndices = [](vector<ParamInfo>& vec, const std::vector<int>& indices) -> void
            {
                // Copy indices to a new vector and sort it in descending order
                std::vector<int> sortedIndices(indices);
                std::sort(sortedIndices.rbegin(), sortedIndices.rend());

                // Remove elements based on sorted indices
                for (int index : sortedIndices)
                {
                    if (index >= 0 && index < vec.size())
                        vec.erase(vec.begin() + index);
                    else
                        std::cerr << "Index out of bounds: " << index << std::endl;
                }
            };

            auto removeColOrRow = [](const MatrixXd& matrix, const vector<int>& idxToRemove, int cor) -> MatrixXd // set cor = 0 to remove cols, 1 to remove rows
            {
                MatrixXd matrix_tp = matrix;

                if (cor == 1)
                    matrix_tp.transposeInPlace();

                vector<int> idxToRemove_;
                for(int idx : idxToRemove)
                    if(idx < matrix_tp.cols())
                        idxToRemove_.push_back(idx);

                // for(auto idx : idxToRemove_)
                //     printf("To remove: %d\n", idx);

                // Determine the number of columns to keep
                int idxToKeep = matrix_tp.cols() - idxToRemove_.size();
                if (idxToKeep <= 0)
                    throw std::invalid_argument("All columns (all rows) are removed or invalid number of columns (or rows) to keep");

                // Create a new matrix with the appropriate size
                MatrixXd result(matrix_tp.rows(), idxToKeep);

                // Copy columns that are not in idxToRemove
                int currentCol = 0;
                for (int col = 0; col < matrix_tp.cols(); ++col)
                    if (std::find(idxToRemove_.begin(), idxToRemove_.end(), col) == idxToRemove_.end())
                    {
                        result.col(currentCol) = matrix_tp.col(col);
                        currentCol++;
                    }

                if (cor == 1)
                    result.transposeInPlace();

                return result;
            };

            int cidx = 0;
            vector<int> removed_cidx;
            for(int idx = 0; idx < margInfo->keptParamInfo.size(); idx++)
            {

                int &param_cols = margInfo->keptParamInfo[idx].delta_size;
                if(std::find(missing_param_idx.begin(), missing_param_idx.end(), idx) != missing_param_idx.end())
                    for(int c = 0; c < param_cols; c++)
                    {
                        removed_cidx.push_back(cidx + c);
                        // yolos("%d %d %d\n", cidx, c, removed_cidx.size());
                        // printf("idx: %d. param_cols: %d. cidx: %d. c: %d\n", idx, param_cols, cidx, c);
                    }
                cidx += (margInfo->keptParamInfo[idx].delta_size);
            }

            // Remove the rows and collumns of the marginalization matrices
            margInfo->Hkeep = removeColOrRow(margInfo->Hkeep, removed_cidx, 0);
            margInfo->Hkeep = removeColOrRow(margInfo->Hkeep, removed_cidx, 1);
            margInfo->bkeep = removeColOrRow(margInfo->bkeep, removed_cidx, 1);

            margInfo->Jkeep = removeColOrRow(margInfo->Jkeep, removed_cidx, 0);
            margInfo->Jkeep = removeColOrRow(margInfo->Jkeep, removed_cidx, 1);
            margInfo->rkeep = removeColOrRow(margInfo->rkeep, removed_cidx, 1);

            // printf("Jkeep: %d %d. rkeep: %d %d. Hkeep: %d %d. bkeep: %d %d. ParamPrior: %d. ParamInfo: %d. missing_param_idx: %d\n",
            //         margInfo->Jkeep.rows(), margInfo->Jkeep.cols(),
            //         margInfo->rkeep.rows(), margInfo->rkeep.cols(),
            //         margInfo->Hkeep.rows(), margInfo->Hkeep.cols(),
            //         margInfo->bkeep.rows(), margInfo->bkeep.cols(),
            //         margInfo->keptParamPrior.size(),
            //         margInfo->keptParamInfo.size(),
            //         missing_param_idx.size());

            // Remove the stored priors
            for(auto &param_idx : missing_param_idx)
            {
                // printf("Deleting %d\n", param_idx);
                margInfo->keptParamPrior.erase(margInfo->keptParamInfo[param_idx].address);
            }

            // Remove the unfound states
            removeElementsByIndices(margInfo->keptParamInfo, missing_param_idx);

            // printf("Jkeep: %d %d. rkeep: %d %d. Hkeep: %d %d. bkeep: %d %d. ParamPrior: %d. ParamInfo: %d. missing_param_idx: %d\n",
            //         margInfo->Jkeep.rows(), margInfo->Jkeep.cols(),
            //         margInfo->rkeep.rows(), margInfo->rkeep.cols(),
            //         margInfo->Hkeep.rows(), margInfo->Hkeep.cols(),
            //         margInfo->bkeep.rows(), margInfo->bkeep.cols(),
            //         margInfo->keptParamPrior.size(),
            //         margInfo->keptParamInfo.size(),
            //         missing_param_idx.size());

            // Add the factor
            {
                MarginalizationFactor* margFactor = new MarginalizationFactor(margInfo, paramInfoMap);

                // Add the involved parameters blocks
                auto res_block = problem.AddResidualBlock(margFactor, NULL, margInfo->getAllParamBlocks());

                // Save the residual block
                factorMeta.res.push_back(res_block);

                // Add the coupled param
                factorMeta.coupled_params.push_back(margInfo->keptParamInfo);

                // Record the time stamp of the factor
                factorMeta.stamp.push_back(tmin);
            }
        }
        else
            printf(KYEL "All kept params in marginalization missing. Please check\n" RESET);
    }    

    void AddTDOAFactors(
        ceres::Problem &problem, GaussianProcessPtr &traj,
        map<double*, ParamInfo> &paramInfo, FactorMeta &factorMeta,
        const vector<TDOAData> &tdoaData, std::map<uint16_t, Eigen::Vector3d>& pos_anchors,
        double tmin, double tmax, double w_tdoa)
    {
        for (auto &tdoa : tdoaData)
        {
            // for (auto &coef : Coef)
            // {
                // Skip if lidar coef is not assigned
                // if (coef.t < 0)
                    // continue;
                if (!traj->TimeInInterval(tdoa.t, 1e-6)) {
                    std::cout << "warn: !traj->TimeInInterval(tdoa.t, 1e-6)" << std::endl;
                    continue;
                }
                    
                // skip++;
                // if (skip % lidar_ds_rate != 0)
                //     continue;
                
                auto   us = traj->computeTimeIndex(tdoa.t);
                int    u  = us.first;
                double s  = us.second;

                if (traj->getKnotTime(u) < tmin || traj->getKnotTime(u+1) > tmax) {
                    std::cout << "warn: traj->getKnotTime(u) <= tmin || traj->getKnotTime(u+1) >= tmax tdoa.t: "
                              << tdoa.t << " u: " << u << " traj->getKnotTime(u): " << traj->getKnotTime(u)
                              << " traj->getKnotTime(u+1): " << traj->getKnotTime(u+1) 
                              << " tmin: " << tmin << " tmax: " << tmax << std::endl;
                    continue;
                }

                vector<double *> factor_param_blocks;
                factorMeta.coupled_params.push_back(vector<ParamInfo>());
                // Add the parameter blocks for rotation
                for (int kidx = u; kidx < u + 2; kidx++)
                {
                    factor_param_blocks.push_back(traj->getKnotSO3(kidx).data());
                    factor_param_blocks.push_back(traj->getKnotOmg(kidx).data());
                    factor_param_blocks.push_back(traj->getKnotAlp(kidx).data());
                    factor_param_blocks.push_back(traj->getKnotPos(kidx).data());
                    factor_param_blocks.push_back(traj->getKnotVel(kidx).data());
                    factor_param_blocks.push_back(traj->getKnotAcc(kidx).data());

                    // Record the param info
                    factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotSO3(kidx).data()]);
                    factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotOmg(kidx).data()]);
                    factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotAlp(kidx).data()]);
                    factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotPos(kidx).data()]);
                    factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotVel(kidx).data()]);
                    factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotAcc(kidx).data()]);
                }

                Eigen::Vector3d pos_an_A = pos_anchors[tdoa.idA];
                Eigen::Vector3d pos_an_B = pos_anchors[tdoa.idB];
                
                double tdoa_loss_thres = -1.0;
                ceres::LossFunction *tdoa_loss_function = tdoa_loss_thres == -1 ? NULL : new ceres::HuberLoss(tdoa_loss_thres);
                ceres::CostFunction *cost_function = new GPTDOAFactor(tdoa.data, pos_an_A, pos_an_B, w_tdoa, traj->getGPMixerPtr(), s);
                auto res = problem.AddResidualBlock(cost_function, tdoa_loss_function, factor_param_blocks);

                // Record the residual block
                factorMeta.res.push_back(res);

                // Record the knot indices
                // factorMeta.kidx.push_back({u, u + 1});

                // Record the time stamp of the factor
                factorMeta.stamp.push_back(tdoa.t);
            // }
        }
    }

    void Evaluate(int iter, GaussianProcessPtr &traj,
                  double tmin, double tmax, double tmid,
                  const vector<TDOAData> &tdoaData, std::map<uint16_t, Eigen::Vector3d>& pos_anchors,
                  bool do_marginalization, double w_tdoa)
    {
        TicToc tt_build;

        // Ceres problem
        ceres::Problem problem;
        ceres::Solver::Options options;
        ceres::Solver::Summary summary;

        // Set up the ceres problem
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.num_threads = 1;
        options.max_num_iterations = 50;
        options.check_gradients = false;
        options.gradient_check_relative_precision = 0.02;

        // Documenting the parameter blocks
        paramInfoMap.clear();
        // Add the parameter blocks
        {
            // Add the parameter blocks for rotation
            // for(int tidx = 0; tidx < trajs.size(); tidx++)
            // {
                AddTrajParams(problem, traj, 0, paramInfoMap, tmin, tmax, tmid);
                AddTrajParams(problem, traj, 0, paramInfoMap, tmin, tmax, tmid);
            // }
            // Add the extrinsic params
            // problem.AddParameterBlock(R_Lx_Ly.data(), 4, new GPSO3dLocalParameterization());
            // problem.AddParameterBlock(P_Lx_Ly.data(), 3);
            // paramInfoMap.insert(make_pair(R_Lx_Ly.data(), ParamInfo(R_Lx_Ly.data(), ParamType::SO3, ParamRole::EXTRINSIC, paramInfoMap.size(), -1, -1, 0)));
            // paramInfoMap.insert(make_pair(P_Lx_Ly.data(), ParamInfo(P_Lx_Ly.data(), ParamType::RV3, ParamRole::EXTRINSIC, paramInfoMap.size(), -1, -1, 1)));

            // Sanity check
            for(auto &param_ : paramInfoMap)
            {
                ParamInfo param = param_.second;

                // int tidx = param.tidx;
                int tidx = 0;
                int kidx = param.kidx;
                int sidx = param.sidx;

                if(param.tidx != -1 && param.kidx != -1)
                {
                    switch(sidx)
                    {
                        case 0:
                            ROS_ASSERT(param.address == traj->getKnotSO3(kidx).data());
                            break;
                        case 1:
                            ROS_ASSERT(param.address == traj->getKnotOmg(kidx).data());
                            break;
                        case 2:
                            ROS_ASSERT(param.address == traj->getKnotAlp(kidx).data());
                            break;
                        case 3:
                            ROS_ASSERT(param.address == traj->getKnotPos(kidx).data());
                            break;
                        case 4:
                            ROS_ASSERT(param.address == traj->getKnotVel(kidx).data());
                            break;
                        case 5:
                            ROS_ASSERT(param.address == traj->getKnotAcc(kidx).data());
                            break;
                        default:
                            printf("Unrecognized param block! %d, %d, %d\n", tidx, kidx, sidx);
                            break;
                    }
                }
                else
                {
                    // if(sidx == 0)
                    //     ROS_ASSERT(param.address == R_Lx_Ly.data());
                    // if(sidx == 1)    
                    //     ROS_ASSERT(param.address == P_Lx_Ly.data());
                }
            }
        }

        // Add the motion prior factor
        FactorMeta factorMetaMp2k;
        double cost_mp2k_init = -1, cost_mp2k_final = -1;
        // for(int tidx = 0; tidx < trajs.size(); tidx++)
            AddMP2KFactorsUI(problem, traj, paramInfoMap, factorMetaMp2k, tmin, tmax);

        // Add the TDOA factors
        FactorMeta factorMetaTDOA;
        double cost_tdoa_init = -1; double cost_tdoa_final = -1;
        // for(int tidx = 0; tidx < trajs.size(); tidx++)
            AddTDOAFactors(problem, traj, paramInfoMap, factorMetaTDOA, tdoaData, pos_anchors, tmin, tmax, w_tdoa);

        // Add the extrinsics factors
        // FactorMeta factorMetaGpx;
        // double cost_gpx_init = -1; double cost_gpx_final = -1;
        // for(int tidxx = 0; tidxx < trajs.size(); tidxx++)
        //     for(int tidxy = tidxx+1; tidxy < trajs.size(); tidxy++)
        //         AddGPExtrinsicFactors(problem, trajs[tidxx], trajs[tidxy], paramInfoMap, factorMetaGpx, tmin, tmax);

        // Add the prior factor
        FactorMeta factorMetaPrior;
        double cost_prior_init = -1; double cost_prior_final = -1;
        if (margInfo != NULL)
            AddPriorFactor(problem, traj, factorMetaPrior, tmin, tmax);

        tt_build.Toc();

        TicToc tt_slv;

        // Find the initial cost
        Util::ComputeCeresCost(factorMetaMp2k.res,  cost_mp2k_init,  problem);
        Util::ComputeCeresCost(factorMetaTDOA.res, cost_tdoa_init, problem);
        // Util::ComputeCeresCost(factorMetaGpx.res,   cost_gpx_init,   problem);
        Util::ComputeCeresCost(factorMetaPrior.res, cost_prior_init, problem);

        ceres::Solve(options, &problem, &summary);

        std::cout << summary.FullReport() << std::endl;

        Util::ComputeCeresCost(factorMetaMp2k.res,  cost_mp2k_final,  problem);
        Util::ComputeCeresCost(factorMetaTDOA.res, cost_tdoa_final, problem);
        // Util::ComputeCeresCost(factorMetaGpx.res,   cost_gpx_final,   problem);
        Util::ComputeCeresCost(factorMetaPrior.res, cost_prior_final, problem);

        // Determine the factors to remove
        // if (do_marginalization)
            // Marginalize(problem, trajs, tmin, tmax, tmid, paramInfoMap, factorMetaMp2k, factorMetaTDOA, factorMetaPrior);

        tt_slv.Toc();

    }

};

typedef std::shared_ptr<GPMUI> GPMUIPtr;
