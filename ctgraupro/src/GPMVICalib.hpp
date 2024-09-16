#pragma once

#include "GPMUI.hpp"

class GPMVICalib : public GPMUI
{

public:

    // Destructor
   ~GPMVICalib() {};
   
    // Constructor
    GPMVICalib(ros::NodeHandlePtr &nh_) : GPMUI(nh_)
    {

    }

    void AddProjFactors(
        ceres::Problem &problem, GaussianProcessPtr &traj, 
        map<double*, ParamInfo> &paramInfo, FactorMeta &factorMeta,
        const vector<CornerData> &corner_data_cam0, std::map<int, Eigen::Vector3d> &corner_pos_3d, 
        double tmin, double tmax, double w_tdoa, double tdoa_loss_thres)
    {

        auto usmin = traj->computeTimeIndex(tmin);
        auto usmax = traj->computeTimeIndex(tmax);

        int kidxmin = usmin.first;
        int kidxmax = usmax.first+1;        
        for (auto &tdoa : corner_data_cam0)
        {
            if (!traj->TimeInInterval(tdoa.t, 1e-6)) {
                continue;
            }
            
            auto   us = traj->computeTimeIndex(tdoa.t);
            int    u  = us.first;
            double s  = us.second;

            if (u < kidxmin || u+1 > kidxmax) {
                continue;
            }

            // Eigen::Vector3d pos_an_A = pos_anchors[tdoa.idA];
            // Eigen::Vector3d pos_an_B = pos_anchors[tdoa.idB];          

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
            // ceres::LossFunction *tdoa_loss_function = tdoa_loss_thres == -1 ? NULL : new ceres::HuberLoss(tdoa_loss_thres);
            // ceres::CostFunction *cost_function = new GPTDOAFactor(tdoa.data, pos_an_A, pos_an_B, P_I_tag, w_tdoa, traj->getGPMixerPtr(), s);
            // auto res = problem.AddResidualBlock(cost_function, tdoa_loss_function, factor_param_blocks);
            // Record the residual block
            // factorMeta.res.push_back(res);

            // Record the time stamp of the factor
            // factorMeta.stamp.push_back(tdoa.t);
        }
    }    

    void Evaluate(GaussianProcessPtr &traj, Vector3d &XBIG, Vector3d &XBIA,
                  double tmin, double tmax, double tmid,
                  const vector<CornerData> &corner_data_cam0, const vector<CornerData> &corner_data_cam1, const vector<IMUData> &imuData,
                  std::map<int, Eigen::Vector3d> &corner_pos_3d, 
                  bool do_marginalization, double w_corner, double wGyro, double wAcce, double wBiasGyro, double wBiasAcce, double corner_loss_thres, double mp_loss_thres)
    {
        static int cnt = 0;
        TicToc tt_build;
        cnt++;

        // Ceres problem
        ceres::Problem problem;
        ceres::Solver::Options options;
        ceres::Solver::Summary summary;

        // Set up the ceres problem
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.num_threads = MAX_THREADS;
        options.max_num_iterations = 100;
        options.check_gradients = false;
        
        options.gradient_check_relative_precision = 0.02;  

        // Documenting the parameter blocks
        paramInfoMap.clear();
        // Add the parameter blocks
        {
            // Add the parameter blocks for rotation
            AddTrajParams(problem, traj, 0, paramInfoMap, tmin, tmax, tmid);
            problem.AddParameterBlock(XBIG.data(), 3);
            problem.AddParameterBlock(XBIA.data(), 3);

            paramInfoMap.insert(make_pair(XBIG.data(), ParamInfo(XBIG.data(), ParamType::RV3, ParamRole::EXTRINSIC, paramInfoMap.size(), -1, -1, 1)));
            paramInfoMap.insert(make_pair(XBIA.data(), ParamInfo(XBIA.data(), ParamType::RV3, ParamRole::EXTRINSIC, paramInfoMap.size(), -1, -1, 1)));

            // Sanity check
            for(auto &param_ : paramInfoMap)
            {
                ParamInfo param = param_.second;

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
        AddMP2KFactorsUI(problem, traj, paramInfoMap, factorMetaMp2k, tmin, tmax, mp_loss_thres);

        // Add the projection factors
        FactorMeta factorMetaProj;
        double cost_proj_init = -1; double cost_proj_final = -1;
        AddProjFactors(problem, traj, paramInfoMap, factorMetaProj, corner_data_cam0, corner_pos_3d, tmin, tmax, w_corner, corner_loss_thres);

        FactorMeta factorMetaIMU;
        double cost_imu_init = -1; double cost_imu_final = -1;
        AddIMUFactors(problem, traj, XBIG, XBIA, paramInfoMap, factorMetaIMU, imuData, tmin, tmax, wGyro, wAcce, wBiasGyro, wBiasAcce);

        // Add the prior factor
        FactorMeta factorMetaPrior;
        // double cost_prior_init = -1; double cost_prior_final = -1;
        // if (margInfo != NULL) {
        //     AddPriorFactor(problem, traj, factorMetaPrior, tmin, tmax);
        // }
            
        tt_build.Toc();

        TicToc tt_slv;

        // Find the initial cost
        Util::ComputeCeresCost(factorMetaMp2k.res,  cost_mp2k_init,  problem);
        Util::ComputeCeresCost(factorMetaProj.res, cost_proj_init, problem);
        Util::ComputeCeresCost(factorMetaIMU.res,   cost_imu_init,   problem);
        // Util::ComputeCeresCost(factorMetaPrior.res, cost_prior_init, problem);

        ceres::Solve(options, &problem, &summary);

        // std::cout << summary.FullReport() << std::endl;

        Util::ComputeCeresCost(factorMetaMp2k.res,  cost_mp2k_final,  problem);
        Util::ComputeCeresCost(factorMetaProj.res, cost_proj_final, problem);
        Util::ComputeCeresCost(factorMetaIMU.res,   cost_imu_final,   problem);
        // Util::ComputeCeresCost(factorMetaPrior.res, cost_prior_final, problem);
        
        // Determine the factors to remove
        if (do_marginalization) {
            Marginalize(problem, traj, tmin, tmax, tmid, paramInfoMap, factorMetaMp2k, factorMetaProj, factorMetaIMU, factorMetaPrior);
        }

        tt_slv.Toc();
    }

};

typedef std::shared_ptr<GPMVICalib> GPMVICalibPtr;
