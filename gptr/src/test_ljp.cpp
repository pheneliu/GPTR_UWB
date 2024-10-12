#include "utility.h"
#include "GaussianProcess.hpp"
#include "factor/GPIMUFactorAutodiff.h"
#include "factor/GPMotionPriorTwoKnotsFactorAutodiff.h"


// Set simulation
Eigen::Vector3d G_w(0,0,-9.81);

double Dt = 0.02;
double Scale = 100;
double SimulationStep = 500;
double SimulationTime = SimulationStep * Dt;

GaussianProcessPtr traj_gt(new GaussianProcess(Dt, Vector3d(1, 1, 1).asDiagonal(), Vector3d(1, 1, 1).asDiagonal()));
GaussianProcessPtr traj_estimate(new GaussianProcess(Dt, Vector3d(1, 1, 1).asDiagonal(), Vector3d(1, 1, 1).asDiagonal()));

// IMU measurements
double Dt_imu = 0.01;

double ACC_N = 0.005;
double GYRO_N = 0.0001;
struct IMU{
    double time;
    Eigen::Vector3d acc;
    Eigen::Vector3d gyro;
};
std::vector<IMU> imu_meas;
// assume no IMU bias
Eigen::Vector3d ba(0,0,0);
Eigen::Vector3d bg(0,0,0);


// Image measurements

int LandmarkSize = 100;
std::vector<Eigen::Vector3d> Landmarks;
std::vector<Eigen::Vector3d> Landmarks_estimate;
double Dt_image = 1;
double IMAGE_OBS_N = 0.001;

struct ImageMeas{
    double time;
    int landmarkId;
    Eigen::Vector3d obs; // saved as the bearing vectors
};

std::vector<ImageMeas> image_meas; 

class GPBearingvecFactorAutodiff
{
public:

    // Destructor
    ~GPBearingvecFactorAutodiff() {};

    // Constructor
    GPBearingvecFactorAutodiff(ImageMeas imgmeas_, GPMixerPtr gpm_, double s_, int n_knots_)
    :   imgmeas     (imgmeas_         ), 
        s           (s_               ),
        gpm         (gpm_             ),
        n_knots     (n_knots_         )

    { }

    template <class T>
    bool operator()(T const *const *parameters, T *residuals) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        GPState<T> Xa(0);  gpm->MapParamToState(parameters, RaIdx, Xa);
        GPState<T> Xb(Dt); gpm->MapParamToState(parameters, RbIdx, Xb);
        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the pose at sampling time --------------------------------------------------------------*/

        GPState<T> Xt(s*Dt); vector<vector<Matrix<T, 3, 3>>> DXt_DXa; vector<vector<Matrix<T, 3, 3>>> DXt_DXb;

        Eigen::Matrix<T, 9, 1> gammaa;
        Eigen::Matrix<T, 9, 1> gammab;
        Eigen::Matrix<T, 9, 1> gammat;

        gpm->ComputeXtAndJacobians(Xa, Xb, Xt, DXt_DXa, DXt_DXb, gammaa, gammab, gammat);

        // Landmark Coordinate
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> landmark (parameters[12]);
        Eigen::Matrix<T, 3, 1> predict_meas = Xt.R.matrix().transpose()*(landmark - Xt.P);
        // Residual 3x1
        Eigen::Map<Matrix<T, 3, 1>> residual(residuals);   
        residual = (predict_meas/predict_meas.norm() - imgmeas.obs.cast<T>());
        return true;
    }

private:
    // bearing vector
    ImageMeas imgmeas;

    // Gaussian process params
    
    const int Ridx = 0;
    const int Oidx = 1;
    const int Sidx = 2;
    const int Pidx = 3;
    const int Vidx = 4;
    const int Aidx = 5;

    const int RaIdx = 0;
    const int OaIdx = 1;
    const int SaIdx = 2;
    const int PaIdx = 3;
    const int VaIdx = 4;
    const int AaIdx = 5;

    const int RbIdx = 6;
    const int ObIdx = 7;
    const int SbIdx = 8;
    const int PbIdx = 9;
    const int VbIdx = 10;
    const int AbIdx = 11;

    // Spline param
    double Dt;
    double s;
    int n_knots;

    GPMixerPtr gpm;
};


void generateGTtrajectory()
{
    yolos("generateGTtrajectory");
    traj_gt->setStartTime(0);
    GPState<double> X0;
    X0.R = SO3d::exp(Vec3::Random()* M_PI);
    X0.O = Vec3::Zero();
    X0.S = Vec3::Zero();
    X0.P = Vec3::Zero();
    X0.V = Vec3::Ones();
    X0.A = Vec3::Zero();

    traj_gt->setKnot(0,X0);

    for (size_t i = 0; i < SimulationTime/Dt; i++)
    {
        traj_gt->extendOneKnot();
    }    
}

void generateIMUMeasurements()
{
    yolos("generateIMUMeasurements");
    for (size_t i = 0; i < SimulationTime/Dt_imu; i++)
    {
        // std::cout<<i<<"\n";
        double currentTime = i*Dt_imu;
        GPState<double> currentState =  traj_gt->getStateAt(currentTime);
        Eigen::Vector3d acc_w = currentState.A;
        Eigen::Vector3d gyro_w = currentState.O;

        Eigen::Vector3d acc_b = currentState.R.inverse() * (acc_w-G_w) ;
        Eigen::Vector3d gyro_b = gyro_w;


        IMU timu;
        timu.time = currentTime;
        timu.acc = acc_b + Vec3::Random() * ACC_N;
        timu.gyro = gyro_b + Vec3::Random() * GYRO_N;
        imu_meas.push_back(timu);
    }
    yolos("IMU size: %d",imu_meas.size());
    
}

void generateLandMarks()
{
    yolos("generateLandmarks");
    for (size_t i = 0; i < LandmarkSize; i++)
    {
        Landmarks.push_back(Vec3::Random() * Scale);
        Landmarks_estimate.push_back(Landmarks.back() + Vec3::Random());
    }

    for (size_t i = 0; i < SimulationTime/Dt_image; i++)
    {
        double currentTime = i*Dt_image;

        // get pose
        GPState<double> currentState = traj_gt->getStateAt(currentTime);

        for (size_t j = 0; j < Landmarks.size(); j++)
        {
            // projection
            Eigen::Vector3d t_bearingvec = currentState.R.inverse() * (Landmarks[j] - currentState.P);
            ImageMeas t_image;
            t_image.landmarkId = j;
            t_image.time = currentTime;
            t_image.obs = t_bearingvec.normalized() + IMAGE_OBS_N * Vec3::Random();
            image_meas.push_back(t_image);
        }        

    }
    yolos("Image measurement size: %d",image_meas.size());
}

void generateInitialTrajectory()
{
    yolos("generateInitialtrajectory");
    
    traj_estimate->setStartTime(0);
    
    // check position accuracy
    double error = 0;
    for (size_t i = 0; i < SimulationStep; i++)
    {
        // double currentTime = i*Dt;
        GPState<double> currentGTState = traj_gt->getKnot(i);
        GPState<double> currentEstiState = currentGTState;

        // add noise
        if(i!=0)
        {
            currentEstiState.P += Vec3::Random() * 10;
            currentEstiState.R *= SO3d::exp(0.1*Vec3::Random()* M_PI);
        }
        
        traj_estimate->extendOneKnot();
        traj_estimate->setKnot(i, currentEstiState);
        error += (currentGTState.P-currentEstiState.P).norm();
    }   
    yolos("Initial Average Position Error: %f",error/SimulationStep); 

    // check measurement error
    error = 0;
    for (size_t i = 0; i < image_meas.size(); i++)
    {
        image_meas[i].obs;
        double currentTime = image_meas[i].time;
        // image_meas[i].landmarkId;

        Eigen::Vector3d currentLandMark = Landmarks[image_meas[i].landmarkId];
        GPState<double> currentState =  traj_estimate->getStateAt(currentTime);

        error += 1-fabs(image_meas[i].obs.transpose()*(currentState.R.inverse()*(currentLandMark-currentState.P)).normalized());
    }
    yolos("Initial Average Image Measurement Error: %f",error/image_meas.size()); 
}

void optimize()
{
    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = MAX_THREADS;
    // options.max_num_iterations = 50;
    int KNOTS = traj_estimate->getNumKnots();
    // Add the parameter blocks for rotation
    for (int kidx = 0; kidx < KNOTS; kidx++)
    {
        problem.AddParameterBlock(traj_estimate->getKnotSO3(kidx).data(), 4, new GPSO3dLocalParameterization());
        problem.AddParameterBlock(traj_estimate->getKnotOmg(kidx).data(), 3);
        problem.AddParameterBlock(traj_estimate->getKnotAlp(kidx).data(), 3);
        problem.AddParameterBlock(traj_estimate->getKnotPos(kidx).data(), 3);
        problem.AddParameterBlock(traj_estimate->getKnotVel(kidx).data(), 3);
        problem.AddParameterBlock(traj_estimate->getKnotAcc(kidx).data(), 3);
    }
    // Fix the knots
    double fixed_start = 0.01;
    for (int kidx = 0; kidx < KNOTS; kidx++)
    {
        if (traj_estimate->getKnotTime(kidx) <= traj_estimate->getMinTime() + fixed_start)
        {
            problem.SetParameterBlockConstant(traj_estimate->getKnotSO3(kidx).data());
            problem.SetParameterBlockConstant(traj_estimate->getKnotOmg(kidx).data());
            problem.SetParameterBlockConstant(traj_estimate->getKnotAlp(kidx).data());
            problem.SetParameterBlockConstant(traj_estimate->getKnotPos(kidx).data());
            problem.SetParameterBlockConstant(traj_estimate->getKnotVel(kidx).data());
            problem.SetParameterBlockConstant(traj_estimate->getKnotAcc(kidx).data());
            printf("Fixed knot %d\n", kidx);
        }
    }

    // must add motion constraints
    for (int kidx = 0; kidx < traj_estimate->getNumKnots() - 1; kidx++)
    {
        // Create the factor
        GPMotionPriorTwoKnotsFactorAutodiff *GPMPFactor = new GPMotionPriorTwoKnotsFactorAutodiff(traj_estimate->getGPMixerPtr());
        auto *cost_function = new ceres::DynamicAutoDiffCostFunction<GPMotionPriorTwoKnotsFactorAutodiff>(GPMPFactor);
        cost_function->SetNumResiduals(18);
        vector<double *> factor_param_blocks;
        // Add the parameter blocks
        for (int knot_idx = kidx; knot_idx < kidx + 2; knot_idx++)
        {
            factor_param_blocks.push_back(traj_estimate->getKnotSO3(knot_idx).data());
            factor_param_blocks.push_back(traj_estimate->getKnotOmg(knot_idx).data());
            factor_param_blocks.push_back(traj_estimate->getKnotAlp(knot_idx).data());
            factor_param_blocks.push_back(traj_estimate->getKnotPos(knot_idx).data());
            factor_param_blocks.push_back(traj_estimate->getKnotVel(knot_idx).data());
            factor_param_blocks.push_back(traj_estimate->getKnotAcc(knot_idx).data());
            cost_function->AddParameterBlock(4);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
        }
        auto res_block = problem.AddResidualBlock(cost_function, nullptr, factor_param_blocks);
    }

    // add the landmarks to states
    for (size_t i = 0; i < Landmarks_estimate.size(); i++)
    {
        problem.AddParameterBlock(Landmarks_estimate[i].data(), 3);
        // problem.SetParameterBlockConstant(Landmarks_estimate[i].data());
    }

    // add image obs constraints
    for (size_t i = 0; i < image_meas.size(); i++)
    {
        ImageMeas t_img = image_meas[i];
        double currentTime = image_meas[i].time;
        auto   us = traj_estimate->computeTimeIndex(t_img.time);
        int    u  = us.first;
        double s  = us.second;

        if(t_img.time > traj_estimate->getMaxTime() - Dt)
        {
            continue;
        }

        GPBearingvecFactorAutodiff * GPBVFactor = new GPBearingvecFactorAutodiff(t_img, traj_estimate->getGPMixerPtr(), s, traj_estimate->getNumKnots());
        auto *cost_function = new ceres::DynamicAutoDiffCostFunction<GPBearingvecFactorAutodiff>(GPBVFactor);
        cost_function->SetNumResiduals(3);
        std::vector<double *> factor_param_blocks;
        // Add the parameter blocks
        for (int knot_idx = u; knot_idx < u + 2; knot_idx++)
        {
            factor_param_blocks.push_back(traj_estimate->getKnotSO3(knot_idx).data());
            factor_param_blocks.push_back(traj_estimate->getKnotOmg(knot_idx).data());
            factor_param_blocks.push_back(traj_estimate->getKnotAlp(knot_idx).data());
            factor_param_blocks.push_back(traj_estimate->getKnotPos(knot_idx).data());
            factor_param_blocks.push_back(traj_estimate->getKnotVel(knot_idx).data());
            factor_param_blocks.push_back(traj_estimate->getKnotAcc(knot_idx).data());
            cost_function->AddParameterBlock(4);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
        }
        factor_param_blocks.push_back(Landmarks_estimate[t_img.landmarkId].data());
        cost_function->AddParameterBlock(3);

        auto res_block = problem.AddResidualBlock(cost_function, nullptr, factor_param_blocks);
        // ceres::Problem::EvaluateOptions e_option;
        // ceres::CRSMatrix Jacobian;
        // e_option.residual_blocks.push_back(res_block);
        // double cost;
        // std::vector<double> residual;
        // problem.Evaluate(e_option, &cost, &residual, nullptr, nullptr);
        // printf("%f %f %f\n",residual[0],residual[1],residual[2]);

    }

    // add IMU constraints
    for (size_t i = 0; i < imu_meas.size(); i++)
    {
        IMU t_imu = imu_meas[i];
        auto   us = traj_estimate->computeTimeIndex(t_imu.time);
        int    u  = us.first;
        double s  = us.second;

        if(t_imu.time > traj_estimate->getMaxTime() - Dt)
        {
            continue;
        }

        GPIMUFactorAutodiff *GPMPFactor = new GPIMUFactorAutodiff(t_imu.acc, t_imu.gyro, ba, bg, 1./ACC_N, 1./GYRO_N, 10000000, 10000000, traj_estimate->getGPMixerPtr(), s, traj_estimate->getNumKnots());
        auto *cost_function = new ceres::DynamicAutoDiffCostFunction<GPIMUFactorAutodiff>(GPMPFactor);
        cost_function->SetNumResiduals(12);
        std::vector<double *> factor_param_blocks;
        // Add the parameter blocks
        for (int knot_idx = u; knot_idx < u + 2; knot_idx++)
        {
            factor_param_blocks.push_back(traj_estimate->getKnotSO3(knot_idx).data());
            factor_param_blocks.push_back(traj_estimate->getKnotOmg(knot_idx).data());
            factor_param_blocks.push_back(traj_estimate->getKnotAlp(knot_idx).data());
            factor_param_blocks.push_back(traj_estimate->getKnotPos(knot_idx).data());
            factor_param_blocks.push_back(traj_estimate->getKnotVel(knot_idx).data());
            factor_param_blocks.push_back(traj_estimate->getKnotAcc(knot_idx).data());
            cost_function->AddParameterBlock(4);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
        }
        factor_param_blocks.push_back(ba.data());
        factor_param_blocks.push_back(bg.data());
        cost_function->AddParameterBlock(3);
        cost_function->AddParameterBlock(3);

        auto res_block = problem.AddResidualBlock(cost_function, nullptr, factor_param_blocks);
        
        // ceres::Problem::EvaluateOptions e_option;
        // ceres::CRSMatrix Jacobian;
        // e_option.residual_blocks.push_back(res_block);
        // double cost;
        // std::vector<double> residual;
        // problem.Evaluate(e_option, &cost, &residual, nullptr, nullptr);
        // std::cout<<residual.transpose()<<"\n";
        // printf("%f %f %f %f %f %f \n",residual[0],residual[1],residual[2],residual[3],residual[4],residual[5]);
        
    }
    
    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.FullReport() << "\n";
    std::cout << "Estimated BA:" << ba.transpose() <<"\n";
    std::cout << "Estimated BG:" << bg.transpose() <<"\n";

    // check position accuracy
    double error = 0;
    for (size_t i = 0; i < SimulationStep-1; i++)
    {
        // double currentTime = i*Dt;
        GPState<double> currentGTState = traj_gt->getKnot(i);
        GPState<double> currentEstiState = traj_estimate->getKnot(i);
        error += (currentGTState.P-currentEstiState.P).norm();
    }   
    yolos("Optimized Average Position Error: %f",error/SimulationStep); 


}

int main()
{
    yolos("begin simulation");

    generateGTtrajectory();
    generateIMUMeasurements();
    generateLandMarks();
    generateInitialTrajectory();
    optimize();

    yolos("finish");
}