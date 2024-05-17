
#include <Eigen/Dense>
#include "CloudMatcher.hpp"
#include "utility.h"

using namespace Eigen;
typedef Matrix<double, 12, 12> MatrixNd;
typedef Matrix<double, 12, 1>  VectorNd;

class StateWithCov
{
public:
    // time stamp
    double tcurr;

    // State estimate
    SO3d     Rot = SO3d(Quaternd(1, 0, 0, 0));
    Vector3d Pos = Vector3d(0, 0, 0);
    Vector3d Omg = Vector3d(0, 0, 0);
    Vector3d Vel = Vector3d(0, 0, 0);
    
    // Error state covariance
    MatrixNd Cov = MatrixNd::Identity();

    // Destructor
   ~StateWithCov() {};
    
    // Constructor
    
    StateWithCov()
        : tcurr(0), Rot(SO3d(Quaternd(1, 0, 0, 0))), Pos(Vector3d(0, 0, 0)), Omg(Vector3d(0, 0, 0)), Vel(Vector3d(0, 0, 0)), Cov(MatrixNd::Identity())
    {}

    StateWithCov(double t, Quaternd &Q, Vector3d &P, Vector3d W, Vector3d V, MatrixNd C)
        : tcurr(t), Rot(SO3d(Q)), Pos(P), Omg(W), Vel(V), Cov(C)
    {}

    StateWithCov(double t, SO3d &Q, Vector3d &P, Vector3d W, Vector3d V, MatrixNd C)
        : tcurr(t), Rot(Q), Pos(P), Omg(W), Vel(V), Cov(C)
    {}

    StateWithCov(double t, Quaternd &Q, Vector3d &P, Vector3d W, Vector3d V, double sigma)
        : tcurr(t), Rot(SO3d(Q)), Pos(P), Omg(W), Vel(V), Cov(MatrixNd::Identity()*sigma)
    {}


    StateWithCov(const StateWithCov &X)
        : tcurr(X.tcurr), Rot(SO3d(X.Rot)), Pos(X.Pos), Omg(X.Omg), Vel(X.Vel), Cov(X.Cov)
    {}

    StateWithCov& operator=(const StateWithCov &other)
    {
        if (this != &other)
        {
            this->tcurr = other.tcurr;
            this->Rot   = other.Rot  ;
            this->Pos   = other.Pos  ;
            this->Omg   = other.Omg  ;
            this->Vel   = other.Vel  ;
            this->Cov   = other.Cov  ;
        }

        return *this;
    }

    Matrix3d rightJacobian(const Vector3d &phi) const
    {
        Matrix3d Jr;
        Sophus::rightJacobianSO3(phi, Jr);
        return Jr;
    }

    Matrix3d rightJacobianInvSO3(const Vector3d &phi) const
    {
        Matrix3d Jr;
        Sophus::rightJacobianInvSO3(phi, Jr);
        return Jr;
    }

    // Increment
    StateWithCov boxplusf(double tn, const MatrixNd &Rm)
    {
        // Load the previous states
        double   tc   = tcurr;
        SO3d     Qc   = Rot;
        Vector3d Pc   = Pos;
        Vector3d Wc   = Omg;
        Vector3d Vc   = Vel;
        MatrixNd Covc = Cov;

        // Find the delta t
        double dt = tn - tc;

        // Intermediary quanities
        Vector3d dA = Wc*dt;
        SO3d dQ = SO3d::exp(Wc*dt);
        SO3d Qcinv = Qc.inverse();
        
        // Predict the state 
        SO3d Qn = Qc*dQ;
        Vector3d Pn = Pc + Vc*dt;
        Vector3d Wn = Wc;
        Vector3d Vn = Vc;

        // Calculate the transition matrix
        MatrixNd Fx = MatrixNd::Identity();
        
        // Map the blocks to Fx
        Eigen::Block<MatrixNd, 3, 3> dQdR = Fx.block<3, 3>(0, 0);
        Eigen::Block<MatrixNd, 3, 3> dQdW = Fx.block<3, 3>(0, 6);
        
        Eigen::Block<MatrixNd, 3, 3> dPdQ = Fx.block<3, 3>(3, 0);
        Eigen::Block<MatrixNd, 3, 3> dPdP = Fx.block<3, 3>(3, 3);
        Eigen::Block<MatrixNd, 3, 3> dPdW = Fx.block<3, 3>(3, 6);
        Eigen::Block<MatrixNd, 3, 3> dPdV = Fx.block<3, 3>(3, 9);

        Eigen::Block<MatrixNd, 3, 3> dWdW = Fx.block<3, 3>(6, 6);

        Eigen::Block<MatrixNd, 3, 3> dVdQ = Fx.block<3, 3>(9, 0);
        Eigen::Block<MatrixNd, 3, 3> dVdW = Fx.block<3, 3>(9, 6);
        Eigen::Block<MatrixNd, 3, 3> dVdV = Fx.block<3, 3>(9, 9);

        // Populate the blocks with calculations
        dQdR = dQ.matrix().transpose();
        dQdW = rightJacobian(dA)*dt;

        dPdP = Matrix3d::Identity();
        dPdV = Matrix3d::Identity()*dt;

        dWdW = Matrix3d::Identity();
        dVdV = Matrix3d::Identity();

        MatrixNd Fm = MatrixNd::Zero();

        // Map the blocks of Fm
        Eigen::Block<MatrixNd, 3, 3> dWdEW = Fm.block<3, 3>(6, 6);
        Eigen::Block<MatrixNd, 3, 3> dVdEV = Fm.block<3, 3>(9, 9);

        dWdEW = Matrix3d::Identity()*dt;
        dVdEV = Matrix3d::Identity()*dt;

        MatrixNd Covn = Fx*Covc*Fx.transpose() + Fm*Rm*Fm.transpose();

        // printf("Fx: \n");
        // cout << Fx << endl;
        // printf("Fm: \n");
        // cout << Fm << endl;
        // printf("Covc: \n");
        // cout << Covc << endl;
        // printf("Covn: \n");
        // cout << Covn << endl;

        return StateWithCov(tn, Qn, Pn, Wn, Vn, Covn);
    }

    void boxplusd(VectorNd &dX)
    {
        SO3d dQ = SO3d::exp(dX.block<3, 1>(0, 0));
        Vector3d dP = dX.block<3, 1>(3, 0);
        Vector3d dW = dX.block<3, 1>(6, 0);
        Vector3d dV = dX.block<3, 1>(9, 0);

        Rot *= dQ;
        Pos += dP;
        Omg += dW;
        Vel += dV;
    }

    void SetCov(MatrixNd &Covn)
    {
        Cov = Covn;
    }

    VectorNd boxminus(StateWithCov &Xbar)
    {
        VectorNd r;
        r << (Xbar.Rot.inverse()*Rot).log(), Pos - Xbar.Pos, Vel - Xbar.Vel, Omg - Xbar.Omg;
        return r;
    }

    VectorNd boxminus(StateWithCov &Xbar, MatrixNd &J)
    {
        Vector3d Phi = (Xbar.Rot.inverse()*Rot).log();
        VectorNd r;
        r << Phi, Pos - Xbar.Pos, Vel - Xbar.Vel, Omg - Xbar.Omg;
        
        J = MatrixNd::Identity();
        J.block<3, 3>(0, 0) = rightJacobianInvSO3(Phi);

        return r;
    }
    
    Vector3d YPR()
    {
        return Util::Quat2YPR(Rot.unit_quaternion());
    }
};

class EKFLO
{
private:

    StateWithCov Xhat;

    // Covariance of random processes
    double Rw, Rv;
    MatrixNd Rm = MatrixNd::Zero();

    // Associate params
    int knnSize = 6;
    double minKnnSqDis = 0.5*0.5;
    double minKnnNbrDis = 0.1;

    // Max iterations
    int MAX_ITER = 5;
    double lambda = 0.0;

public:

    // Destructor
   ~EKFLO() {};

    EKFLO(const StateWithCov &X0, double Rw_, double Rv_, double minKnnSqDis_, double minKnnNbrDis_)
    : Xhat(X0), Rw(Rw_), Rv(Rv_), minKnnSqDis(minKnnSqDis_), minKnnNbrDis(minKnnNbrDis_)
    {
        // // Initialize the covariance
        // Cov = MatrixNd::Identity()*Cov0;
        Eigen::VectorXd Rm_(12);
        Rm_ << 0, 0, 0, 0, 0, 0, Rw, Rw, Rw, Rv, Rv, Rv;
        Rm = Rm_.asDiagonal();
    }   

    void Update()
    {
    }

    void Deskew(CloudXYZITPtr &cloud, CloudXYZIPtr &cloudDeskewedInW,
                myTf<double> &tf_W_Bb, myTf<double> &tf_W_Be,
                double tc, double dt)
    {
        int Npoint = cloud->size();
        cloudDeskewedInW->resize(Npoint);
        #pragma omp parallel for num_threads(MAX_THREADS)
        for(int pidx = 0; pidx < Npoint; pidx++)
        {
            PointXYZIT &pi = cloud->points[pidx];
            PointXYZI  &po = cloudDeskewedInW->points[pidx];

            po.intensity = pi.intensity;
            
            // Interpolation factor
            double s = (pi.t - tc) / dt;
            // Interpolated pose
            myTf tf_W_Bs = tf_W_Bb.slerp(s, tf_W_Be);
            // Transformed point into another frame
            PointXYZIT po_;
            Util::transform_point(tf_W_Bs, pi, po_);
            po.x = po_.x;
            po.y = po_.y;
            po.z = po_.z;
        }
    }

    void Associate(const KdFLANNPtr &kdtreeMap, const CloudXYZIPtr &priormap,
                   const CloudXYZIPtr &cloudInB, const CloudXYZIPtr &cloudInW,
                   deque<Vector3d> &features, deque<Vector4d> &coefficients)
    {
        int totalFeature = 0;
        if (priormap->size() > knnSize)
        {
            int pointsCount = cloudInW->points.size();
            vector<LidarCoef> Coef; Coef.reserve(pointsCount);
            
            #pragma omp parallel for num_threads(MAX_THREADS)
            for (int pidx = 0; pidx < pointsCount; pidx++)
            {
                PointXYZI pointInW = cloudInW->points[pidx];
                PointXYZI pointInB = cloudInB->points[pidx];
                
                Coef[pidx].n = Vector4d(0, 0, 0, 0);
                Coef[pidx].t = -1;

                if(!Util::PointIsValid(pointInB))
                {
                    // printf(KRED "Invalid surf point!: %f, %f, %f\n" RESET, pointInB.x, pointInB.y, pointInB.z);
                    pointInB.x = 0; pointInB.y = 0; pointInB.z = 0; pointInB.intensity = 0;
                    continue;
                }

                // Calculating the coefficients
                MatrixXd mat_A0(knnSize, 3);
                MatrixXd mat_B0(knnSize, 1);
                Vector3d mat_X0;
                Matrix3d mat_A1;
                MatrixXd mat_D1(1, 3);
                Matrix3d mat_V1;

                mat_A0.setZero();
                mat_B0.setConstant(-1);
                mat_X0.setZero();

                mat_A1.setZero();
                mat_D1.setZero();
                mat_V1.setZero();

                vector<int> knn_idx(knnSize, 0); vector<float> knn_sq_dis(knnSize, 0);
                kdtreeMap->nearestKSearch(pointInW, knnSize, knn_idx, knn_sq_dis);

                if (knn_sq_dis.back() < minKnnSqDis)
                {
                    // printf(KGRN "Pidx: %d. Nbr: %d\n" RESET, pidx, knn_idx.size());

                    for (int j = 0; j < knn_idx.size(); j++)
                    {
                        mat_A0(j, 0) = priormap->points[knn_idx[j]].x;
                        mat_A0(j, 1) = priormap->points[knn_idx[j]].y;
                        mat_A0(j, 2) = priormap->points[knn_idx[j]].z;
                    }
                    mat_X0 = mat_A0.colPivHouseholderQr().solve(mat_B0);

                    float pa = mat_X0(0, 0);
                    float pb = mat_X0(1, 0);
                    float pc = mat_X0(2, 0);
                    float pd = 1;

                    float ps = sqrt(pa * pa + pb * pb + pc * pc);
                    pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                    // NOTE: plane as (x y z)*w+1 = 0

                    bool valid_plane = true;
                    for (int j = 0; j < knn_idx.size(); j++)
                    {
                        double d2p = pa * priormap->points[knn_idx[j]].x +
                                     pb * priormap->points[knn_idx[j]].y +
                                     pc * priormap->points[knn_idx[j]].z + pd;

                        if (fabs(d2p) > minKnnNbrDis)
                        {
                            valid_plane = false;
                            break;
                        }
                    }

                    if (valid_plane)
                    {
                        float d2p = pa * pointInB.x + pb * pointInB.y + pc * pointInB.z + pd;

                        // Weightage based on close the point is to the plane ?
                        float score = (1 - 0.9f * fabs(d2p)) / Util::pointDistance(pointInB);

                        if (score > 0)
                        {
                            Coef[pidx].t    = 0;
                            Coef[pidx].f    = Vector3d(pointInB.x, pointInB.y, pointInB.z);
                            Coef[pidx].fdsk = Vector3d(pointInB.x, pointInB.y, pointInB.z);
                            Coef[pidx].n    = Vector4d(score * pa, score * pb, score * pc, score * pd);

                            // printf("Pidx %d admitted. Score: %f.\n", pidx, score);
                        }
                    }
                }
                // else
                //     printf(KRED "Pidx: %d. Nbr: %d\n" RESET, pidx, knn_idx.size());
            }
            
            // Copy the coefficients to the buffer
            features.clear();
            coefficients.clear();
            
            for(int pidx = 0; pidx < pointsCount; pidx++)
            {
                LidarCoef &coef = Coef[pidx];
                if (coef.t >= 0)
                {
                    features.push_back(coef.f);
                    coefficients.push_back(coef.n);
                    totalFeature++;
                    // printf("Feature %d admitted.\n", features.size());
                }
            }
        }
    }

    void EvaluateLidar(const StateWithCov &Xpred, const deque<Vector3d> &features, const deque<Vector4d> &coefficients,
                       VectorXd &RESIDUAL, MatrixXd &JACOBIAN)
    {
        int Nf = features.size();

        #pragma omp parallel for num_threads(MAX_THREADS)
        for(int fidx = 0; fidx < Nf; fidx++)
        {
            Vector3d f = features[fidx];
            Vector3d n = coefficients[fidx].head<3>();
            double   w = coefficients[fidx].w();
            MatrixXd R = Xpred.Rot.matrix();
            Vector3d p = Xpred.Pos;

            RESIDUAL(fidx) = n.dot(R*f + p) + w;
            JACOBIAN.block(fidx, 0, 1, 12) << -n.transpose()*R*Util::skewSymmetric(f)
                                              ,n.transpose()
                                              ,0, 0, 0, 0, 0, 0;
        }
    }

    void findTraj(const KdFLANNPtr &kdTreeMap, const CloudXYZIPtr priormap,
                 vector<CloudXYZITPtr> &clouds, vector<ros::Time> &cloudstamp,
                 CloudPosePtr &poseprior, boost::shared_ptr<ros::NodeHandle> &nh_ptr)
    {
        int Ncloud = clouds.size();
        ROS_ASSERT(Ncloud == poseprior->size());
        for(int cidx = 0; cidx < Ncloud; cidx++)
        {
            // Step 0: Save current state and identify the time step to be the end of the scan
            myTf tf_W_Bb(Xhat.Rot.matrix(), Xhat.Pos);

            double tc = clouds[cidx]->points.front().t; // -> tcurr, ideally should be equal to tcurr, and cloudstamp[cidx].stamp
            double tn = clouds[cidx]->points.back().t; // -> tnext
            double dt = tn - tc;
            // tn > tc and time step arround 0.1 second
            ROS_ASSERT_MSG(dt > 0 && fabs(dt - 0.1) < 0.02, "Time step error: %f", dt);

            // Step 1: Predict the trajectory, use this as the prior
            StateWithCov Xprior = Xhat.boxplusf(tn, Rm);

            // Step 2: Iterative update with internal deskew and association
            StateWithCov Xpred = Xprior;
            for (int iidx = 0; iidx < MAX_ITER; iidx++)
            {
                MatrixNd Jprior;
                VectorNd rprior = Xpred.boxminus(Xprior, Jprior);

                // Pose at end time
                myTf tf_W_Be(Xpred.Rot.matrix(), Xpred.Pos);
                myTf tf_Be_W = tf_W_Be.inverse();

                // Step 2: Deskew the pointcloud, transform all points to
                CloudXYZIPtr cloudDeskewedInB(new CloudXYZI());
                CloudXYZIPtr cloudDeskewedInW(new CloudXYZI());
                Deskew(clouds[cidx], cloudDeskewedInW, tf_W_Bb, tf_W_Be, tc, dt);
                pcl::transformPointCloud(*cloudDeskewedInW, *cloudDeskewedInB, tf_Be_W.pos, tf_Be_W.rot);

                // Step 3: Associate pointcloud with map
                deque<Vector3d> features;
                deque<Vector4d> coefficients;
                Associate(kdTreeMap, priormap, cloudDeskewedInB, cloudDeskewedInW, features, coefficients);

                static ros::Publisher cloudDskPub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/clouddsk_inW", 1);
                Util::publishCloud(cloudDskPub, *cloudDeskewedInW, ros::Time::now(), "world");
                std::this_thread::sleep_for(std::chrono::seconds(1));

                // Step 4: Update the state with the detected features
                int Nf = features.size();
                VectorXd RESIDUAL(Nf, 1);
                MatrixXd JACOBIAN(Nf, 12);

                EvaluateLidar(Xpred, features, coefficients, RESIDUAL, JACOBIAN);

                // Solve for best increment
                bool solver_failed; VectorNd dX; MatrixNd Cov;
                // Sparsify the matrices for easy calculation
                SparseMatrix<double> J    = JACOBIAN.sparseView(); J.makeCompressed();
                SparseMatrix<double> Jtp  = J.transpose();
                SparseMatrix<double> Jp   = Jprior.sparseView(); Jp.makeCompressed();
                SparseMatrix<double> Jptp = Jp.transpose();
                MatrixXd B = -Jtp*RESIDUAL - Jptp*Xprior.Cov*rprior;
                SparseMatrix<double> H = Jtp*J + Jptp*Xprior.Cov*Jp;
                // Solve using solver and LM method
                SparseMatrix<double> I(H.cols(), H.cols()); I.setIdentity();
                SparseMatrix<double> S = H + lambda/pow(2, (MAX_ITER - 1) - iidx)*I;
                // Factorize
                Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
                solver.analyzePattern(S);
                solver.factorize(S);
                // Solve
                solver_failed = solver.info() != Eigen::Success;
                dX = solver.solve(B);
                Cov = H.toDense();

                // If solving is not successful, return false
                if (solver_failed || dX.hasNaN())
                {
                    printf(KRED"Failed to solve!\n"RESET);
                    cout << dX;
                }
                else
                {   
                    // Cap the change
                    if (dX.norm() > 0.5)
                        dX = dX/dX.norm();

                    Xpred.boxplusd(dX);
                    Xpred.SetCov(Cov);
                    
                    printf("CIDX: %d. ITER: %d. Time: %f. Features: %d\n", cidx, iidx, Xpred.tcurr, Nf);
                    // printf("\tCov: \n");
                    // cout << S.toDense() << endl;

                    printf("\tdX  : dR : %6.3f, %6.3f, %6.3f. dP : %6.3f, %6.3f, %6.3f. dW : %6.3f, %6.3f, %6.3f. dV : %6.3f, %6.3f, %6.3f\n",
                            dX(0), dX(1), dX(2), dX(3), dX(4), dX(5), dX(6), dX(7), dX(8), dX(9), dX(10), dX(11));

                    printf("\tXhat: Pos: %6.3f, %6.3f, %6.3f. Rot: %6.3f, %6.3f, %6.3f. Omg: %6.3f, %6.3f, %6.3f. Vel: %6.3f, %6.3f, %6.3f.\n",
                            Xpred.Pos(0),    Xpred.Pos(1),    Xpred.Pos(2),
                            Xpred.YPR().x(), Xpred.YPR().y(), Xpred.YPR().z(),
                            Xpred.Omg(0),    Xpred.Omg(1),    Xpred.Omg(2),
                            Xpred.Vel(0),    Xpred.Vel(1),    Xpred.Vel(2));

                    printf("\tX.Cov: \n");
                    cout << Xpred.Cov << endl;
                }
            }

            Xhat = Xpred;
            printf("\n");
            
            PointPose pose = myTf(Xpred.Rot.matrix(), Xpred.Pos).Pose6D(Xpred.tcurr);
            poseprior->points[cidx] = pose;

            // Check
            printf(KYEL "\tXhat: Pos: %6.3f, %6.3f, %6.3f. Rot: %6.3f, %6.3f, %6.3f. Omg: %6.3f, %6.3f, %6.3f. Vel: %6.3f, %6.3f, %6.3f.\n" RESET,
                   Xhat.Pos(0),    Xhat.Pos(1),    Xhat.Pos(2),
                   Xhat.YPR().x(), Xhat.YPR().y(), Xhat.YPR().z(),
                   Xhat.Omg(0),    Xhat.Omg(1),    Xhat.Omg(2),
                   Xhat.Vel(0),    Xhat.Vel(1),    Xhat.Vel(2));
                    
            // // DEBUG: Break if loops 5 times.
            // static int debug_count = 0;
            // debug_count++;
            // if (debug_count > 0)
            //     break;
        }
    }
};