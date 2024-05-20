
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

    StateWithCov(double t, const Quaternd &Q, const Vector3d &P, Vector3d W, Vector3d V, MatrixNd C)
        : tcurr(t), Rot(SO3d(Q)), Pos(P), Omg(W), Vel(V), Cov(C)
    {}

    StateWithCov(double t, const SO3d &Q, const Vector3d &P, Vector3d W, Vector3d V, MatrixNd C)
        : tcurr(t), Rot(Q), Pos(P), Omg(W), Vel(V), Cov(C)
    {}

    StateWithCov(double t, const Quaternd &Q, const Vector3d &P, Vector3d W, Vector3d V, double sigma)
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
    StateWithCov Propagation(double tn, StateWithCov &Xhatprev, const MatrixNd &Rm)
    {
        // Find the old delta t
        double dt_ = (tcurr - Xhatprev.tcurr);

        // Load the previous states
        double   tc   = tcurr;
        SO3d     Qc   = Rot;
        Vector3d Pc   = Pos;
        Vector3d Wc   = dt_ == 0.0 ? Vector3d(0, 0, 0) : (Xhatprev.Rot.inverse()*Rot).log() / dt_;
        Vector3d Vc   = dt_ == 0.0 ? Vector3d(0, 0, 0) : (Pos - Xhatprev.Pos) / dt_;
        MatrixNd Covc = Cov;

        // Find the new delta t
        double dt = tn - tc;

        // Intermediary quanities
        Vector3d dA    = Wc*dt;
        SO3d     dQ    = SO3d::exp(Wc*dt);
        SO3d     Qcinv = Qc.inverse();
        
        // Predict the state 
        SO3d     Qn = Qc*dQ;
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

        // MatrixNd W = Eigen::LLT<Eigen::Matrix<double, 6, 6>>(Xbar.Cov.inverse()).matrixL().transpose();
        return r;
    }
    
    Vector3d YPR()
    {
        return Util::Quat2YPR(Rot.unit_quaternion());
    }
};

class GPLO
{
private:

    NodeHandlePtr &nh_ptr;

    StateWithCov Xhatprev;
    StateWithCov Xhat;

    // Covariance of random processes
    double Rw, Rv;
    MatrixNd Rm = MatrixNd::Zero();

    // Associate params
    int knnSize = 6;
    double minKnnSqDis = 0.5*0.5;
    double minKnnNbrDis = 0.1;

    // Max iterations
    int MAX_ITER  = 10;
    double lambda = 0.0;
    double dXmax  = 0.5;

    // Number of debug steps
    int DEBUG_STEPS = 1;

    // Lidar index
    int lidx;

    // Visualization
    ros::Publisher assocPub;
    ros::Publisher cloudDskPub;
    ros::Publisher pppub;

    // Pose prior for visualization
    CloudPosePtr poseprior = CloudPosePtr(new CloudPose());
    int debug_count = 0;

public:

    // Destructor
   ~GPLO() {};

    // Constructor
    GPLO(int lidx_, const StateWithCov &X0, double Rw_, double Rv_, double minKnnSqDis_, double minKnnNbrDis_, NodeHandlePtr &nh_ptr_, mutex &nh_mtx)
    : lidx(lidx_), Xhatprev(X0), Xhat(X0), Rw(Rw_), Rv(Rv_), minKnnSqDis(minKnnSqDis_), minKnnNbrDis(minKnnNbrDis_), nh_ptr(nh_ptr_)
    {
        // Initialize the covariance of velocity
        Eigen::VectorXd Rm_(12);
        Rm_ << 0, 0, 0, 0, 0, 0, Rw, Rw, Rw, Rv, Rv, Rv;
        Rm = Rm_.asDiagonal();

        // Get the maximum change
        nh_ptr->getParam("dXmax", dXmax);

        // Get the number of max iterations
        nh_ptr->getParam("MAX_ITER", MAX_ITER);

        // Debug steps
        nh_ptr->getParam("DEBUG_STEPS", DEBUG_STEPS);

        nh_mtx.lock();

        // Advertise report
        assocPub    = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/assoc_cloud",  lidx), 1);
        cloudDskPub = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/clouddsk_inW", lidx), 1);
        pppub       = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/pose_prior",   lidx), 1);

        nh_mtx.unlock();
    }

    void Deskew(const CloudXYZITPtr &cloud, CloudXYZIPtr &cloudDeskewedInB,
                const myTf<double> &tf_W_Bb, const myTf<double> &tf_W_Be, double tc, double dt)
    {
        // Total number of points
        int Npoint = cloud->size();

        // Transform
        myTf tf_Be_Bb = tf_W_Be.inverse()*tf_W_Bb;
        myTf tfI;

        // Reset the output pointcloud
        cloudDeskewedInB->clear(); cloudDeskewedInB->resize(Npoint);

        // Assign the new value
        #pragma omp parallel for num_threads(MAX_THREADS)
        for(int pidx = 0; pidx < Npoint; pidx++)
        {
            PointXYZIT &pi = cloud->points[pidx];
            PointXYZI  &po = cloudDeskewedInB->points[pidx];
            po.intensity = pi.intensity;
            
            // Interpolation factor
            double s = 1.0 - min(1.0, (pi.t - tc) / dt);

            // Interpolated pose
            myTf tf_Be_Bs = tfI.slerp(s, tf_Be_Bb);

            // Transformed point into another frame
            PointXYZIT po_;
            Util::transform_point(tf_Be_Bs, pi, po_);
            po.x = po_.x;
            po.y = po_.y;
            po.z = po_.z;
        }
    }

    void Associate(const KdFLANNPtr &kdtreeMap, const CloudXYZIPtr &priormap,
                   const CloudXYZIPtr &cloudInB, const CloudXYZIPtr &cloudInW, vector<LidarCoef> &Coef)
    {
        if (priormap->size() > knnSize)
        {
            int pointsCount = cloudInW->points.size();
            vector<LidarCoef> Coef_;
            Coef_.resize(pointsCount);
            
            #pragma omp parallel for num_threads(MAX_THREADS)
            for (int pidx = 0; pidx < pointsCount; pidx++)
            {
                PointXYZI pointInB = cloudInB->points[pidx];
                PointXYZI pointInW = cloudInW->points[pidx];

                Coef_[pidx].n = Vector4d(0, 0, 0, 0);
                Coef_[pidx].t = -1;

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
                        float score = 1;//(1 - 0.9f * fabs(d2p) / Util::pointDistance(pointInB));

                        if (score > 0)
                        {
                            Coef_[pidx].t      = 0;
                            Coef_[pidx].finW   = Vector3d(pointInW.x, pointInW.y, pointInW.z);
                            Coef_[pidx].fdsk   = Vector3d(pointInB.x, pointInB.y, pointInB.z);
                            Coef_[pidx].n      = Vector4d(pa, pb, pc, pd);
                            Coef_[pidx].plnrty = score;

                            // printf("Pidx %d admitted. Score: %f.\n", pidx, score);
                        }
                    }
                }
                // else
                //     printf(KRED "Pidx: %d. Nbr: %d\n" RESET, pidx, knn_idx.size());
            }
            
            // Copy the coefficients to the buffer
            Coef.clear();
            int totalFeature = 0;
            for(int pidx = 0; pidx < pointsCount; pidx++)
            {
                LidarCoef &coef = Coef_[pidx];
                if (coef.t >= 0)
                {
                    totalFeature++;
                    Coef.push_back(coef);
                }
            }
        }
    }

    void EvaluateLidar(const StateWithCov &Xpred, const vector<LidarCoef> &Coef,
                       VectorXd &RESIDUAL, MatrixXd &JACOBIAN)
    {
        int Nf = Coef.size();

        #pragma omp parallel for num_threads(MAX_THREADS)
        for(int fidx = 0; fidx < Nf; fidx++)
        {
            Vector3d f = Coef[fidx].fdsk;
            Vector3d n = Coef[fidx].n.head<3>();
            double   m = Coef[fidx].n.w();
            double   w = Coef[fidx].plnrty;
            MatrixXd R = Xpred.Rot.matrix();
            Vector3d p = Xpred.Pos;

            RESIDUAL(fidx) = w*(n.dot(R*f + p) + m);
            JACOBIAN.block(fidx, 0, 1, 12) << -w*n.transpose()*R*Util::skewSymmetric(f)
                                              ,w*n.transpose()
                                              ,0, 0, 0, 0, 0, 0;
        }
    }

    void FindTraj(const KdFLANNPtr &kdTreeMap, const CloudXYZIPtr priormap,
                  const vector<CloudXYZITPtr> &clouds, const vector<ros::Time> &cloudstamp)
    {
        int Ncloud = clouds.size();
        for(int cidx = 0; cidx < Ncloud && ros::ok(); cidx++)
        {
            // Step 0: Save current state and identify the time step to be the end of the scan
            myTf tf_W_Bb(Xhat.Rot.matrix(), Xhat.Pos);

            double tc = clouds[cidx]->points.front().t; // -> tcurr, ideally should be equal to tcurr, and cloudstamp[cidx].stamp
            double tn = clouds[cidx]->points.back().t; // -> tnext
            double dt = tn - tc;

            // assert that tn > tc and time steps differ less then 0.1 seconds
            ROS_ASSERT_MSG(dt > 0 && fabs(dt - 0.1) < 0.02, "Time step error: %f", dt);

            // Step 1: Predict the trajectory, use this as the prior
            StateWithCov Xpred = Xhat.Propagation(tn, Xhatprev, Rm);
            StateWithCov Xprior = Xpred;

            // Step 2: Iterative update with internal deskew and association
            for (int iidx = 0; iidx < MAX_ITER; iidx++)
            {
                double J0, JK;

                MatrixNd Jprior;
                VectorNd rprior = Xpred.boxminus(Xprior, Jprior);

                // Pose at end time
                myTf tf_W_Be(Xpred.Rot.matrix(), Xpred.Pos);
                myTf tf_Be_W = tf_W_Be.inverse();

                // Step 2: Deskew the pointcloud, transform all points to
                CloudXYZIPtr cloudDeskewedInB(new CloudXYZI());
                CloudXYZIPtr cloudDeskewedInW(new CloudXYZI());

                // Step 2.1: Deskew
                Deskew(clouds[cidx], cloudDeskewedInB, tf_W_Bb, tf_W_Be, tc, dt);
                pcl::transformPointCloud(*cloudDeskewedInB, *cloudDeskewedInW, tf_W_Be.pos, tf_W_Be.rot);

                // Step 2.2: Associate pointcloud with map
                vector<LidarCoef> Coef;
                Associate(kdTreeMap, priormap, cloudDeskewedInB, cloudDeskewedInW, Coef);

                // Step 2.3: Find the increment and update the states
                int Nf = Coef.size();
                VectorXd RESIDUAL(Nf, 1);
                MatrixXd JACOBIAN(Nf, 12);
                EvaluateLidar(Xpred, Coef, RESIDUAL, JACOBIAN);

                // Solve for best increment
                VectorNd dX; // increment of the state

                // Sparsify the matrices for easy calculation
                typedef SparseMatrix<double> SparseMat;
                SparseMat J    = JACOBIAN.sparseView(); J.makeCompressed();
                SparseMat Jtp  = J.transpose();
                SparseMat Jp   = Jprior.sparseView(); Jp.makeCompressed();
                SparseMat Jptp = Jp.transpose();
                MatrixNd Gpinv = Xprior.Cov.inverse();

                MatrixXd  B = -Jtp*RESIDUAL -Jptp*Gpinv*rprior;
                SparseMat A =  Jtp*J + Jptp*Gpinv*Jp;

                // Build the Ax=B and solve
                SparseMat I(A.cols(), A.cols()); I.setIdentity();
                Eigen::SparseLU<SparseMat> solver;
                solver.analyzePattern(A);
                solver.factorize(A);
                bool solver_failed = solver.info() != Eigen::Success;

                // Solve
                dX = solver.solve(B);

                // Calculate the initial cost
                J0 = RESIDUAL.dot(RESIDUAL) + rprior.dot(Gpinv*rprior);

                // If solving is not successful, return false
                if (solver_failed || dX.hasNaN())
                {
                    printf(KRED"Failed to solve!\n"RESET);
                    cout << dX;
                    break;
                }

                // Cap the change
                if (dX.norm() > dXmax)
                    dX = dXmax*dX/dX.norm();

                // Back up the states
                StateWithCov Xpred_ = Xpred;

                // Update the state
                Xpred.boxplusd(dX);

                // Update the covariance
                MatrixNd G = Xpred_.Cov;                 
                G  = (I - (A + G.inverse()).toDense().inverse()*A)*G;
                // G = A.toDense().inverse();
                Xpred.SetCov(G);

                // Re-evaluate to update the cost
                EvaluateLidar(Xpred, Coef, RESIDUAL, JACOBIAN);
                JK = RESIDUAL.dot(RESIDUAL) + rprior.dot(Gpinv*rprior);

                double dJ = JK - J0;
                // Check the increase
                if (dJ > 0)
                {
                    printf(KRED "Cost increases. Revert.\n" RESET);
                    Xpred = Xpred_;
                    break;
                }

                // Visualization
                {
                    // Pointcloud deskewed
                    Util::publishCloud(cloudDskPub, *cloudDeskewedInW, ros::Time::now(), "world");

                    // Associated the associated points
                    CloudXYZIPtr assocCloud(new CloudXYZI());
                    assocCloud->resize(Coef.size());
                    for(int pidx = 0; pidx < Coef.size(); pidx++)
                    {
                        auto &point = assocCloud->points[pidx];
                        point.x = Coef[pidx].finW.x();
                        point.y = Coef[pidx].finW.y();
                        point.z = Coef[pidx].finW.z();
                    }
                    Util::publishCloud(assocPub, *assocCloud, ros::Time::now(), "world");

                    printf("LIDX: %d, CIDX: %d. ITER: %d. Time: %f. Features: %d. J: %f -> %f\n",
                            lidx, cidx, iidx, Xpred.tcurr, Nf, J0, JK);

                    printf("\t|dX| %6.3f. dR : %6.3f, %6.3f, %6.3f. dP : %6.3f, %6.3f, %6.3f. dW : %6.3f, %6.3f, %6.3f. dV : %6.3f, %6.3f, %6.3f\n",
                            dX.norm(), dX(0), dX(1), dX(2), dX(3), dX(4), dX(5), dX(6), dX(7), dX(8), dX(9), dX(10), dX(11));

                    printf("\tXpred: Pos: %6.3f, %6.3f, %6.3f. Rot: %6.3f, %6.3f, %6.3f. Omg: %6.3f, %6.3f, %6.3f. Vel: %6.3f, %6.3f, %6.3f.\n",
                            Xpred.Pos(0),    Xpred.Pos(1),    Xpred.Pos(2),
                            Xpred.YPR().x(), Xpred.YPR().y(), Xpred.YPR().z(),
                            Xpred.Omg(0),    Xpred.Omg(1),    Xpred.Omg(2),
                            Xpred.Vel(0),    Xpred.Vel(1),    Xpred.Vel(2));
                }
            }

            // Update the states
            Xhatprev = Xhat;
            Xhat = Xpred;

            // Publish the estimated pose for visualization
            if (poseprior->size() != Ncloud)
                poseprior->resize(Ncloud);

            PointPose pose = myTf(Xpred.Rot.matrix(), Xpred.Pos).Pose6D(Xpred.tcurr);
            poseprior->points[cidx] = pose;
            Util::publishCloud(pppub, *poseprior, ros::Time::now(), "world");

            // DEBUG: Break after n steps
            debug_count++;
            if (DEBUG_STEPS > 0 && debug_count >= DEBUG_STEPS)
                break;
        }
    }
};