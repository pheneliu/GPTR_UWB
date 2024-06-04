#pragma once

#include <iostream>
#include <algorithm>    // Include this header for std::max
#include <Eigen/Dense>

typedef Sophus::SO3<double> SO3d;
typedef Sophus::SE3<double> SE3d;
typedef Vector3d Vec3;

using namespace std;
using namespace Eigen;

template <typename MatrixType1, typename MatrixType2>
MatrixXd kron(const MatrixType1& A, const MatrixType2& B) {
    MatrixXd result(A.rows() * B.rows(), A.cols() * B.cols());
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            result.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
        }
    }
    return result;
}

Matrix3d SO3Jr(const Vector3d &phi)
{
    Matrix3d Jr;
    Sophus::rightJacobianSO3(phi, Jr);
    return Jr;
}

Matrix3d SO3JrInv(const Vector3d &phi)
{
    Matrix3d Jr_inv;
    Sophus::rightJacobianInvSO3(phi, Jr_inv);
    return Jr_inv;
}

// Define of the states for convenience in initialization and copying
class StateStamped
{
public:

    double t;
    SO3d R;
    Vec3 O;
    Vec3 P;
    Vec3 V;
    Vec3 A;

    // Destructor
    ~StateStamped(){};
    
    // Constructor
    StateStamped()
        : t(0), R(SO3d()), O(Vector3d(0, 0, 0)), P(Vector3d(0, 0, 0)), V(Vector3d(0, 0, 0)), A(Vector3d(0, 0, 0)) {}
    
    StateStamped(double t_)
        : t(t_), R(SO3d()), O(Vector3d(0, 0, 0)), P(Vector3d(0, 0, 0)), V(Vector3d(0, 0, 0)), A(Vector3d(0, 0, 0)) {}

    StateStamped(double t_, SO3d &R_, Vec3 &O_, Vec3 &P_, Vec3 &V_, Vec3 &A_)
        : t(t_), R(R_), O(O_), P(P_), V(V_), A(A_) {}

    StateStamped(const StateStamped &other)
        : t(other.t), R(other.R), O(other.O), P(other.P), V(other.V), A(other.A) {}

    StateStamped(double t_, const StateStamped &other)
        : t(t_), R(other.R), O(other.O), P(other.P), V(other.V), A(other.A) {}

    StateStamped &operator=(const StateStamped &Xother)
    {
        this->t = Xother.t;
        this->R = Xother.R;
        this->O = Xother.O;
        this->P = Xother.P;
        this->V = Xother.V;
        this->A = Xother.A;
        return *this;
    }

    StateStamped interpolate(double s, const StateStamped &Xb)
    {

    }
};

class GPMixer
{
private:

    double dt = 0.0;

public:

    // Destructor
   ~GPMixer() {};

    // Constructor
    GPMixer(double dt_) : dt(dt_) {};

    // Transition Matrix, PHI(tau, 0)
    MatrixXd TransMat(const double dtau, int N) const
    {
        std::function<int(int)> factorial = [&factorial](int n) -> int {return (n <= 1) ? 1 : n * factorial(n - 1);};

        MatrixXd Phi = MatrixXd::Identity(N, N);
        for(int n = 0; n < N; n++)
            for(int m = n + 1; m < N; m++)
                Phi(n, m) = pow(dtau, m-n)/factorial(m-n);

        return Phi;    
    }

    // Gaussian Process covariance, Q = \int{Phi*F*Qc*F'*Phi'}
    MatrixXd GPCov(const double dtau, int N) const 
    {
        std::function<int(int)> factorial = [&factorial](int n) -> int {return (n <= 1) ? 1 : n * factorial(n - 1);};
        
        MatrixXd Q(N, N);
        for(int n = 0; n < N; n++)
            for(int m = 0; m < N; m++)
                Q(n, m) = 1.0/(2*N-2-n-m+1)*pow(dtau, 2*N-2-n-m+1)/factorial(N-1-n)/factorial(N-1-m);

        return Q;    
    }

    // PSI in x(tau) = LAMBDA(tau)x(k-1) + PSI(tau)x*(k)
    MatrixXd PSI(const double dtau, int N) const 
    {
        return GPCov(dtau, N)*TransMat(dt - dtau, N).transpose()*GPCov(dt, N).inverse();
    }
    
    // PSI in x(tau) = LAMBDA(tau)x(k-1) + PSI(tau)x*(k)
    MatrixXd LAMBDA(const double dtau, int N) const
    {
        return TransMat(dtau, N) - PSI(dtau, N)*TransMat(dt, N);
    }

    MatrixXd TransMat_RO(const double dtau)  const { return kron(TransMat(dtau, 2), Matrix3d::Identity()); }
    MatrixXd GPCov_RO(const double dtau)     const { return kron(GPCov   (dtau, 2), Matrix3d::Identity()); }
    MatrixXd PSI_RO(const double dtau)       const { return kron(PSI     (dtau, 2), Matrix3d::Identity()); }
    MatrixXd LAMBDA_RO(const double dtau)    const { return kron(LAMBDA  (dtau, 2), Matrix3d::Identity()); }

    MatrixXd TransMat_PVA(const double dtau) const { return kron(TransMat(dtau, 3), Matrix3d::Identity()); }
    MatrixXd GPCov_PVA(const double dtau)    const { return kron(GPCov   (dtau, 3), Matrix3d::Identity()); }
    MatrixXd PSI_PVA(const double dtau)      const { return kron(PSI     (dtau, 3), Matrix3d::Identity()); }
    MatrixXd LAMBDA_PVA(const double dtau)   const { return kron(LAMBDA  (dtau, 3), Matrix3d::Identity()); }

    Matrix3d Jr(const Vector3d &phi) const
    {
        Matrix3d Jr;
        Sophus::rightJacobianSO3(phi, Jr);
        return Jr;
    }

    Matrix3d JrInv(const Vector3d &phi) const
    {
        Matrix3d JrInv;
        Sophus::rightJacobianInvSO3(phi, JrInv);
        return JrInv;
    }

    void MapParamToState(double const *const *parameters, int base, StateStamped &X) const
    {
        X.R = Eigen::Map<SO3d const>(parameters[base + 0]);
        X.O = Eigen::Map<Vec3 const>(parameters[base + 1]);
        X.P = Eigen::Map<Vec3 const>(parameters[base + 2]);
        X.V = Eigen::Map<Vec3 const>(parameters[base + 3]);
        X.A = Eigen::Map<Vec3 const>(parameters[base + 4]);
    }

    void ComputeXtAndDerivs(const StateStamped &Xa, const StateStamped &Xb, StateStamped &Xt,
                            vector<vector<Matrix3d>> &DXt_DXa, vector<vector<Matrix3d>> &DXt_DXb) const
    {
        // Map the variables of the state
        double tau = Xt.t; SO3d &Rt = Xt.R; Vec3 &Ot = Xt.O; Vec3 &Pt = Xt.P; Vec3 &Vt = Xt.V; Vec3 &At = Xt.A;
        
        // Calculate the the mixer matrixes
        MatrixXd LAM_ROt  = LAMBDA_RO(tau);
        MatrixXd PSI_ROt  = PSI_RO(tau);
        MatrixXd LAM_PVAt = LAMBDA_PVA(tau);
        MatrixXd PSI_PVAt = PSI_PVA(tau);

        // Extract the blocks of SO3 states
        Matrix3d LAM_RO11 = LAM_ROt.block<3, 3>(0, 0);
        Matrix3d LAM_RO12 = LAM_ROt.block<3, 3>(0, 3);

        Matrix3d LAM_RO21 = LAM_ROt.block<3, 3>(3, 0);
        Matrix3d LAM_RO22 = LAM_ROt.block<3, 3>(3, 3);

        Matrix3d PSI_RO11 = PSI_ROt.block<3, 3>(0, 0);
        Matrix3d PSI_RO12 = PSI_ROt.block<3, 3>(0, 3);

        Matrix3d PSI_RO21 = PSI_ROt.block<3, 3>(3, 0);
        Matrix3d PSI_RO22 = PSI_ROt.block<3, 3>(3, 3);

        // Extract the blocks of R3 states
        Matrix3d LAM_PVA11 = LAM_PVAt.block<3, 3>(0, 0);
        Matrix3d LAM_PVA12 = LAM_PVAt.block<3, 3>(0, 3);
        Matrix3d LAM_PVA13 = LAM_PVAt.block<3, 3>(0, 6);

        Matrix3d LAM_PVA21 = LAM_PVAt.block<3, 3>(3, 0);
        Matrix3d LAM_PVA22 = LAM_PVAt.block<3, 3>(3, 3);
        Matrix3d LAM_PVA23 = LAM_PVAt.block<3, 3>(3, 6);

        Matrix3d LAM_PVA31 = LAM_PVAt.block<3, 3>(6, 0);
        Matrix3d LAM_PVA32 = LAM_PVAt.block<3, 3>(6, 3);
        Matrix3d LAM_PVA33 = LAM_PVAt.block<3, 3>(6, 6);

        Matrix3d PSI_PVA11 = PSI_PVAt.block<3, 3>(0, 0);
        Matrix3d PSI_PVA12 = PSI_PVAt.block<3, 3>(0, 3);
        Matrix3d PSI_PVA13 = PSI_PVAt.block<3, 3>(0, 6);

        Matrix3d PSI_PVA21 = PSI_PVAt.block<3, 3>(3, 0);
        Matrix3d PSI_PVA22 = PSI_PVAt.block<3, 3>(3, 3);
        Matrix3d PSI_PVA23 = PSI_PVAt.block<3, 3>(3, 6);

        Matrix3d PSI_PVA31 = PSI_PVAt.block<3, 3>(6, 0);
        Matrix3d PSI_PVA32 = PSI_PVAt.block<3, 3>(6, 3);
        Matrix3d PSI_PVA33 = PSI_PVAt.block<3, 3>(6, 6);

        // Find the relative rotation 
        SO3d Rab = Xa.R.inverse()*Xb.R;

        // Calculate the SO3 knots in relative form
        Vec3 thetaa    = Vector3d(0, 0, 0);
        Vec3 thetadota = Xa.O;
        Vec3 thetab    = Rab.log();
        Vec3 thetadotb = JrInv(thetab)*Xb.O;
        // Put them in vector form
        Matrix<double, 6, 1> gammaa; gammaa << thetaa, thetadota;
        Matrix<double, 6, 1> gammab; gammab << thetab, thetadotb;

        // Calculate the knot euclid states and put them in vector form
        Matrix<double, 9, 1> pvaa; pvaa << Xa.P, Xa.V, Xa.A;
        Matrix<double, 9, 1> pvab; pvab << Xb.P, Xb.V, Xb.A;

        // Mix the knots to get the interpolated states
        Matrix<double, 6, 1> gammat = LAM_ROt*gammaa + PSI_ROt*gammab;
        Matrix<double, 9, 1> pvat   = LAM_PVAt*pvaa  + PSI_PVAt*pvab;

        // Retrive the interpolated SO3 in relative form
        Vector3d thetat    = gammat.block<3, 1>(0, 0);
        Vector3d thetadott = gammat.block<3, 1>(3, 0);

        // Assign the interpolated state
        Rt = Xa.R*SO3d::exp(thetat);
        Ot = JrInv(thetat)*thetadott;
        Pt = pvat.block<3, 1>(0, 0);
        Vt = pvat.block<3, 1>(3, 0);
        At = pvat.block<3, 1>(6, 0);

        // Calculate the Jacobian
        DXt_DXa = vector<vector<Matrix3d>>(5, vector<Matrix3d>(5, Matrix3d::Zero()));
        DXt_DXb = vector<vector<Matrix3d>>(5, vector<Matrix3d>(5, Matrix3d::Zero()));

        // Local index for the states in the state vector
        const int RIDX = 0;
        const int OIDX = 1;
        const int PIDX = 2;
        const int VIDX = 3;
        const int AIDX = 4;
        
        // DRt_DRa
        DXt_DXa[RIDX][RIDX] = SO3d::exp(-thetat).matrix() - Jr(thetat)*(PSI_RO11 - PSI_RO12*SO3d::hat(Xb.O))*JrInv(thetab)*Rab.inverse().matrix();
        // DRt_DOa
        DXt_DXa[RIDX][OIDX] = Jr(thetat)*LAM_RO12;
        
        // DRt_DPa DRt_DVa DRt_DAa are all zeros
        
        // TODO:
        // DOt_Ra still needs to be computed
        // DXt_DXa[OIDX][RIDX];
        // DOt_Oa still needs to be computed
        // DXt_DXa[OIDX][OIDX];
        
        // DOt_DPa DOt_DVa DOt_DAa are all zeros

        // DPt_DRa DPt_DOa are all all zeros
        // DPt_DPa
        DXt_DXa[PIDX][PIDX] = LAM_PVA11;
        // DPt_DVa
        DXt_DXa[PIDX][VIDX] = LAM_PVA12;
        // DPt_DAa
        DXt_DXa[PIDX][AIDX] = LAM_PVA13;
        
        // DVt_DPa
        DXt_DXa[VIDX][PIDX] = LAM_PVA21;
        // DVt_DVa
        DXt_DXa[VIDX][VIDX] = LAM_PVA22;
        // DVt_DAa
        DXt_DXa[VIDX][AIDX] = LAM_PVA23;

        // DAt_DPa
        DXt_DXa[AIDX][PIDX] = LAM_PVA31;
        // DAt_DVa
        DXt_DXa[AIDX][VIDX] = LAM_PVA32;
        // DAt_DAa
        DXt_DXa[AIDX][AIDX] = LAM_PVA33;




        // DRt_DRb
        DXt_DXb[RIDX][RIDX] = Jr(thetat)*(PSI_RO11 - PSI_RO12*SO3d::hat(Xb.O))*JrInv(thetab);
        // DRt_DOb
        DXt_DXb[RIDX][OIDX] = Jr(thetat)*PSI_RO12*JrInv(thetab);
        // DRt_DPb DRt_DVb DRt_DAb are all zeros

        // DRt_DPb DRt_DVb DRt_DAb are all zeros
        
        // TODO:
        // DOt_Rb still needs to be computed
        // DXt_DXb[OIDX][RIDX];
        // DOt_Ob still needs to be computed
        // DXt_DXb[OIDX][OIDX];
        
        // DOt_DPb DOt_DVb DOt_DAb are all zeros

        // DPt_DRb DPt_DOb are all all zeros
        // DPt_DPb
        DXt_DXb[PIDX][PIDX] = PSI_PVA11;
        // DRt_DPb
        DXt_DXb[PIDX][VIDX] = PSI_PVA12;
        // DRt_DAb
        DXt_DXb[PIDX][AIDX] = PSI_PVA13;

        // DVt_DPb
        DXt_DXb[VIDX][PIDX] = PSI_PVA21;
        // DVt_DVb
        DXt_DXb[VIDX][VIDX] = PSI_PVA22;
        // DVt_DAb
        DXt_DXb[VIDX][AIDX] = PSI_PVA23;
        
        // DAt_DPb
        DXt_DXb[AIDX][PIDX] = PSI_PVA21;
        // DAt_DVb
        DXt_DXb[AIDX][VIDX] = PSI_PVA22;
        // DAt_DAb
        DXt_DXb[AIDX][AIDX] = PSI_PVA23;
    }
};

class GaussianProcess
{
private:

    // Start time
    double t0 = 0;
    double dt = 0.0;

    GPMixer gpm;

    // State vector
    Eigen::aligned_deque<SO3d> R;
    Eigen::aligned_deque<Vec3> O;
    Eigen::aligned_deque<Vec3> P;
    Eigen::aligned_deque<Vec3> V;
    Eigen::aligned_deque<Vec3> A;

public:

    // Destructor
    ~GaussianProcess(){};

    // Constructor
    GaussianProcess(double dt_)
        : dt(dt_), gpm(dt_) {};

    double getMinTime()
    {
        return t0;
    }

    double getMaxTime()
    {
        return t0 + max(0, int(R.size()) - 1)*dt;
    }

    int getNumKnots()
    {
        return int(R.size());
    }

    double getKnotTime(int kidx)
    {
        return t0 + kidx*dt;
    }

    double getDt()
    {
        return dt;
    }

    bool TimeInInterval(double t, double eps=0.0)
    {
        return (t >= getMinTime() + eps && t <= getMaxTime() - eps);
    }

    pair<int, double> computeTimeIndex(double t)
    {
        int u = int((t - t0)/dt);
        double s = double(t - t0)/dt - u;
        return make_pair(u, s);
    }

    StateStamped getStateAt(double t)
    {
        // Find the index of the interval to find interpolation
        auto   us = computeTimeIndex(t);
        int    u  = us.first;
        double s  = us.second;
        assert(u <= getNumKnots() - 1);
        int ua = u;  
        int ub = u+1;

        // Extract the states of the two adjacent knots
        StateStamped Xa = StateStamped(t0 + ua*dt, R[ua], O[ua], P[ua], V[ua], A[ua]);
        StateStamped Xb = StateStamped(t0 + ub*dt, R[ub], O[ub], P[ub], V[ub], A[ub]);

        // Relative angle between two knots
        Vector3d thetaa = Vector3d(0, 0, 0);
        Vector3d thetab = (Xa.R.inverse() * Xb.R).log();

        Matrix<double, 6, 1> gammaa; gammaa << thetaa, Xa.O;
        Matrix<double, 6, 1> gammab; gammab << thetab, SO3JrInv(thetab)*Xb.O;

        Matrix<double, 9, 1> pvaa; pvaa << Xa.P, Xa.V, Xa.A;
        Matrix<double, 9, 1> pvab; pvab << Xb.P, Xb.V, Xb.A;
        
        Matrix<double, 6, 1> gammat; // Containing on-manifold states (rotation and angular velocity)
        Matrix<double, 9, 1> pvat;   // Position, velocity, acceleration

        gammat = gpm.LAMBDA_RO(s*dt)  * gammaa + gpm.PSI_RO(s*dt)  * gammab;
        pvat   = gpm.LAMBDA_PVA(s*dt) * pvaa   + gpm.PSI_PVA(s*dt) * pvab;

        SO3d Rt = SO3d::exp(gammat.block<3, 1>(0, 0));
        Vec3 Ot = gammat.block<3, 1>(3, 0);
        Vec3 Pt = pvat.block<3, 1>(0, 0);
        Vec3 Vt = pvat.block<3, 1>(3, 0);
        Vec3 At = pvat.block<3, 1>(6, 0);

        return StateStamped(t, Rt, Ot, Pt, Vt, At);
    }

    StateStamped getKnot(int kidx)
    {
        return StateStamped(getKnotTime(kidx), R[kidx], O[kidx], P[kidx], V[kidx], A[kidx]);
    }

    SE3d pose(double t)
    {
        StateStamped X = getStateAt(t);
        return SE3d(X.R, X.P);
    }

    inline SO3d &getKnotSO3(size_t kidx) { return R[kidx]; }
    inline Vec3 &getKnotOmg(size_t kidx) { return O[kidx]; }
    inline Vec3 &getKnotPos(size_t kidx) { return P[kidx]; }
    inline Vec3 &getKnotVel(size_t kidx) { return V[kidx]; }
    inline Vec3 &getKnotAcc(size_t kidx) { return A[kidx]; }

    void setStartTime(double t)
    {
        t0 = t;
        R  = {SO3d()};
        O  = {Vec3(0, 0, 0)};
        P  = {Vec3(0, 0, 0)};
        V  = {Vec3(0, 0, 0)};
        A  = {Vec3(0, 0, 0)};
    }

    void extendKnotsTo(double t, const StateStamped &Xn=StateStamped())
    {
        if(t0 == 0)
        {
            printf("MIN TIME HAS NOT BEEN INITIALIZED. "
                   "PLEASE CHECK, OR ELSE THE KNOT NUMBERS CAN BE VERY LARGE!");
            exit(-1);
        }
        
        // double tmax = getMaxTime();
        // if (tmax > t)
        //     return;

        // // Find the total number of knots at the new max time
        // int newknots = (t - t0 + dt - 1)/dt + 1;

        // Push the new state to the queue
        while(getMaxTime() < t)
        {
            R.push_back(Xn.R);
            O.push_back(Xn.O);
            P.push_back(Xn.P);
            V.push_back(Xn.V);
            A.push_back(Xn.A);
        }
    }

    void setKnot(int kidx, const StateStamped &Xn)
    {
        R[kidx] = Xn.R;
        O[kidx] = Xn.O;
        P[kidx] = Xn.P;
        V[kidx] = Xn.V;
        A[kidx] = Xn.A;
    }

    // Copy constructor
    GaussianProcess &operator=(const GaussianProcess &GPother)
    {
        this->R = GPother.R;
        this->O = GPother.O;
        this->P = GPother.P;
        this->V = GPother.V;
        this->A = GPother.A;        
        return *this;
    }
};