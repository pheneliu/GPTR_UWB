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