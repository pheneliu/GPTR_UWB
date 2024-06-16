#pragma once

#include <iostream>
#include <algorithm>    // Include this header for std::max
#include <Eigen/Dense>

typedef Sophus::SO3<double> SO3d;
typedef Sophus::SE3<double> SE3d;
typedef Vector3d Vec3;
typedef Matrix3d Mat3;

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

// Define of the states for convenience in initialization and copying
#define STATE_DIM 15
template <class T = double>
class StateStamped
{

public:

    using SO3T  = Sophus::SO3<T>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Mat3T = Eigen::Matrix<T, 3, 3>;

    double t;
    SO3T  R;
    Vec3T O;
    Vec3T P;
    Vec3T V;
    Vec3T A;

    // Destructor
    ~StateStamped(){};
    
    // Constructor
    StateStamped()
        : t(0), R(SO3T()), O(Vec3T(0, 0, 0)), P(Vec3T(0, 0, 0)), V(Vec3T(0, 0, 0)), A(Vec3T(0, 0, 0)) {}
    
    StateStamped(double t_)
        : t(t_), R(SO3T()), O(Vec3T()), P(Vec3T()), V(Vec3T()), A(Vec3T()) {}

    StateStamped(double t_, const SE3d &pose)
        : t(t_), R(pose.so3().cast<T>()), O(Vec3T(0, 0, 0)), P(pose.translation().cast<T>()), V(Vec3T(0, 0, 0)), A(Vec3T(0, 0, 0)) {}

    StateStamped(double t_, const SO3d &R_, const Vec3 &O_, const Vec3 &P_, const Vec3 &V_, const Vec3 &A_)
        : t(t_), R(R_.cast<T>()), O(O_.cast<T>()), P(P_.cast<T>()), V(V_.cast<T>()), A(A_.cast<T>()) {}

    StateStamped(const StateStamped<T> &other)
        : t(other.t), R(other.R), O(other.O), P(other.P), V(other.V), A(other.A) {}

    StateStamped(double t_, const StateStamped<T> &other)
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
                Q(n, m) = pow(dtau, 2*N-1-n-m)/double(2*N-1-n-m)/double(factorial(N-1-n))/double(factorial(N-1-m));
        // cout << "MyQ: " << Q << endl;
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

    MatrixXd TransMat_RO(const double dtau)  const { return kron(TransMat(dtau, 2), Mat3::Identity()); }
    MatrixXd GPCov_RO(const double dtau)     const { return kron(GPCov   (dtau, 2), Mat3::Identity()); }
    MatrixXd PSI_RO(const double dtau)       const { return kron(PSI     (dtau, 2), Mat3::Identity()); }
    MatrixXd LAMBDA_RO(const double dtau)    const { return kron(LAMBDA  (dtau, 2), Mat3::Identity()); }

    MatrixXd TransMat_PVA(const double dtau) const { return kron(TransMat(dtau, 3), Mat3::Identity()); }
    MatrixXd GPCov_PVA(const double dtau)    const { return kron(GPCov   (dtau, 3), Mat3::Identity()); }
    MatrixXd PSI_PVA(const double dtau)      const { return kron(PSI     (dtau, 3), Mat3::Identity()); }
    MatrixXd LAMBDA_PVA(const double dtau)   const { return kron(LAMBDA  (dtau, 3), Mat3::Identity()); }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> Jr(const Eigen::Matrix<T, 3, 1> &phi) const
    {
        Eigen::Matrix<T, 3, 3> Jr;
        Sophus::rightJacobianSO3(phi, Jr);
        return Jr;
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> JrInv(const Eigen::Matrix<T, 3, 1> &phi) const
    {
        Eigen::Matrix<T, 3, 3> JrInv;
        Sophus::rightJacobianInvSO3(phi, JrInv);
        return JrInv;
    }

    template <class T = double>
    void MapParamToState(T const *const *parameters, int base, StateStamped<T> &X) const
    {
        X.R = Eigen::Map<Sophus::SO3<T> const>(parameters[base + 0]);
        X.O = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 1]);
        X.P = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 2]);
        X.V = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 3]);
        X.A = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 4]);
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DJrXV_DX(Eigen::Matrix<T, 3, 1> X, const Eigen::Matrix<T, 3, 1> &V) const
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Xn = X.norm();

        if(Xn < 1e-5)
            return 0.5*SO3T::hat(V);

        // Extract the elements of input
        T tx = X(0); T ty = X(1); T tz = X(2);
        T vx = V(0); T vy = V(1); T vz = V(2);

        T Xnp2  = Xn*Xn;
        T Xnp3  = Xnp2*Xn;
        T Xnp4  = Xnp3*Xn;

        T sXn   = sin(Xn);
        // T sXnp2 = sXn*sXn;
        
        T cXn   = cos(Xn);
        // T cXnp2 = cXn*cXn;
        
        T gXn   = (1.0 - cXn)/Xnp2;
        T DgXn_DXn = sXn/Xnp2 - 2.0*(1.0 - cXn)/Xnp3;

        T hXn   = (Xn - sXn)/Xnp3;
        T DhXn_DXn = (1.0 - cXn)/Xnp3 - 3.0*(Xn - sXn)/Xnp4;

        Matrix<T, 1, 3> Xb = (X.transpose()/Xn);
        
        Mat3T DXskwsqV_DX;
        DXskwsqV_DX << ty*vy + tz*vz,     tx*vy - 2.0*ty*vx, tx*vz - 2.0*tz*vx,
                       ty*vx - 2.0*tx*vy, tx*vx + tz*vz,     ty*vz - 2.0*tz*vy,
                       tz*vx - 2.0*tx*vz, tz*vy - 2.0*ty*vz, tx*vx + ty*vy;

        return SO3T::hat(V)*gXn - SO3T::hat(X)*V*DgXn_DXn*Xb + DXskwsqV_DX*hXn + SO3T::hat(X)*SO3T::hat(X)*V*DhXn_DXn*Xb;
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DJrInvXV_DX(Eigen::Matrix<T, 3, 1> X, const Eigen::Matrix<T, 3, 1> &V) const
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Xn = X.norm();
        if(Xn < 1e-5)
            return -0.5*SO3T::hat(V);

        // Extract the elements of input
        T tx = X(0); T ty = X(1); T tz = X(2);
        T vx = V(0); T vy = V(1); T vz = V(2);

        T Xnp2  = Xn*Xn;
        T Xnp3  = Xnp2*Xn;
        
        T sXn   = sin(Xn);
        T sXnp2 = sXn*sXn;
        
        T cXn   = cos(Xn);
        // T cXnp2 = cXn*cXn;
        
        T gXn   = (1.0/Xnp2 - (1.0 + cXn)/(2.0*Xn*sXn));
        T DgXn_DXn = -2.0/Xnp3 + (Xn*sXnp2 + (sXn + Xn*cXn)*(1.0 + cXn))/(2.0*Xnp2*sXnp2);
        
        Mat3T DXskwsqV_DX;
        DXskwsqV_DX << ty*vy + tz*vz,     tx*vy - 2.0*ty*vx, tx*vz - 2.0*tz*vx,
                       ty*vx - 2.0*tx*vy, tx*vx + tz*vz,     ty*vz - 2.0*tz*vy,
                       tz*vx - 2.0*tx*vz, tz*vy - 2.0*ty*vz, tx*vx + ty*vy;

        return -0.5*SO3T::hat(V) + DXskwsqV_DX*gXn + SO3T::hat(X)*SO3T::hat(X)*V*DgXn_DXn*(X.transpose()/Xn);
    }

    template <class T = double>
    void ComputeXtAndDerivs(const StateStamped<T> &Xa,
                            const StateStamped<T> &Xb,
                                  StateStamped<T> &Xt,
                            vector<vector<Eigen::Matrix<T, 3, 3>>> &DXt_DXa,
                            vector<vector<Eigen::Matrix<T, 3, 3>>> &DXt_DXb,
                            Eigen::Matrix<T, 6, 1> &gammaa_,
                            Eigen::Matrix<T, 6, 1> &gammab_,
                            Eigen::Matrix<T, 6, 1> &gammat_
                           ) const
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        // Map the variables of the state
        double tau = Xt.t;
        SO3T   &Rt = Xt.R;
        Vec3T  &Ot = Xt.O;
        Vec3T  &Pt = Xt.P;
        Vec3T  &Vt = Xt.V;
        Vec3T  &At = Xt.A;
        
        // Calculate the the mixer matrixes
        Matrix<T, Dynamic, Dynamic> LAM_ROt  = LAMBDA_RO(tau).cast<T>();
        Matrix<T, Dynamic, Dynamic> PSI_ROt  = PSI_RO(tau).cast<T>();
        Matrix<T, Dynamic, Dynamic> LAM_PVAt = LAMBDA_PVA(tau).cast<T>();
        Matrix<T, Dynamic, Dynamic> PSI_PVAt = PSI_PVA(tau).cast<T>();

        // Extract the blocks of SO3 states
        Mat3T LAM_RO11 = LAM_ROt.block(0, 0, 3, 3);
        Mat3T LAM_RO12 = LAM_ROt.block(0, 3, 3, 3);

        Mat3T LAM_RO21 = LAM_ROt.block(3, 0, 3, 3);
        Mat3T LAM_RO22 = LAM_ROt.block(3, 3, 3, 3);

        Mat3T PSI_RO11 = PSI_ROt.block(0, 0, 3, 3);
        Mat3T PSI_RO12 = PSI_ROt.block(0, 3, 3, 3);

        Mat3T PSI_RO21 = PSI_ROt.block(3, 0, 3, 3);
        Mat3T PSI_RO22 = PSI_ROt.block(3, 3, 3, 3);

        // Extract the blocks of R3 states
        Mat3T LAM_PVA11 = LAM_PVAt.block(0, 0, 3, 3);
        Mat3T LAM_PVA12 = LAM_PVAt.block(0, 3, 3, 3);
        Mat3T LAM_PVA13 = LAM_PVAt.block(0, 6, 3, 3);

        Mat3T LAM_PVA21 = LAM_PVAt.block(3, 0, 3, 3);
        Mat3T LAM_PVA22 = LAM_PVAt.block(3, 3, 3, 3);
        Mat3T LAM_PVA23 = LAM_PVAt.block(3, 6, 3, 3);

        Mat3T LAM_PVA31 = LAM_PVAt.block(6, 0, 3, 3);
        Mat3T LAM_PVA32 = LAM_PVAt.block(6, 3, 3, 3);
        Mat3T LAM_PVA33 = LAM_PVAt.block(6, 6, 3, 3);

        Mat3T PSI_PVA11 = PSI_PVAt.block(0, 0, 3, 3);
        Mat3T PSI_PVA12 = PSI_PVAt.block(0, 3, 3, 3);
        Mat3T PSI_PVA13 = PSI_PVAt.block(0, 6, 3, 3);

        Mat3T PSI_PVA21 = PSI_PVAt.block(3, 0, 3, 3);
        Mat3T PSI_PVA22 = PSI_PVAt.block(3, 3, 3, 3);
        Mat3T PSI_PVA23 = PSI_PVAt.block(3, 6, 3, 3);

        Mat3T PSI_PVA31 = PSI_PVAt.block(6, 0, 3, 3);
        Mat3T PSI_PVA32 = PSI_PVAt.block(6, 3, 3, 3);
        Mat3T PSI_PVA33 = PSI_PVAt.block(6, 6, 3, 3);

        // Find the relative rotation 
        Sophus::SO3<T> Rab = Xa.R.inverse()*Xb.R;

        // Calculate the SO3 knots in relative form
        Eigen::Matrix<T, 3, 1> thetaa    = Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> thetadota = Xa.O;
        Eigen::Matrix<T, 3, 1> thetab    = Rab.log();
        Eigen::Matrix<T, 3, 1> thetadotb = JrInv(thetab)*Xb.O;
        // Put them in vector form
        Eigen::Matrix<T, 6, 1> gammaa; gammaa << thetaa, thetadota;
        Eigen::Matrix<T, 6, 1> gammab; gammab << thetab, thetadotb;

        // Calculate the knot euclid states and put them in vector form
        Eigen::Matrix<T, 9, 1> pvaa; pvaa << Xa.P, Xa.V, Xa.A;
        Eigen::Matrix<T, 9, 1> pvab; pvab << Xb.P, Xb.V, Xb.A;

        // Mix the knots to get the interpolated states
        Eigen::Matrix<T, 6, 1> gammat = LAM_ROt*gammaa + PSI_ROt*gammab;
        Eigen::Matrix<T, 9, 1> pvat   = LAM_PVAt*pvaa  + PSI_PVAt*pvab;

        // Retrive the interpolated SO3 in relative form
        Eigen::Matrix<T, 3, 1> thetat    = gammat.block(0, 0, 3, 1);
        Eigen::Matrix<T, 3, 1> thetadott = gammat.block(3, 0, 3, 1);

        // Assign the interpolated state
        Rt = Xa.R*Sophus::SO3<T>::exp(thetat);
        Ot = Jr(thetat)*thetadott;
        Pt = pvat.block(0, 0, 3, 1);
        Vt = pvat.block(3, 0, 3, 1);
        At = pvat.block(6, 0, 3, 1);

        // Calculate the Jacobian
        DXt_DXa = vector<vector<Eigen::Matrix<T, 3, 3>>>(5, vector<Eigen::Matrix<T, 3, 3>>(5, Eigen::Matrix<T, 3, 3>::Zero()));
        DXt_DXb = vector<vector<Eigen::Matrix<T, 3, 3>>>(5, vector<Eigen::Matrix<T, 3, 3>>(5, Eigen::Matrix<T, 3, 3>::Zero()));

        // Local index for the states in the state vector
        const int RIDX = 0;
        const int OIDX = 1;
        const int PIDX = 2;
        const int VIDX = 3;
        const int AIDX = 4;

        // Intermediaries to be reused

        Eigen::Matrix<T, 3, 3> Jrthetat = Jr(thetat);
        Eigen::Matrix<T, 3, 3> JrInvthetab = JrInv(thetab);

        Eigen::Matrix<T, 3, 3> Dthetat_Dthetab = (PSI_RO11 + PSI_RO12*DJrInvXV_DX(thetab, Xb.O));
        Eigen::Matrix<T, 3, 3> Dthetadott_Dthetab = (PSI_RO21 + PSI_RO22*DJrInvXV_DX(thetab, Xb.O));

        Eigen::Matrix<T, 3, 3> Dthetab_DRa = -JrInvthetab*Rab.inverse().matrix();
        Eigen::Matrix<T, 3, 3> Dthetat_DRa = Dthetat_Dthetab*Dthetab_DRa;

        Eigen::Matrix<T, 3, 3> Dthetab_DRb = JrInvthetab;
        Eigen::Matrix<T, 3, 3> Dthetat_DRb = Dthetat_Dthetab*Dthetab_DRb;

        Eigen::Matrix<T, 3, 3> Domgt_Dthetat = DJrXV_DX(thetat, thetadott);
        
        // DRt_DRa
        DXt_DXa[RIDX][RIDX] = Sophus::SO3<T>::exp(-thetat).matrix() + Jrthetat*Dthetat_DRa;
        // DRt_DOa
        DXt_DXa[RIDX][OIDX] = Jrthetat*LAM_RO12;
        // DRt_DPa DRt_DVa DRt_DAa are all zeros
        
        // DOt_Ra still needs to be computed
        DXt_DXa[OIDX][RIDX] = Domgt_Dthetat*Dthetat_DRa + Jrthetat*Dthetadott_Dthetab*Dthetab_DRa;
        // DOt_Oa still needs to be computed
        DXt_DXa[OIDX][OIDX] = Domgt_Dthetat*LAM_RO12 + Jrthetat*LAM_RO22;
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
        DXt_DXb[RIDX][RIDX] = Jrthetat*Dthetat_DRb;
        // DRt_DOb
        DXt_DXb[RIDX][OIDX] = Jrthetat*PSI_RO12*JrInvthetab;
        // DRt_DPb DRt_DVb DRt_DAb are all zeros
        
        // TODO:
        // DOt_Rb still needs to be computed
        DXt_DXb[OIDX][RIDX] = Domgt_Dthetat*Dthetat_DRb + Jrthetat*Dthetadott_Dthetab*Dthetab_DRb;
        // DOt_Ob still needs to be computed
        DXt_DXb[OIDX][OIDX] = Domgt_Dthetat*PSI_RO12*JrInvthetab + Jrthetat*PSI_RO22*JrInvthetab;
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
        DXt_DXb[AIDX][PIDX] = PSI_PVA31;
        // DAt_DVb
        DXt_DXb[AIDX][VIDX] = PSI_PVA32;
        // DAt_DAb
        DXt_DXb[AIDX][AIDX] = PSI_PVA33;

        gammaa_ = gammaa;
        gammab_ = gammab;
        gammat_ = gammat;
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

    Mat3 JrInv(const Vec3 &phi) const
    {
        Mat3 JrInv;
        Sophus::rightJacobianInvSO3(phi, JrInv);
        return JrInv;
    }

    StateStamped<double> getStateAt(double t)
    {
        // Find the index of the interval to find interpolation
        auto   us = computeTimeIndex(t);
        int    u  = us.first;
        double s  = us.second;

        int ua = u;  
        int ub = u+1;

        if (ub >= R.size() && fabs(1.0 - s) < 1e-5)
            return StateStamped(t0 + ua*dt, R[ua], O[ua], P[ua], V[ua], A[ua]);

        // Extract the states of the two adjacent knots
        StateStamped Xa = StateStamped(t0 + ua*dt, R[ua], O[ua], P[ua], V[ua], A[ua]);
        if (fabs(s) < 1e-5)
            return Xa;

        StateStamped Xb = StateStamped(t0 + ub*dt, R[ub], O[ub], P[ub], V[ub], A[ub]);
        if (fabs(1.0 - s) < 1e-5)
            return Xb;

        SO3d Rab = Xa.R.inverse()*Xb.R;

        // Relative angle between two knots
        Vec3 thetaa    = Vec3::Zero();
        Vec3 thetadota = Xa.O;
        Vec3 thetab    = Rab.log();
        Vec3 thetadotb = JrInv(thetab)*Xb.O;

        Eigen::Matrix<double, 6, 1> gammaa; gammaa << thetaa, thetadota;
        Eigen::Matrix<double, 6, 1> gammab; gammab << thetab, thetadotb;

        Eigen::Matrix<double, 9, 1> pvaa; pvaa << Xa.P, Xa.V, Xa.A;
        Eigen::Matrix<double, 9, 1> pvab; pvab << Xb.P, Xb.V, Xb.A;

        Eigen::Matrix<double, 6, 1> gammat; // Containing on-manifold states (rotation and angular velocity)
        Eigen::Matrix<double, 9, 1> pvat;   // Position, velocity, acceleration

        gammat = gpm.LAMBDA_RO(s*dt)  * gammaa + gpm.PSI_RO(s*dt)  * gammab;
        pvat   = gpm.LAMBDA_PVA(s*dt) * pvaa   + gpm.PSI_PVA(s*dt) * pvab;

        // Retrive the interpolated SO3 in relative form
        Vec3 thetat    = gammat.block(0, 0, 3, 1);
        Vec3 thetadott = gammat.block(3, 0, 3, 1);

        // Assign the interpolated state
        SO3d Rt = Xa.R*SO3d::exp(thetat);
        Vec3 Ot = gpm.Jr(thetat)*thetadott;
        Vec3 Pt = pvat.block<3, 1>(0, 0);
        Vec3 Vt = pvat.block<3, 1>(3, 0);
        Vec3 At = pvat.block<3, 1>(6, 0);

        return StateStamped<double>(t, Rt, Ot, Pt, Vt, At);
    }

    StateStamped<double> getKnot(int kidx)
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
        if (R.size() == 0)
        {
            R  = {SO3d()};
            O  = {Vec3(0, 0, 0)};
            P  = {Vec3(0, 0, 0)};
            V  = {Vec3(0, 0, 0)};
            A  = {Vec3(0, 0, 0)};
        }
    }

    void extendKnotsTo(double t, const StateStamped<double> &Xn=StateStamped())
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

    void extendOneKnot()
    {
        Matrix<double, 6, 1> gammac; gammac << Vector3d(0, 0, 0), O.back();
        Matrix<double, 6, 1> gamman = gpm.TransMat_RO(dt)*gammac;

        Matrix<double, 9, 1> pvac; pvac << P.back(), V.back(), A.back();
        Matrix<double, 9, 1> pvan = gpm.TransMat_PVA(dt)*pvac;

        Vec3 thetan = gamman.block<3, 1>(0, 0);
        Vec3 thetadotn = gamman.block<3, 1>(3, 0);

        SO3d Rn = R.back()*SO3d::exp(thetan);
        Vec3 On = gpm.Jr(thetan)*thetadotn;
        Vec3 Pn = pvan.block<3, 1>(0, 0);
        Vec3 Vn = pvan.block<3, 1>(3, 0);
        Vec3 An = pvan.block<3, 1>(6, 0);

        R.push_back(Rn);
        O.push_back(On);
        P.push_back(Pn);
        V.push_back(Vn);
        A.push_back(An);
    }

    void extendOneKnot(const StateStamped<double> &Xn)
    {
        R.push_back(Xn.R);
        O.push_back(Xn.O);
        P.push_back(Xn.P);
        V.push_back(Xn.V);
        A.push_back(Xn.A);
    }

    void setKnot(int kidx, const StateStamped<double> &Xn)
    {
        R[kidx] = Xn.R;
        O[kidx] = Xn.O;
        P[kidx] = Xn.P;
        V[kidx] = Xn.V;
        A[kidx] = Xn.A;
    }

    void updateKnot(int kidx, Matrix<double, STATE_DIM, 1> dX)
    {
        R[kidx] = R[kidx]*SO3d::exp(dX.block<3, 1>(0, 0));
        O[kidx] = O[kidx] + dX.block<3, 1>(3, 0);
        P[kidx] = P[kidx] + dX.block<3, 1>(6, 0);
        V[kidx] = V[kidx] + dX.block<3, 1>(9, 0);
        A[kidx] = A[kidx] + dX.block<3, 1>(12, 0);
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

typedef std::shared_ptr<GaussianProcess> GaussianProcessPtr;