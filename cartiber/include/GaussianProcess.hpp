#pragma once

#include <iostream>
#include <algorithm>    // Include this header for std::max
#include <Eigen/Dense>

#include <ceres/local_parameterization.h>   // For the local parameterization

// #define KNRM  "\x1B[0m"
// #define KRED  "\x1B[31m"
// #define KGRN  "\x1B[32m"
// #define KYEL  "\x1B[33m"
// #define KBLU  "\x1B[34m"
// #define KMAG  "\x1B[35m"
// #define KCYN  "\x1B[36m"
// #define KWHT  "\x1B[37m"
// #define RESET "\033[0m"

typedef Sophus::SO3<double> SO3d;
typedef Sophus::SE3<double> SE3d;
typedef Vector3d Vec3;
typedef Matrix3d Mat3;

using namespace std;
using namespace Eigen;

// Local parameterization when using ceres
template <class Groupd>
class GPSO3LocalParameterization : public ceres::LocalParameterization
{
public:
    virtual ~GPSO3LocalParameterization() {}

    using Tangentd = typename Groupd::Tangent;

    /// @brief plus operation for Ceres
    ///
    ///  T * exp(x)
    ///
    virtual bool Plus(double const *T_raw, double const *delta_raw,
                        double *T_plus_delta_raw) const
    {
        Eigen::Map<Groupd const> const T(T_raw);
        Eigen::Map<Tangentd const> const delta(delta_raw);
        Eigen::Map<Groupd> T_plus_delta(T_plus_delta_raw);
        T_plus_delta = T * Groupd::exp(delta);
        return true;
    }

    virtual bool ComputeJacobian(double const *T_raw,
                                    double *jacobian_raw) const
    {
        Eigen::Map<Groupd const> T(T_raw);
        Eigen::Map<Eigen::Matrix<double, Groupd::num_parameters, Groupd::DoF,
                                    Eigen::RowMajor>>
            jacobian(jacobian_raw);
        jacobian.setZero();
        jacobian(0, 0) = 1;
        jacobian(1, 1) = 1;
        jacobian(2, 2) = 1;
        return true;
    }

    ///@brief Global size
    virtual int GlobalSize() const { return Groupd::num_parameters; }

    ///@brief Local size
    virtual int LocalSize() const { return Groupd::DoF; }
};
typedef GPSO3LocalParameterization<SO3d> GPSO3dLocalParameterization;

// Define the states for convenience in initialization and copying
#define STATE_DIM 18
template <class T = double>
class GPState
{
public:

    using SO3T  = Sophus::SO3<T>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Mat3T = Eigen::Matrix<T, 3, 3>;

    double t;
    SO3T  R;
    Vec3T O;
    Vec3T S;
    Vec3T P;
    Vec3T V;
    Vec3T A;

    // Destructor
    ~GPState(){};
    
    // Constructor
    GPState()
        : t(0), R(SO3T()), O(Vec3T(0, 0, 0)), S(Vec3T(0, 0, 0)), P(Vec3T(0, 0, 0)), V(Vec3T(0, 0, 0)), A(Vec3T(0, 0, 0)) {}
    
    GPState(double t_)
        : t(t_), R(SO3T()), O(Vec3T()), S(Vec3T()), P(Vec3T()), V(Vec3T()), A(Vec3T()) {}

    GPState(double t_, const SE3d &pose)
        : t(t_), R(pose.so3().cast<T>()), O(Vec3T(0, 0, 0)), S(Vec3T(0, 0, 0)), P(pose.translation().cast<T>()), V(Vec3T(0, 0, 0)), A(Vec3T(0, 0, 0)) {}

    GPState(double t_, const SO3d &R_, const Vec3 &O_, const Vec3 &S_, const Vec3 &P_, const Vec3 &V_, const Vec3 &A_)
        : t(t_), R(R_.cast<T>()), O(O_.cast<T>()), S(S_.cast<T>()), P(P_.cast<T>()), V(V_.cast<T>()), A(A_.cast<T>()) {}

    GPState(const GPState<T> &other)
        : t(other.t), R(other.R), O(other.O), S(other.S), P(other.P), V(other.V), A(other.A) {}

    GPState(double t_, const GPState<T> &other)
        : t(t_), R(other.R), O(other.O), S(other.S), P(other.P), V(other.V), A(other.A) {}

    GPState &operator=(const GPState &Xother)
    {
        this->t = Xother.t;
        this->R = Xother.R;
        this->O = Xother.O;
        this->S = Xother.S;
        this->P = Xother.P;
        this->V = Xother.V;
        this->A = Xother.A;
        return *this;
    }

    Matrix<double, STATE_DIM, 1> boxminus(const GPState &Xother) const
    {
        Matrix<double, STATE_DIM, 1> dX;
        dX << (Xother.R.inverse()*R).log(),
               O - Xother.O,
               S - Xother.S,
               P - Xother.P,
               V - Xother.V,
               A - Xother.A;
        return dX;
    }

    double yaw()
    {
        Eigen::Vector3d n = R.matrix().col(0);
        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        return y / M_PI * 180.0;
    }

    double pitch()
    {
        Eigen::Vector3d n = R.matrix().col(0);
        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        return p / M_PI * 180.0;
    }

    double roll()
    {
        Eigen::Vector3d n = R.matrix().col(0);
        Eigen::Vector3d o = R.matrix().col(1);
        Eigen::Vector3d a = R.matrix().col(2);
        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        return r / M_PI * 180.0;
    }
};

// Utility for propagation and interpolation matrices, elementary jacobians dXt/dXk, J_r, H_r, Hprime_r ...
class GPMixer
{
private:

    double dt = 0.0;

public:

    // Destructor
   ~GPMixer() {};

    // Constructor
    GPMixer(double dt_) : dt(dt_) {};

    template <typename MatrixType1, typename MatrixType2>
    MatrixXd kron(const MatrixType1& A, const MatrixType2& B) const
    {
        MatrixXd result(A.rows() * B.rows(), A.cols() * B.cols());
        for (int i = 0; i < A.rows(); ++i)
            for (int j = 0; j < A.cols(); ++j)
                result.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;

        return result;
    }

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

    MatrixXd TransMat_ROS(const double dtau)  const { return kron(TransMat(dtau, 3), Mat3::Identity()); }
    MatrixXd GPCov_ROS(const double dtau)     const { return kron(GPCov   (dtau, 3), Mat3::Identity()); }
    MatrixXd PSI_ROS(const double dtau)       const { return kron(PSI     (dtau, 3), Mat3::Identity()); }
    MatrixXd LAM_ROS(const double dtau)       const { return kron(LAMBDA  (dtau, 3), Mat3::Identity()); }

    MatrixXd TransMat_PVA(const double dtau) const { return kron(TransMat(dtau, 3), Mat3::Identity()); }
    MatrixXd GPCov_PVA(const double dtau)    const { return kron(GPCov   (dtau, 3), Mat3::Identity()); }
    MatrixXd PSI_PVA(const double dtau)      const { return kron(PSI     (dtau, 3), Mat3::Identity()); }
    MatrixXd LAM_PVA(const double dtau)      const { return kron(LAMBDA  (dtau, 3), Mat3::Identity()); }

    template <class T = double>
    static Eigen::Matrix<T, 3, 3> Jr(const Eigen::Matrix<T, 3, 1> &phi)
    {
        Eigen::Matrix<T, 3, 3> Jr;
        Sophus::rightJacobianSO3(phi, Jr);
        return Jr;
    }

    template <class T = double>
    static Eigen::Matrix<T, 3, 3> JrInv(const Eigen::Matrix<T, 3, 1> &phi)
    {
        Eigen::Matrix<T, 3, 3> JrInv;
        Sophus::rightJacobianInvSO3(phi, JrInv);
        return JrInv;
    }

    template <class T = double>
    void MapParamToState(T const *const *parameters, int base, GPState<T> &X) const
    {
        X.R = Eigen::Map<Sophus::SO3<T> const>(parameters[base + 0]);
        X.O = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 1]);
        X.S = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 2]);
        X.P = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 3]);
        X.V = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 4]);
        X.A = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 5]);
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> Fu(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V) const
    {
        // Extract the elements of input
        T ux = U(0); T uy = U(1); T uz = U(2);
        T vx = V(0); T vy = V(1); T vz = V(2);

        Eigen::Matrix<T, 3, 3> Fu_;
        Fu_ << uy*vy +     uz*vz, ux*vy - 2.0*uy*vx, ux*vz - 2.0*uz*vx,
               uy*vx - 2.0*ux*vy, ux*vx +     uz*vz, uy*vz - 2.0*uz*vy,
               uz*vx - 2.0*ux*vz, uz*vy - 2.0*uy*vz, ux*vx +     uy*vy;
        return Fu_; 
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> Fv(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V) const
    {
        return Sophus::SO3<T>::hat(U)*Sophus::SO3<T>::hat(U);
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> Fuu(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &A) const
    {
        // Extract the elements of input
        // T ux = U(0); T uy = U(1); T uz = U(2);
        T vx = V(0); T vy = V(1); T vz = V(2);
        T ax = A(0); T ay = A(1); T az = A(2);

        Eigen::Matrix<T, 3, 3> Fuu_;
        Fuu_ << ay*vy +     az*vz, ax*vy - 2.0*ay*vx, ax*vz - 2.0*az*vx,
                ay*vx - 2.0*ax*vy, ax*vx +     az*vz, ay*vz - 2.0*az*vy,
                az*vx - 2.0*ax*vz, az*vy - 2.0*ay*vz, ax*vx +     ay*vy; 
        return Fuu_; 
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> Fuv(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &A) const
    {
        // Extract the elements of input
        T ux = U(0); T uy = U(1); T uz = U(2);
        // T vx = V(0); T vy = V(1); T vz = V(2);
        T ax = A(0); T ay = A(1); T az = A(2);

        Eigen::Matrix<T, 3, 3> Fuv_;
        Fuv_ << -2.0*ay*uy - 2.0*az*uz,      ax*uy +     ay*ux,      ax*uz +     az*ux,
                     ax*uy +     ay*ux, -2.0*ax*ux - 2.0*az*uz,      ay*uz +     az*uy,
                     ax*uz +     az*ux,      ay*uz +     az*uy, -2.0*ax*ux - 2.0*ay*uy;
        return Fuv_; 
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DJrXV_DX(const Eigen::Matrix<T, 3, 1> &X, const Eigen::Matrix<T, 3, 1> &V) const
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Xn = X.norm();

        if(Xn < 1e-4)
            return 0.5*SO3T::hat(V);

        T Xnp2 = Xn*Xn;
        T Xnp3 = Xnp2*Xn;
        T Xnp4 = Xnp3*Xn;

        T sXn = sin(Xn);
        // T sXnp2 = sXn*sXn;
        
        T cXn = cos(Xn);
        // T cXnp2 = cXn*cXn;
        
        T gXn = (1.0 - cXn)/Xnp2;
        T DgXn_DXn = sXn/Xnp2 - 2.0*(1.0 - cXn)/Xnp3;

        T hXn = (Xn - sXn)/Xnp3;
        T DhXn_DXn = (1.0 - cXn)/Xnp3 - 3.0*(Xn - sXn)/Xnp4;

        Vec3T Xb = X/Xn;
        
        Vec3T XsksqV = SO3T::hat(X)*SO3T::hat(X)*V;
        Mat3T DXsksqV_DX = Fu<T>(X, V);

        return SO3T::hat(V)*gXn - SO3T::hat(X)*V*DgXn_DXn*Xb.transpose() + DXsksqV_DX*hXn + XsksqV*DhXn_DXn*Xb.transpose();
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DDJrXVA_DXDX(const Eigen::Matrix<T, 3, 1> &X, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &A) const
    {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Xn = X.norm();

        if(Xn < 1e-4)
            return Fuu(X, V, A)/6.0;

        T Xnp2 = Xn*Xn;
        T Xnp3 = Xnp2*Xn;
        T Xnp4 = Xnp3*Xn;
        T Xnp5 = Xnp4*Xn;

        T sXn = sin(Xn);
        // T sXnp2 = sXn*sXn;
        
        T cXn = cos(Xn);
        // T cXnp2 = cXn*cXn;
        
        // T gXn = (1.0 - cXn)/Xnp2;
        T DgXn_DXn = sXn/Xnp2 - 2.0*(1.0 - cXn)/Xnp3;
        T DDgXn_DXnDXn = cXn/Xnp2 - 4.0*sXn/Xnp3 + 6.0*(1.0 - cXn)/Xnp4;

        T hXn = (Xn - sXn)/Xnp3;
        T DhXn_DXn = (1.0 - cXn)/Xnp3 - 3.0*(Xn - sXn)/Xnp4;
        T DDhXn_DXnDXn = 6.0/Xnp4 + sXn/Xnp3 + 6.0*cXn/Xnp4 - 12.0*sXn/Xnp5;

        Vec3T Xb = X/Xn;
        Mat3T DXb_DX = 1.0/Xn*(Mat3T::Identity(3, 3) - Xb*Xb.transpose());

        Vec3T XsksqV = SO3T::hat(X)*SO3T::hat(X)*V;
        Mat3T DXsksqV_DX = Fu(X, V);
        Mat3T DDXsksqVA_DXDX = Fuu(X, V, A);

        Mat3T Vsk = SO3T::hat(V);
        T AtpXb = A.transpose()*Xb;
        Eigen::Matrix<T, 1, 3> AtpDXb = A.transpose()*DXb_DX;

        return  Vsk*A*DgXn_DXn*Xb.transpose()

              + Vsk*AtpXb*DgXn_DXn
              + Vsk*X*AtpDXb*DgXn_DXn
              + Vsk*X*AtpXb*Xb.transpose()*DDgXn_DXnDXn

              + DDXsksqVA_DXDX*hXn
              + DXsksqV_DX*A*Xb.transpose()*DhXn_DXn

              + DXsksqV_DX*AtpXb*DhXn_DXn
              + XsksqV*AtpDXb*DhXn_DXn
              + XsksqV*AtpXb*Xb.transpose()*DDhXn_DXnDXn;
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DDJrXVA_DXDV(const Eigen::Matrix<T, 3, 1> &X, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &A) const
    {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Xn = X.norm();

        if(Xn < 1e-4)
            return -0.5*SO3T::hat(A);

        T Xnp2 = Xn*Xn;
        T Xnp3 = Xnp2*Xn;
        T Xnp4 = Xnp3*Xn;
        // T Xnp5 = Xnp4*Xn;

        T sXn = sin(Xn);
        // T sXnp2 = sXn*sXn;
        
        T cXn = cos(Xn);
        // T cXnp2 = cXn*cXn;
        
        T gXn = (1.0 - cXn)/Xnp2;
        T DgXn_DXn = sXn/Xnp2 - 2.0*(1.0 - cXn)/Xnp3;
        // T DDgXn_DXnDXn = cXn/Xnp2 - 4.0*sXn/Xnp3 + 6.0*(1.0 - cXn)/Xnp4;

        T hXn = (Xn - sXn)/Xnp3;
        T DhXn_DXn = (1.0 - cXn)/Xnp3 - 3.0*(Xn - sXn)/Xnp4;
        // T DDhXn_DXnDXn = 6.0/Xnp4 + sXn/Xnp3 + 6.0*cXn/Xnp4 - 12*sXn/Xnp5;

        Vec3T Xb = X/Xn;
        // Mat3T DXb_DX = 1.0/Xn*(Mat3T::Identity(3, 3) - Xb*Xb.transpose());

        Mat3T DXsksqV_DV = Fv(X, V);
        Mat3T DDXsksqVA_DXDV = Fuv(X, V, A);

        T AtpXb = A.transpose()*Xb;

        return -SO3T::hat(A)*gXn - SO3T::hat(X)*DgXn_DXn*AtpXb + DDXsksqVA_DXDV*hXn + DXsksqV_DV*DhXn_DXn*AtpXb;
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DJrInvXV_DX(const Eigen::Matrix<T, 3, 1> &X, const Eigen::Matrix<T, 3, 1> &V) const
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Xn = X.norm();
        if(Xn < 1e-4)
            return -0.5*SO3T::hat(V);

        T Xnp2 = Xn*Xn;
        T Xnp3 = Xnp2*Xn;
        
        T sXn = sin(Xn);
        T sXnp2 = sXn*sXn;
        
        T cXn = cos(Xn);
        // T cXnp2 = cXn*cXn;
        
        T gXn = (1.0/Xnp2 - (1.0 + cXn)/(2.0*Xn*sXn));
        T DgXn_DXn = -2.0/Xnp3 + (Xn*sXnp2 + (sXn + Xn*cXn)*(1.0 + cXn))/(2.0*Xnp2*sXnp2);

        Vec3T Xb = X/Xn;

        Vec3T XsksqV = SO3T::hat(X)*SO3T::hat(X)*V;
        Mat3T DXsksqV_DX = Fu(X, V);

        return -0.5*SO3T::hat(V) + DXsksqV_DX*gXn + XsksqV*DgXn_DXn*Xb.transpose();
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DDJrInvXVA_DXDX(const Eigen::Matrix<T, 3, 1> &X, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &A) const
    {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Xn = X.norm();

        if(Xn < 1e-4)
            return Fuu(X, V, A)/12.0;

        T Xnp2 = Xn*Xn;
        T Xnp3 = Xnp2*Xn;
        T Xnp4 = Xnp3*Xn;
        // T Xnp5 = Xnp4*Xn;

        T sXn = sin(Xn);
        T sXnp2 = sXn*sXn;
        T s2Xn = sin(2.0*Xn);
        
        T cXn = cos(Xn);
        // T cXnp2 = cXn*cXn;
        T c2Xn = cos(2.0*Xn);
        
        T gXn = (1.0/Xnp2 - (1.0 + cXn)/(2.0*Xn*sXn));
        T DgXn_DXn = -2.0/Xnp3 + (Xn*sXnp2 + (sXn + Xn*cXn)*(1.0 + cXn))/(2.0*Xnp2*sXnp2);
        T DDgXn_DXnDXn = (Xn + 6.0*s2Xn - 12.0*sXn - Xn*c2Xn + Xnp3*cXn + 2.0*Xnp2*sXn + Xnp3)/(Xnp4*(s2Xn - 2.0*sXn));

        Vec3T Xb = X/Xn;
        Mat3T DXb_DX = 1.0/Xn*(Mat3T::Identity(3, 3) - Xb*Xb.transpose());

        Vec3T XsksqV = SO3T::hat(X)*SO3T::hat(X)*V;
        Mat3T DXsksqV_DX = Fu(X, V);
        Mat3T DDXsksqVA_DXDX = Fuu(X, V, A);

        // Mat3T Vsk = SO3T::hat(V);
        T AtpXb = A.transpose()*Xb;
        Eigen::Matrix<T, 1, 3> AtpDXb = A.transpose()*DXb_DX;

        return   DDXsksqVA_DXDX*gXn
               + DXsksqV_DX*A*Xb.transpose()*DgXn_DXn

               + DXsksqV_DX*AtpXb*DgXn_DXn
               + XsksqV*AtpDXb*DgXn_DXn
               + XsksqV*AtpXb*Xb.transpose()*DDgXn_DXnDXn;
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DDJrInvXVA_DXDV(const Eigen::Matrix<T, 3, 1> &X, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &A) const
    {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Xn = X.norm();

        if(Xn < 1e-4)
            return 0.5*SO3T::hat(A);

        T Xnp2 = Xn*Xn;
        T Xnp3 = Xnp2*Xn;
        // T Xnp4 = Xnp3*Xn;
        // T Xnp5 = Xnp4*Xn;

        T sXn = sin(Xn);
        T sXnp2 = sXn*sXn;
        // T s2Xn = sin(2*Xn);
        
        T cXn = cos(Xn);
        // T cXnp2 = cXn*cXn;
        // T c2Xn = cos(2*Xn);
        
        T gXn = (1.0/Xnp2 - (1.0 + cXn)/(2.0*Xn*sXn));
        T DgXn_DXn = -2.0/Xnp3 + (Xn*sXnp2 + (sXn + Xn*cXn)*(1.0 + cXn))/(2.0*Xnp2*sXnp2);
        // T DDgXn_DXnDXn = (Xn + 6.0*s2Xn - 12.0*sXn - Xn*c2Xn + Xnp3*cXn + 2.0*Xnp2*sXn + Xnp3)/(Xnp4*(s2Xn - 2.0*sXn));

        Vec3T Xb = X/Xn;
        // Mat3T DXb_DX = 1.0/Xn*(Mat3T::Identity(3, 3) - Xb*Xb.transpose());

        Mat3T DXsksqV_DV = Fv(X, V);
        // Mat3T DXsksqV_DX = Fu(X, V);
        Mat3T DDXsksqVA_DXDV = Fuv(X, V, A);

        T AtpXb = A.transpose()*Xb;

        return 0.5*SO3T::hat(A) + DDXsksqVA_DXDV*gXn + DXsksqV_DV*DgXn_DXn*AtpXb;
    }

    template <class T = double>
    void ComputeXtAndJacobians(const GPState<T> &Xa,
                               const GPState<T> &Xb,
                                     GPState<T> &Xt,
                               vector<vector<Eigen::Matrix<T, 3, 3>>> &DXt_DXa,
                               vector<vector<Eigen::Matrix<T, 3, 3>>> &DXt_DXb,
                               Eigen::Matrix<T, 9, 1> &gammaa_,
                               Eigen::Matrix<T, 9, 1> &gammab_,
                               Eigen::Matrix<T, 9, 1> &gammat_
                              ) const
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Vec6T = Eigen::Matrix<T, 6, 1>;
        using Vec9T = Eigen::Matrix<T, 9, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        // Map the variables of the state
        double tau = Xt.t;
        SO3T   &Rt = Xt.R;
        Vec3T  &Ot = Xt.O;
        Vec3T  &St = Xt.S;
        Vec3T  &Pt = Xt.P;
        Vec3T  &Vt = Xt.V;
        Vec3T  &At = Xt.A;
        
        // Calculate the the mixer matrixes
        Matrix<T, Dynamic, Dynamic> LAM_ROSt = LAM_ROS(tau).cast<T>();
        Matrix<T, Dynamic, Dynamic> PSI_ROSt = PSI_ROS(tau).cast<T>();
        Matrix<T, Dynamic, Dynamic> LAM_PVAt = LAM_PVA(tau).cast<T>();
        Matrix<T, Dynamic, Dynamic> PSI_PVAt = PSI_PVA(tau).cast<T>();

        // Extract the blocks of SO3 states
        Mat3T LAM_ROS11 = LAM_ROSt.block(0, 0, 3, 3); Mat3T LAM_ROS12 = LAM_ROSt.block(0, 3, 3, 3); Mat3T LAM_ROS13 = LAM_ROSt.block(0, 6, 3, 3);
        Mat3T LAM_ROS21 = LAM_ROSt.block(3, 0, 3, 3); Mat3T LAM_ROS22 = LAM_ROSt.block(3, 3, 3, 3); Mat3T LAM_ROS23 = LAM_ROSt.block(3, 6, 3, 3);
        Mat3T LAM_ROS31 = LAM_ROSt.block(6, 0, 3, 3); Mat3T LAM_ROS32 = LAM_ROSt.block(6, 3, 3, 3); Mat3T LAM_ROS33 = LAM_ROSt.block(6, 6, 3, 3);
        
        Mat3T PSI_ROS11 = PSI_ROSt.block(0, 0, 3, 3); Mat3T PSI_ROS12 = PSI_ROSt.block(0, 3, 3, 3); Mat3T PSI_ROS13 = PSI_ROSt.block(0, 6, 3, 3);
        Mat3T PSI_ROS21 = PSI_ROSt.block(3, 0, 3, 3); Mat3T PSI_ROS22 = PSI_ROSt.block(3, 3, 3, 3); Mat3T PSI_ROS23 = PSI_ROSt.block(3, 6, 3, 3);
        Mat3T PSI_ROS31 = PSI_ROSt.block(6, 0, 3, 3); Mat3T PSI_ROS32 = PSI_ROSt.block(6, 3, 3, 3); Mat3T PSI_ROS33 = PSI_ROSt.block(6, 6, 3, 3);

        // Extract the blocks of R3 states
        Mat3T LAM_PVA11 = LAM_PVAt.block(0, 0, 3, 3); Mat3T LAM_PVA12 = LAM_PVAt.block(0, 3, 3, 3); Mat3T LAM_PVA13 = LAM_PVAt.block(0, 6, 3, 3);
        Mat3T LAM_PVA21 = LAM_PVAt.block(3, 0, 3, 3); Mat3T LAM_PVA22 = LAM_PVAt.block(3, 3, 3, 3); Mat3T LAM_PVA23 = LAM_PVAt.block(3, 6, 3, 3);
        Mat3T LAM_PVA31 = LAM_PVAt.block(6, 0, 3, 3); Mat3T LAM_PVA32 = LAM_PVAt.block(6, 3, 3, 3); Mat3T LAM_PVA33 = LAM_PVAt.block(6, 6, 3, 3);

        Mat3T PSI_PVA11 = PSI_PVAt.block(0, 0, 3, 3); Mat3T PSI_PVA12 = PSI_PVAt.block(0, 3, 3, 3); Mat3T PSI_PVA13 = PSI_PVAt.block(0, 6, 3, 3);
        Mat3T PSI_PVA21 = PSI_PVAt.block(3, 0, 3, 3); Mat3T PSI_PVA22 = PSI_PVAt.block(3, 3, 3, 3); Mat3T PSI_PVA23 = PSI_PVAt.block(3, 6, 3, 3);
        Mat3T PSI_PVA31 = PSI_PVAt.block(6, 0, 3, 3); Mat3T PSI_PVA32 = PSI_PVAt.block(6, 3, 3, 3); Mat3T PSI_PVA33 = PSI_PVAt.block(6, 6, 3, 3);

        // Find the relative rotation 
        Sophus::SO3<T> Rab = Xa.R.inverse()*Xb.R;

        // Calculate the SO3 knots in relative form
        Vec3T Thea     = Vec3T::Zero();
        Vec3T Thedota  = Xa.O;
        Vec3T Theddota = Xa.S;

        Vec3T Theb     = Rab.log();
        Vec3T Thedotb  = JrInv(Theb)*Xb.O;
        Vec3T Theddotb = JrInv(Theb)*Xb.S + DJrInvXV_DX(Theb, Xb.O)*Thedotb;

        // Put them in vector form
        Vec9T gammaa; gammaa << Thea, Thedota, Theddota;
        Vec9T gammab; gammab << Theb, Thedotb, Theddotb;

        // Calculate the knot euclid states and put them in vector form
        Vec9T pvaa; pvaa << Xa.P, Xa.V, Xa.A;
        Vec9T pvab; pvab << Xb.P, Xb.V, Xb.A;

        // Mix the knots to get the interpolated states
        Vec9T gammat = LAM_ROSt*gammaa + PSI_ROSt*gammab;
        Vec9T pvat   = LAM_PVAt*pvaa   + PSI_PVAt*pvab;

        // Retrive the interpolated SO3 in relative form
        Vec3T Thet     = gammat.block(0, 0, 3, 1);
        Vec3T Thedott  = gammat.block(3, 0, 3, 1);
        Vec3T Theddott = gammat.block(6, 0, 3, 1);

        // Assign the interpolated state
        Rt = Xa.R*Sophus::SO3<T>::exp(Thet);
        Ot = Jr(Thet)*Thedott;
        St = Jr(Thet)*Theddott + DJrXV_DX(Thet, Thedott)*Thedott;
        Pt = pvat.block(0, 0, 3, 1);
        Vt = pvat.block(3, 0, 3, 1);
        At = pvat.block(6, 0, 3, 1);

        // Calculate the Jacobian
        DXt_DXa = vector<vector<Mat3T>>(6, vector<Mat3T>(6, Mat3T::Zero()));
        DXt_DXb = vector<vector<Mat3T>>(6, vector<Mat3T>(6, Mat3T::Zero()));

        // Local index for the states in the state vector
        const int RIDX = 0;
        const int OIDX = 1;
        const int SIDX = 2;
        const int PIDX = 3;
        const int VIDX = 4;
        const int AIDX = 5;

        Mat3T JrInvTheb =  JrInv(Theb);
        Mat3T DTheb_DRa = -JrInvTheb*Rab.inverse().matrix();
        Mat3T DTheb_DRb =  JrInvTheb;

        // Intermediaries to be reused
        Mat3T DJrInvThebOb_DTheb = DJrInvXV_DX(Theb, Xb.O);
        Mat3T DJrInvThebSb_DTheb = DJrInvXV_DX(Theb, Xb.S);
        Mat3T DDJrInvThebObThedotb_DThebDTheb = DDJrInvXVA_DXDX(Theb, Xb.O, Thedotb);
        Mat3T DDJrInvThebObThedotb_DThebDOb = DDJrInvXVA_DXDV(Theb, Xb.O, Thedotb);
        Mat3T DTheddotb_DTheb = DJrInvThebSb_DTheb + DDJrInvThebObThedotb_DThebDTheb + DJrInvThebOb_DTheb*DJrInvThebOb_DTheb;

        Mat3T DJrThetTheddott_DThet = DJrXV_DX(Thet, Theddott);
        Mat3T DDJrThetThedottThedott_DThetDThet = DDJrXVA_DXDX(Thet, Thedott, Thedott);
        Mat3T DDJrThetThedottThedott_DThetDThedott = DDJrXVA_DXDV(Thet, Thedott, Thedott) + DJrXV_DX(Thet, Thedott);

        Mat3T DThet_DTheb = PSI_ROS11 + PSI_ROS12*DJrInvThebOb_DTheb + PSI_ROS13*DTheddotb_DTheb;
        Mat3T DThedott_DTheb = PSI_ROS21 + PSI_ROS22*DJrInvThebOb_DTheb + PSI_ROS23*DTheddotb_DTheb;
        Mat3T DTheddott_DTheb = PSI_ROS31 + PSI_ROS32*DJrInvThebOb_DTheb + PSI_ROS33*DTheddotb_DTheb;
        // Mat3T DJrThetTheddot_DThet = DJrXV_DX(Thet, Thet);

        Mat3T DTheddotb_DOb = DDJrInvThebObThedotb_DThebDOb + DJrInvThebOb_DTheb*JrInvTheb;

        // Dependance of thetat on the states
        Mat3T DThet_DRa = DThet_DTheb*DTheb_DRa;
        Mat3T DThet_DRb = DThet_DTheb*DTheb_DRb;
        Mat3T DThet_DOa = LAM_ROS12;
        Mat3T DThet_DOb = PSI_ROS12*JrInvTheb + PSI_ROS13*DTheddotb_DOb;
        Mat3T DThet_DSa = LAM_ROS13;
        Mat3T DThet_DSb = PSI_ROS13*JrInvTheb;

        Mat3T DThedott_DRa = DThedott_DTheb*DTheb_DRa;
        Mat3T DThedott_DRb = DThedott_DTheb*DTheb_DRb;
        Mat3T DThedott_DOa = LAM_ROS22;
        Mat3T DThedott_DOb = PSI_ROS22*JrInvTheb + PSI_ROS23*DTheddotb_DOb;
        Mat3T DThedott_DSa = LAM_ROS23;
        Mat3T DThedott_DSb = PSI_ROS23*JrInvTheb;

        Mat3T DTheddott_DRa = DTheddott_DTheb*DTheb_DRa;
        Mat3T DTheddott_DRb = DTheddott_DTheb*DTheb_DRb;
        Mat3T DTheddott_DOa = LAM_ROS32;
        Mat3T DTheddott_DOb = PSI_ROS32*JrInvTheb + PSI_ROS33*DTheddotb_DOb;
        Mat3T DTheddott_DSa = LAM_ROS33;
        Mat3T DTheddott_DSb = PSI_ROS33*JrInvTheb;

        Mat3T JrThet = Jr(Thet);
        // Mat3T JrInvThet = JrInv(Thet);

        // // Dependance of Ot on the states
        Mat3T DJrThetThedott_DThet = DJrXV_DX(Thet, Thedott);
        // Mat3T DOt_DThedott = JrThet;
        
        // DRt_DRa
        DXt_DXa[RIDX][RIDX] = Sophus::SO3<T>::exp(-Thet).matrix() + JrThet*DThet_DRa;
        // DRt_DOa
        DXt_DXa[RIDX][OIDX] = JrThet*DThet_DOa;
        // DRt_DSa
        DXt_DXa[RIDX][SIDX] = JrThet*DThet_DSa;
        // DRt_DPa DRt_DVa DRt_DAa are all zeros
        
        // DOt_Ra
        DXt_DXa[OIDX][RIDX] = DJrThetThedott_DThet*DThet_DRa + JrThet*DThedott_DRa;
        // DOt_Oa
        DXt_DXa[OIDX][OIDX] = DJrThetThedott_DThet*DThet_DOa + JrThet*DThedott_DOa;
        // DOt_Sa
        DXt_DXa[OIDX][SIDX] = DJrThetThedott_DThet*DThet_DSa + JrThet*DThedott_DSa;
        // DOt_DPa DOt_DVa DOt_DAa are all zeros

        // DSt_Ra
        DXt_DXa[SIDX][RIDX] = DJrThetTheddott_DThet*DThet_DRa + JrThet*DTheddott_DRa + DDJrThetThedottThedott_DThetDThet*DThet_DRa + DDJrThetThedottThedott_DThetDThedott*DThedott_DRa;
        // DSt_Oa
        DXt_DXa[SIDX][OIDX] = DJrThetTheddott_DThet*DThet_DOa + JrThet*DTheddott_DOa + DDJrThetThedottThedott_DThetDThet*DThet_DOa + DDJrThetThedottThedott_DThetDThedott*DThedott_DOa;
        // DSt_Sa
        DXt_DXa[SIDX][SIDX] = DJrThetTheddott_DThet*DThet_DSa + JrThet*DTheddott_DSa + DDJrThetThedottThedott_DThetDThet*DThet_DSa + DDJrThetThedottThedott_DThetDThedott*DThedott_DSa;
        // DSt_DPa DSt_DVa DSt_DAa are all zeros


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
        DXt_DXb[RIDX][RIDX] = JrThet*DThet_DRb;
        // DRt_DOb
        DXt_DXb[RIDX][OIDX] = JrThet*DThet_DOb;
        // DRt_DSb
        DXt_DXb[RIDX][SIDX] = JrThet*DThet_DSb;
        // DRt_DPb DRt_DVb DRt_DAb are all zeros
        
        // DOt_Rb
        DXt_DXb[OIDX][RIDX] = DJrThetThedott_DThet*DThet_DRb + JrThet*DThedott_DRb;
        // DOt_Ob
        DXt_DXb[OIDX][OIDX] = DJrThetThedott_DThet*DThet_DOb + JrThet*DThedott_DOb;
        // DOt_Sb
        DXt_DXb[OIDX][SIDX] = DJrThetThedott_DThet*DThet_DSb + JrThet*DThedott_DSb;
        // DOt_DPb DOt_DVb DOt_DAb are all zeros

        // DSt_Rb
        DXt_DXb[SIDX][RIDX] = (DJrThetTheddott_DThet + DDJrThetThedottThedott_DThetDThet)*DThet_DRb + DDJrThetThedottThedott_DThetDThedott*DThedott_DRb + JrThet*DTheddott_DRb;
        // DSt_Ob
        DXt_DXb[SIDX][OIDX] = (DJrThetTheddott_DThet + DDJrThetThedottThedott_DThetDThet)*DThet_DOb + DDJrThetThedottThedott_DThetDThedott*DThedott_DOb + JrThet*DTheddott_DOb;
        // DSt_Sb
        DXt_DXb[SIDX][SIDX] = (DJrThetTheddott_DThet + DDJrThetThedottThedott_DThetDThet)*DThet_DSb + DDJrThetThedottThedott_DThetDThedott*DThedott_DSb + JrThet*DTheddott_DSb;
        // DSt_DPb DSt_DVb DSt_DAb are all zeros


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

// Managing control points: cration, extension, queries, ...
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
    Eigen::aligned_deque<Vec3> S;
    Eigen::aligned_deque<Vec3> P;
    Eigen::aligned_deque<Vec3> V;
    Eigen::aligned_deque<Vec3> A;

public:

    // Destructor
    ~GaussianProcess(){};

    // Constructor
    GaussianProcess(double dt_)
        : dt(dt_), gpm(GPMixer(dt_)) {};

    double getMinTime() const
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

    double getDt() const
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

    GPState<double> getStateAt(double t)
    {
        // Find the index of the interval to find interpolation
        auto   us = computeTimeIndex(t);
        int    u  = us.first;
        double s  = us.second;

        int ua = u;  
        int ub = u+1;

        if (ub >= R.size() && fabs(1.0 - s) < 1e-5)
        {
            // printf(KYEL "Boundary issue: ub: %d, Rsz: %d, s: %f, 1-s: %f\n" RESET, ub, R.size(), s, fabs(1.0 - s));
            return GPState(t0 + ua*dt, R[ua], O[ua], S[ua], P[ua], V[ua], A[ua]);
        }

        // Extract the states of the two adjacent knots
        GPState Xa = GPState(t0 + ua*dt, R[ua], O[ua], S[ua], P[ua], V[ua], A[ua]);
        if (fabs(s) < 1e-5)
        {
            // printf(KYEL "Boundary issue: ub: %d, Rsz: %d, s: %f, 1-s: %f\n" RESET, ub, R.size(), s, fabs(1.0 - s));
            return Xa;
        }

        GPState Xb = GPState(t0 + ub*dt, R[ub], O[ub], S[ua], P[ub], V[ub], A[ub]);
        if (fabs(1.0 - s) < 1e-5)
        {
            // printf(KYEL "Boundary issue: ub: %d, Rsz: %d, s: %f, 1-s: %f\n" RESET, ub, R.size(), s, fabs(1.0 - s));
            return Xb;
        }

        SO3d Rab = Xa.R.inverse()*Xb.R;

        // Relative angle between two knots
        Vec3 Thea     = Vec3::Zero();
        Vec3 Thedota  = Xa.O;
        Vec3 Theddota = Xa.S;

        Vec3 Theb     = Rab.log();
        Vec3 Thedotb  = gpm.JrInv(Theb)*Xb.O;
        Vec3 Theddotb = gpm.JrInv(Theb)*Xb.S + gpm.DJrInvXV_DX(Theb, Xb.O)*Thedotb;

        Eigen::Matrix<double, 9, 1> gammaa; gammaa << Thea, Thedota, Theddota;
        Eigen::Matrix<double, 9, 1> gammab; gammab << Theb, Thedotb, Theddotb;

        Eigen::Matrix<double, 9, 1> pvaa; pvaa << Xa.P, Xa.V, Xa.A;
        Eigen::Matrix<double, 9, 1> pvab; pvab << Xb.P, Xb.V, Xb.A;

        Eigen::Matrix<double, 9, 1> gammat; // Containing on-manifold states (rotation and angular velocity)
        Eigen::Matrix<double, 9, 1> pvat;   // Position, velocity, acceleration

        gammat = gpm.LAM_ROS(s*dt) * gammaa + gpm.PSI_ROS(s*dt) * gammab;
        pvat   = gpm.LAM_PVA(s*dt) * pvaa   + gpm.PSI_PVA(s*dt) * pvab;

        // Retrive the interpolated SO3 in relative form
        Vec3 Thet     = gammat.block(0, 0, 3, 1);
        Vec3 Thedott  = gammat.block(3, 0, 3, 1);
        Vec3 Theddott = gammat.block(6, 0, 3, 1);

        // Assign the interpolated state
        SO3d Rt = Xa.R*SO3d::exp(Thet);
        Vec3 Ot = gpm.Jr(Thet)*Thedott;
        Vec3 St = gpm.Jr(Thet)*Theddott + gpm.DJrXV_DX(Thet, Thedott)*Thedott;
        Vec3 Pt = pvat.block<3, 1>(0, 0);
        Vec3 Vt = pvat.block<3, 1>(3, 0);
        Vec3 At = pvat.block<3, 1>(6, 0);

        return GPState<double>(t, Rt, Ot, St, Pt, Vt, At);
    }

    GPState<double> getKnot(int kidx)
    {
        return GPState(getKnotTime(kidx), R[kidx], O[kidx], S[kidx], P[kidx], V[kidx], A[kidx]);
    }

    SE3d pose(double t)
    {
        GPState X = getStateAt(t);
        return SE3d(X.R, X.P);
    }

    inline SO3d &getKnotSO3(size_t kidx) { return R[kidx]; }
    inline Vec3 &getKnotOmg(size_t kidx) { return O[kidx]; }
    inline Vec3 &getKnotAlp(size_t kidx) { return S[kidx]; }
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
            S  = {Vec3(0, 0, 0)};
            P  = {Vec3(0, 0, 0)};
            V  = {Vec3(0, 0, 0)};
            A  = {Vec3(0, 0, 0)};
        }
    }

    void extendKnotsTo(double t, const GPState<double> &Xn=GPState())
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
            S.push_back(Xn.S);
            P.push_back(Xn.P);
            V.push_back(Xn.V);
            A.push_back(Xn.A);
        }
    }

    void extendOneKnot()
    {
        SO3d Rc = R.back();
        Vec3 Oc = O.back();
        Vec3 Sc = S.back();
        Vec3 Pc = P.back();
        Vec3 Vc = V.back();
        Vec3 Ac = A.back();

        SO3d Rn = Rc*SO3d::exp(dt*Oc + 0.5*dt*dt*Sc);
        Vec3 On = Oc + dt*Sc;
        Vec3 Sn = Sc;
        Vec3 Pn = Pc + dt*Vc + 0.5*dt*dt*Ac;
        Vec3 Vn = Vc + dt*Ac;
        Vec3 An = Ac;

        R.push_back(Rn);
        O.push_back(On);
        S.push_back(Sn);
        P.push_back(Pn);
        V.push_back(Vn);
        A.push_back(An);
    }

    void extendOneKnot(const GPState<double> &Xn)
    {
        R.push_back(Xn.R);
        O.push_back(Xn.O);
        S.push_back(Xn.S);
        P.push_back(Xn.P);
        V.push_back(Xn.V);
        A.push_back(Xn.A);
    }

    void setKnot(int kidx, const GPState<double> &Xn)
    {
        R[kidx] = Xn.R;
        O[kidx] = Xn.O;
        S[kidx] = Xn.S;
        P[kidx] = Xn.P;
        V[kidx] = Xn.V;
        A[kidx] = Xn.A;
    }

    void updateKnot(int kidx, Matrix<double, STATE_DIM, 1> dX)
    {
        R[kidx] = R[kidx]*SO3d::exp(dX.block<3, 1>(0, 0));
        O[kidx] = O[kidx] + dX.block<3, 1>(3, 0);
        S[kidx] = S[kidx] + dX.block<3, 1>(6, 0);
        P[kidx] = P[kidx] + dX.block<3, 1>(9, 0);
        V[kidx] = V[kidx] + dX.block<3, 1>(12, 0);
        A[kidx] = A[kidx] + dX.block<3, 1>(15, 0);
    }

    void genRandomTrajectory(int n, double scale = 5.0)
    {
        R.clear(); O.clear(); S.clear(); P.clear(); V.clear(); A.clear();

        for(int kidx = 0; kidx < n; kidx++)
        {
            R.push_back(SO3d::exp(Vec3::Random()* M_PI));
            O.push_back(Vec3::Random() * scale);
            S.push_back(Vec3::Random() * scale);
            P.push_back(Vec3::Random() * scale);
            V.push_back(Vec3::Random() * scale);
            A.push_back(Vec3::Random() * scale);
        }
    }

    // Copy constructor
    GaussianProcess &operator=(const GaussianProcess &GPother)
    {
        this->t0 = GPother.getMinTime();
        this->dt = GPother.getDt();
        this->gpm = GPMixer(this->dt);

        this->R = GPother.R;
        this->O = GPother.O;
        this->S = GPother.S;
        this->P = GPother.P;
        this->V = GPother.V;
        this->A = GPother.A;

        return *this;
    }
};
// Define the shared pointer
typedef std::shared_ptr<GaussianProcess> GaussianProcessPtr;