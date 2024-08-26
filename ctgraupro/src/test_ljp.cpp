#include "utility.h"
#include "GaussianProcess.hpp"


int main()
{
    yolos("begin test");

    double Dt = 0.1;
    GaussianProcessPtr traj(new GaussianProcess(Dt, Vector3d(10, 10, 10).asDiagonal(), Vector3d(10, 10, 10).asDiagonal()));
    traj->setStartTime(0);
    // traj->genRandomTrajectory(100);

    GPState<double> X0(0);
    traj->setKnot(0,X0);
    for (size_t i = 0; i < 10; i++)
    {
        traj->extendOneKnot();
    }

    GPState<double> x = traj->getStateAt(1.);
    std::cout<<x.R.unit_quaternion().toRotationMatrix()<<"\n";

    yolos("finish");
}