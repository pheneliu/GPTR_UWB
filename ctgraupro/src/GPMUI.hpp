// All about gaussian process
#include "GPMLC.h"

class GPMUI : public GPMLC
{
private:

public:

//     // Destructor
//    ~GPMUI() {};
   
    // Constructor
    // GPMUI(ros::NodeHandlePtr &nh_) : GPMLC(nh_)
    // {

    // }

    void AddTDOAFactors(
        ceres::Problem &problem, GaussianProcessPtr &traj,
        map<double*, ParamInfo> &paramInfo, FactorMeta &factorMeta,
        // const deque<vector<LidarCoef>> &cloudCoef,
        double tmin, double tmax)
    {

    }

    void Evaluate(int iter, vector<GaussianProcessPtr> &trajs,
                  double tmin, double tmax, double tmid,
                //   const vector<deque<vector<LidarCoef>>> &cloudCoef,
                  bool do_marginalization,
                  myTf<double> &T_B_Li_gndtr)
    {

    }

};

typedef std::shared_ptr<GPMUI> GPMUIPtr;
