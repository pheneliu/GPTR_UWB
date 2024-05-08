/**
 * This file is part of spline_test.
 *
 * Copyright (C) 2020 Thien-Minh Nguyen <thienminh.npn at ieee dot org>,
 * School of EEE
 * Nanyang Technological Univertsity, Singapore
 *
 * For more information please see <https://britsknguyen.github.io>.
 * or <https://github.com/britsknguyen/spline_test>.
 * If you use this code, please cite the respective publications as
 * listed on the above websites.
 *
 * spline_test is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * spline_test is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with spline_test.  If not, see <http://www.gnu.org/licenses/>.
 */

//
// Created by Thien-Minh Nguyen on 15/12/20.
//

#ifndef spline_test_GyroAcceBiasFactor_H_
#define spline_test_GyroAcceBiasFactor_H_

#include <iostream>
#include <ros/assert.h>
#include <ceres/ceres.h>

#include "../utility.h"
#include <basalt/spline/ceres_spline_helper.h>

using namespace basalt;

template <int _N>
class GyroAcceBiasFactor : public basalt::CeresSplineHelper<_N>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GyroAcceBiasFactor(const ImuSample &imu_sample_, const ImuBias &imu_bias_,
                       double dt_, double u_, Vector3d &grav_,
                       double gyro_weight_, double acce_weight_,
                       double gyro_bias_weight_, double acce_bias_weight_,
                       int factor_index_)
        : imu_sample(imu_sample_),
          imu_bias(imu_bias_),
          dt(dt_),
          inv_dt(1.0/dt_),
          u(u_),
          grav(grav_),
          gyro_weight(gyro_weight_),
          acce_weight(acce_weight_),
          gyro_bias_weight(gyro_bias_weight_),
          acce_bias_weight(acce_bias_weight_),
          factor_index(factor_index_)
    {
        // printf(KYEL
        //        "IMU factor data %d.\n"
        //        "Gyro Meas: %f, %f, %f\n"
        //        "Acce Meas: %f, %f, %f\n"
        //        "Gravity:   %f, %f, %f\n" RESET,
        //         factor_index,
        //         gyro_meas(0), gyro_meas(1), gyro_meas(2),
        //         acce_meas(0), acce_meas(1), acce_meas(2),
        //         grav(0), grav(1), grav(2));

        // printf(KGRN
        //        "IMU factor bias %d.\n"
        //        "Gyro Bias: %f, %f, %f\n"
        //        "Acce Bias: %f, %f, %f\n" RESET,
        //         factor_index,
        //         gyro_bias_(0), gyro_bias_(1), gyro_bias_(2),
        //         acce_bias_(0), acce_bias_(1), acce_bias_(2)); 
    }

    template <class T>
    bool operator()(T const *const *sKnots, T *sResiduals) const
    {
        using Vec3T   = Eigen::Matrix<T, 3, 1>;
        using Vec12T  = Eigen::Matrix<T, 12, 1>;
        using SO3T    = Sophus::SO3<T>;
        using Tangent = typename Sophus::SO3<T>::Tangent;

        // Parameter blocks are organized by:
        // [N 4-dimensional quaternions ] + [N 3-dimensional positions] + [3-dimensional gyro bias] + [3-dimensional acce bias]
        size_t R_offset = 0;              // for quaternion
        size_t P_offset = R_offset + _N;  // for position
        size_t B_offset = P_offset + _N;  // for bias
        
        SO3T R_w_i;
        Tangent rot_vel;
        CeresSplineHelper<_N>::template evaluate_lie<T, Sophus::SO3>(sKnots + R_offset, u, inv_dt, &R_w_i, &rot_vel);

        Vec3T accel_w;
        CeresSplineHelper<_N>::template evaluate<T, 3, 2>(sKnots + P_offset, u, inv_dt, &accel_w);

        Eigen::Map<const Vec3T> gyro_bias(sKnots[B_offset]);
        Eigen::Map<const Vec3T> acce_bias(sKnots[B_offset + 1]);

        Vec3T gyro_residuals = rot_vel + gyro_bias - imu_sample.gyro.template cast<T>();
        Vec3T acce_residuals = R_w_i.inverse() * (accel_w + grav) + acce_bias - imu_sample.acce.template cast<T>();

        Eigen::Map<Vec12T> residuals(sResiduals);
        residuals.template block<3, 1>(0, 0) = T(gyro_weight) * gyro_residuals;
        residuals.template block<3, 1>(3, 0) = T(acce_weight) * acce_residuals;
        residuals.template block<3, 1>(6, 0) = T(gyro_bias_weight) * (gyro_bias - imu_bias.gyro_bias.template cast<T>());
        residuals.template block<3, 1>(9, 0) = T(acce_bias_weight) * (acce_bias - imu_bias.acce_bias.template cast<T>());

        // printf("IMU factor %d.\n"
        //        "Pred gyro: %.3f, %.3f, %.3f. Meas: %.3f, %.3f, %.3f. Bias: %.3f, %.3f, %.3f. Res: %.3f, %.3f, %.3f\n"
        //        "Pred acce: %.3f, %.3f, %.3f. Meas: %.3f, %.3f, %.3f.\n",
        //         factor_index,
        //         rot_vel(0), rot_vel(1), rot_vel(2),
        //         gyro_meas.x(), gyro_meas.y(), gyro_meas.z(),
        //         gyro_bias.x(), gyro_bias.y(), gyro_bias.z(),
        //         gyro_residuals(0), gyro_residuals(1), gyro_residuals(2),
        //         accel_w(0),   accel_w(1),   accel_w(2),
        //         acce_meas(0), acce_meas(1), acce_meas(2));

        // printf(KRED
        //        "IMU factor %d.\n"
        //        "Meas: %.3f, %.3f, %.3f. Bias: %.3f, %.3f, %.3f. Res: %.3f, %.3f, %.3f\n" RESET,
        //         factor_index,
        //         gyro_meas.x(), gyro_meas.y(), gyro_meas.z(),
        //         gyro_bias.x(), gyro_bias.y(), gyro_bias.z(),
        //         gyro_residuals(0), gyro_residuals(1), gyro_residuals(2));

        // exit(-1);

        return true;
    }

private:

    ImuSample imu_sample;
    ImuBias imu_bias;
    double dt;
    double inv_dt;
    double u;
    Vector3d grav;
    double gyro_weight;
    double acce_weight;
    double gyro_bias_weight;
    double acce_bias_weight;
    int factor_index;           // For debugging
};

#endif // spline_test_GyroAcceBiasFactor_H_