#ifndef __MS_REWARD_UTILS_H__
#define __MS_REWARD_UTILS_H__

#include <Eigen/Dense>

struct InteractiveRewardTerms
{
    double total;
    double joint;
    double position;
    double energy;
    double health;
};

inline InteractiveRewardTerms computeInteractiveRewardTerms(
    const Eigen::VectorXd &joint_diff,
    const Eigen::VectorXd &position_diff,
    const Eigen::VectorXd &effort,
    bool is_healthy,
    double joint_weight,
    double position_weight,
    double energy_weight,
    double health_bonus)
{
    InteractiveRewardTerms terms;
    terms.joint = -joint_weight * joint_diff.squaredNorm();
    terms.position = -position_weight * position_diff.squaredNorm();
    terms.energy = -energy_weight * effort.squaredNorm();
    terms.health = is_healthy ? health_bonus : 0.0;
    terms.total = terms.joint + terms.position + terms.energy + terms.health;
    return terms;
}

#endif
