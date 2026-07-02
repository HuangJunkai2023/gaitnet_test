#include "sim/RewardUtils.h"

#include <Eigen/Dense>
#include <cassert>
#include <cmath>

int main()
{
    const Eigen::VectorXd joint_diff = Eigen::Vector2d(1.0, 2.0);
    const Eigen::VectorXd position_diff = Eigen::Vector3d(0.5, 0.0, -0.5);
    const Eigen::VectorXd effort = Eigen::Vector2d(3.0, 4.0);

    const InteractiveRewardTerms terms = computeInteractiveRewardTerms(
        joint_diff, position_diff, effort, true, 0.5, 0.25, 0.01, 1.0);

    assert(std::abs(terms.joint + 2.5) < 1e-12);
    assert(std::abs(terms.position + 0.125) < 1e-12);
    assert(std::abs(terms.energy + 0.25) < 1e-12);
    assert(std::abs(terms.health - 1.0) < 1e-12);
    assert(std::abs(terms.total - (-2.5 - 0.125 - 0.25 + 1.0)) < 1e-12);

    const InteractiveRewardTerms unhealthy_terms = computeInteractiveRewardTerms(
        joint_diff, position_diff, effort, false, 0.5, 0.25, 0.01, 1.0);
    assert(std::abs(unhealthy_terms.health) < 1e-12);
    assert(unhealthy_terms.total < terms.total);

    const InteractiveRewardTerms terminal_terms = computeInteractiveRewardTerms(
        joint_diff, position_diff, effort, false, 0.5, 0.25, 0.01, 1.0, true, -200.0);
    assert(std::abs(terminal_terms.terminal + 200.0) < 1e-12);
    assert(std::abs(terminal_terms.total - (unhealthy_terms.total - 200.0)) < 1e-12);

    return 0;
}
