#include "cev_pose_graph_slam/icp_scan_matcher.hpp"

#include "icp/driver.h"
#include "icp/icp.h"

#include <Eigen/Geometry>

#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace cev_slam {

IcpScanMatcher::IcpScanMatcher(Config cfg): cfg_(std::move(cfg)) {
    icp::ICP::Config icp_cfg;
    icp_cfg.set("overlap_rate", cfg_.overlap_rate);
    auto method = icp::ICP::from_method(cfg_.method, icp_cfg);
    if (!method) throw std::invalid_argument("unknown icp method: " + cfg_.method);
    driver_ = std::make_unique<icp::ICPDriver>(std::move(method.value()));
    driver_->set_max_iterations(cfg_.max_iterations);
    driver_->set_transform_tolerance(0.01 * M_PI / 180, 0.001);
}

IcpScanMatcher::~IcpScanMatcher() = default;

std::optional<MatchResult> IcpScanMatcher::match(const PointList& a, const PointList& b,
    const Pose2& guess) {
    if (a.size() < cfg_.min_points || b.size() < cfg_.min_points) return std::nullopt;

    // cevicp finds T with T(a) ~= b, i.e. T maps points from frame a into frame b, so
    // T = rel^-1 where rel is the pose of frame b expressed in frame a.
    Pose2 g_inv = inverse(guess);
    icp::RBTransform t0(icp::Vector(g_inv.x, g_inv.y),
        Eigen::Rotation2Dd(g_inv.th).toRotationMatrix());
    auto result = driver_->converge(a, b, t0);
    icp::RBTransform T_inv = result.transform.inverse();
    Pose2 rel{T_inv.translation.x(), T_inv.translation.y(),
        std::atan2(T_inv.rotation(1, 0), T_inv.rotation(0, 0))};
    if (!std::isfinite(rel.x) || !std::isfinite(rel.y) || !std::isfinite(rel.th)) return std::nullopt;

    // Verify: bring b into frame a and count points that land near a point of a.
    const double gate2 = cfg_.inlier_dist * cfg_.inlier_dist;
    size_t inliers = 0;
    double sq_sum = 0.0;
    for (const auto& pb: b) {
        Point q = transformPoint(rel, pb);
        double best = std::numeric_limits<double>::infinity();
        for (const auto& pa: a) best = std::min(best, (pa - q).squaredNorm());
        if (best < gate2) {
            ++inliers;
            sq_sum += best;
        }
    }
    if (inliers == 0) return std::nullopt;
    MatchResult m{rel, static_cast<double>(inliers) / b.size(), std::sqrt(sq_sum / inliers)};
    if (m.fitness < cfg_.min_fitness || m.rmse > cfg_.max_rmse) return std::nullopt;
    return m;
}

PointList filterScan(const PointList& pts, double min_range, double max_range, double voxel) {
    if (voxel <= 0) {
        PointList out;
        for (const auto& p: pts) {
            double r = p.norm();
            if (std::isfinite(r) && r >= min_range && r <= max_range) out.push_back(p);
        }
        return out;
    }
    std::map<std::pair<long, long>, std::pair<Point, int>> cells;
    for (const auto& p: pts) {
        double r = p.norm();
        if (!std::isfinite(r) || r < min_range || r > max_range) continue;
        auto key = std::make_pair(static_cast<long>(std::floor(p.x() / voxel)),
            static_cast<long>(std::floor(p.y() / voxel)));
        auto& cell = cells[key];
        cell.first = cell.second == 0 ? p : Point(cell.first + p);  // Eigen does not zero-init
        cell.second++;
    }
    PointList out;
    out.reserve(cells.size());
    for (const auto& [key, cell]: cells) out.push_back(cell.first / cell.second);
    return out;
}

}  // namespace cev_slam
