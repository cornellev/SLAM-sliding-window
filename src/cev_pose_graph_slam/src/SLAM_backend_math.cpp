// ROS 2 front end for the sliding-window pose graph backend.
//
//   /odometry/filtered (EKF: ICP odometry + IMU) --> odometry edges (interpolated to scan time)
//   /scan                                         --> keyframes, scan-matching edges, loop closures
//   publishes TF map -> odom, and /slam/path (all keyframe estimates)
//
// All of the math lives in sliding_window_pgo.cpp so it can be tested without ROS
// (see test/sim_test.cpp).

#include "cev_pose_graph_slam/icp_scan_matcher.hpp"
#include "cev_pose_graph_slam/sliding_window_pgo.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"

using namespace cev_slam;

static double yawFromQuat(const geometry_msgs::msg::Quaternion& q) {
    return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

static geometry_msgs::msg::Quaternion quatFromYaw(double yaw) {
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);
    geometry_msgs::msg::Quaternion out;
    out.x = q.x();
    out.y = q.y();
    out.z = q.z();
    out.w = q.w();
    return out;
}

class PgoNode : public rclcpp::Node {
public:
    PgoNode(): Node("pgo_backend") {
        map_frame_ = declare_parameter("map_frame", "map");
        odom_frame_ = declare_parameter("odom_frame", "odom");
        base_frame_ = declare_parameter("base_frame", "base_link");
        odom_max_gap_ = declare_parameter("odom_max_gap", 0.2);
        min_range_ = declare_parameter("scan_min_range", 0.15);
        max_range_ = declare_parameter("scan_max_range", 10.0);
        voxel_ = declare_parameter("scan_voxel", 0.05);

        Params p;
        p.window_size = declare_parameter("window_size", p.window_size);
        p.kf_trans = declare_parameter("keyframe_translation", p.kf_trans);
        p.kf_rot = declare_parameter("keyframe_rotation", p.kf_rot);
        p.local_radius = declare_parameter("local_match_radius", p.local_radius);
        p.max_local_matches = declare_parameter("max_local_matches", p.max_local_matches);
        p.loop_radius = declare_parameter("loop_search_radius", p.loop_radius);
        p.loop_min_gap = declare_parameter("loop_min_keyframe_gap", p.loop_min_gap);
        if (!declare_parameter("use_marginalization", true)) p.mode = WindowMode::Truncate;

        IcpScanMatcher::Config mc;
        mc.method = declare_parameter("icp_method", mc.method);
        mc.overlap_rate = declare_parameter("icp_overlap_rate", mc.overlap_rate);
        mc.min_fitness = declare_parameter("match_min_fitness", mc.min_fitness);
        mc.max_rmse = declare_parameter("match_max_rmse", mc.max_rmse);

        ScanMatcher matcher;
        scan_matching_ = declare_parameter("use_scan_matching", true);
        if (scan_matching_)
            matcher = IcpScanMatcher::asFunction(std::make_shared<IcpScanMatcher>(mc));
        pgo_ = std::make_unique<SlidingWindowPGO>(p, matcher);

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        path_pub_ = create_publisher<nav_msgs::msg::Path>("slam/path", rclcpp::QoS(1).transient_local());

        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            declare_parameter("odom_topic", "/odometry/filtered"), 50,
            std::bind(&PgoNode::odomCallback, this, std::placeholders::_1));
        scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
            declare_parameter("scan_topic", "/scan"), rclcpp::SensorDataQoS(),
            std::bind(&PgoNode::scanCallback, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "Sliding-window PGO ready (window=%d, %s, scan matching %s)",
            p.window_size, p.mode == WindowMode::Marginalize ? "marginalization" : "truncation",
            matcher ? "on" : "off");
    }

private:
    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        // laser -> base_link, looked up once (static on the car).
        if (!laser_to_base_) {
            try {
                auto t = tf_buffer_->lookupTransform(base_frame_, msg->header.frame_id, tf2::TimePointZero);
                laser_to_base_ = Pose2{t.transform.translation.x, t.transform.translation.y,
                    yawFromQuat(t.transform.rotation)};
            } catch (const tf2::TransformException& ex) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                    "No TF %s -> %s yet (%s); dropping scan", base_frame_.c_str(),
                    msg->header.frame_id.c_str(), ex.what());
                return;
            }
        }
        PointList pts;
        pts.reserve(msg->ranges.size());
        for (size_t i = 0; i < msg->ranges.size(); ++i) {
            float r = msg->ranges[i];
            if (!std::isfinite(r) || r < msg->range_min || r > msg->range_max) continue;
            double a = msg->angle_min + i * msg->angle_increment;
            pts.push_back(transformPoint(*laser_to_base_, Point(r * std::cos(a), r * std::sin(a))));
        }
        pending_scans_.emplace_back(rclcpp::Time(msg->header.stamp).seconds(),
            filterScan(pts, min_range_, max_range_, voxel_));
        if (pending_scans_.size() > 10) pending_scans_.pop_front();
        processPendingScans();
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        const auto& pp = msg->pose.pose;
        Pose2 odom{pp.position.x, pp.position.y, yawFromQuat(pp.orientation)};
        double stamp = rclcpp::Time(msg->header.stamp).seconds();

        if (!scan_matching_) {
            addKeyframeCandidate(odom, stamp, {});
        } else {
            odom_buf_.emplace_back(stamp, odom);
            while (odom_buf_.size() > 2 && odom_buf_.front().first < stamp - 2.0) odom_buf_.pop_front();
            processPendingScans();
        }
        publishMapToOdom(msg->header.stamp);
    }

    // Keyframes are taken at scan times: odometry is interpolated to the scan stamp, so the scan
    // and the pose it is attached to describe the same instant. (Pairing a scan with whatever
    // odometry message arrives next biases every scan-matching edge by up to one scan period.)
    void processPendingScans() {
        while (!pending_scans_.empty() && !odom_buf_.empty()) {
            auto& [ts, scan] = pending_scans_.front();
            if (ts > odom_buf_.back().first) return;  // wait for odometry to catch up
            if (ts < odom_buf_.front().first) {        // too old to interpolate
                pending_scans_.pop_front();
                continue;
            }
            auto hi = std::lower_bound(odom_buf_.begin(), odom_buf_.end(), ts,
                [](const auto& o, double t) { return o.first < t; });
            auto lo = hi == odom_buf_.begin() ? hi : std::prev(hi);
            if (hi->first - lo->first <= odom_max_gap_) {
                double a = hi->first > lo->first ? (ts - lo->first) / (hi->first - lo->first) : 0.0;
                const Pose2 &p0 = lo->second, &p1 = hi->second;
                Pose2 odom{p0.x + a * (p1.x - p0.x), p0.y + a * (p1.y - p0.y),
                    wrapAngle(p0.th + a * wrapAngle(p1.th - p0.th))};
                addKeyframeCandidate(odom, ts, scan);
            }
            pending_scans_.pop_front();
        }
    }

    void addKeyframeCandidate(const Pose2& odom, double stamp, const PointList& scan) {
        auto before = pgo_->stats();
        if (!pgo_->addOdometry(odom, stamp, scan)) return;
        const auto& s = pgo_->stats();
        if (s.loop_edges > before.loop_edges)
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000, "Loop closure at keyframe %d (%d total)",
                pgo_->window().back().id, s.loop_edges);
        RCLCPP_DEBUG(get_logger(), "KF %d | window %zu | scan edges %d | rejected %d | chi2 %.2f (%d it)",
            pgo_->window().back().id, pgo_->window().size(), s.scan_edges, s.rejected_matches,
            s.last_chi2, s.last_iterations);
        publishPath(rclcpp::Time(static_cast<int64_t>(stamp * 1e9)));
    }

    void publishMapToOdom(const builtin_interfaces::msg::Time& stamp) {
        const Pose2& T = pgo_->mapToOdom();
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = stamp;  // stamp of the odometry it was computed from, not now()
        t.header.frame_id = map_frame_;
        t.child_frame_id = odom_frame_;
        t.transform.translation.x = T.x;
        t.transform.translation.y = T.y;
        t.transform.rotation = quatFromYaw(T.th);
        tf_broadcaster_->sendTransform(t);
    }

    void publishPath(const builtin_interfaces::msg::Time& stamp) {
        nav_msgs::msg::Path path;
        path.header.stamp = stamp;
        path.header.frame_id = map_frame_;
        auto add = [&](const Keyframe& kf) {
            geometry_msgs::msg::PoseStamped ps;
            ps.header.frame_id = map_frame_;
            ps.header.stamp = rclcpp::Time(static_cast<int64_t>(kf.stamp * 1e9));
            ps.pose.position.x = kf.est.x;
            ps.pose.position.y = kf.est.y;
            ps.pose.orientation = quatFromYaw(kf.est.th);
            path.poses.push_back(ps);
        };
        for (const auto& kf: pgo_->archive()) add(kf);
        for (const auto& kf: pgo_->window()) add(kf);
        path_pub_->publish(path);
    }

    std::string map_frame_, odom_frame_, base_frame_;
    double odom_max_gap_, min_range_, max_range_, voxel_;
    bool scan_matching_ = true;

    std::unique_ptr<SlidingWindowPGO> pgo_;
    std::optional<Pose2> laser_to_base_;
    std::deque<std::pair<double, Pose2>> odom_buf_;          // (stamp, odom pose), last ~2 s
    std::deque<std::pair<double, PointList>> pending_scans_;  // (stamp, base-frame points)

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PgoNode>());
    rclcpp::shutdown();
    return 0;
}
