#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <cmath>
#include <iostream>
#include <vector>
#include <memory>
#include <deque>
#include <algorithm>

static constexpr int WINDOW_SIZE = 20;

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"

// --- Math Helpers ---

static inline double wrapAngle(double a) {
    a = std::fmod(a + M_PI, 2.0 * M_PI);
    if (a < 0) a += 2.0 * M_PI;
    return a - M_PI;
}

struct Pose2 {
    double x = 0, y = 0, th = 0;
};

struct Edge2 {
    int i = 0, j = 0;
    Eigen::Vector3d z;      
    Eigen::Matrix3d Omega;
};

static inline Eigen::Matrix2d R(double th) {
    double c = std::cos(th), s = std::sin(th);
    Eigen::Matrix2d M;
    M << c, -s, s, c;
    return M;
}

static inline Eigen::Vector3d computeError(const Pose2& xi, const Pose2& xj, const Eigen::Vector3d& z) {
    Eigen::Vector2d ti(xi.x, xi.y), tj(xj.x, xj.y);
    Eigen::Vector2d dt = tj - ti;
    Eigen::Vector2d trans_err = R(xi.th).transpose() * dt - z.head<2>();
    double rot_err = wrapAngle((xj.th - xi.th) - z[2]);
    Eigen::Vector3d e;
    e << trans_err, rot_err;
    return e;
}

static inline void computeJacobians(
    const Pose2& xi, const Pose2& xj,
    Eigen::Matrix<double,3,3>& Ji, Eigen::Matrix<double,3,3>& Jj)
{
    Ji.setZero(); Jj.setZero();
    Eigen::Matrix2d RiT = R(xi.th).transpose();
    Ji.block<2,2>(0,0) = -RiT;
    Jj.block<2,2>(0,0) =  RiT;
    double c = std::cos(xi.th), s = std::sin(xi.th);
    Eigen::Matrix2d dRiT;
    dRiT << -s, c, -c, -s;
    Eigen::Vector2d dt(xj.x - xi.x, xj.y - xi.y);
    Eigen::Vector2d d = dRiT * dt;
    Ji(0,2) = d[0]; Ji(1,2) = d[1];
    Ji(2,2) = -1.0; Jj(2,2) = 1.0;
}

static inline void applyIncrement(Pose2& x, const Eigen::Vector3d& dx) {
    x.x += dx[0];
    x.y += dx[1];
    x.th = wrapAngle(x.th + dx[2]);
}

bool gaussNewtonStep(std::deque<Pose2>& W, const std::vector<Edge2>& edges, double damping = 0.0) {
    if (W.size() < 2) return false;
    const int N = static_cast<int>(W.size());
    const int D = 3 * N;
    std::vector<Eigen::Triplet<double>> trips;
    Eigen::VectorXd b = Eigen::VectorXd::Zero(D);

    auto addBlock = [&](int r0, int c0, const Eigen::Matrix3d& M) {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                if (std::abs(M(r,c)) > 1e-12)
                    trips.emplace_back(r0 + r, c0 + c, M(r,c));
    };

    double chi2 = 0.0;
    for (const auto& e : edges) {
        const Pose2& xi = W[e.i];
        const Pose2& xj = W[e.j];
        Eigen::Vector3d err = computeError(xi, xj, e.z);
        chi2 += err.transpose() * e.Omega * err;
        Eigen::Matrix<double,3,3> Ji, Jj;
        computeJacobians(xi, xj, Ji, Jj);
        addBlock(3*e.i, 3*e.i, Ji.transpose() * e.Omega * Ji);
        addBlock(3*e.i, 3*e.j, Ji.transpose() * e.Omega * Jj);
        addBlock(3*e.j, 3*e.i, Jj.transpose() * e.Omega * Ji);
        addBlock(3*e.j, 3*e.j, Jj.transpose() * e.Omega * Jj);
        b.segment<3>(3*e.i) += Ji.transpose() * e.Omega * err;
        b.segment<3>(3*e.j) += Jj.transpose() * e.Omega * err;
    }

    Eigen::SparseMatrix<double> H(D, D);
    H.setFromTriplets(trips.begin(), trips.end());
    if (damping > 0.0) {
        for (int k = 0; k < D; ++k) H.coeffRef(k,k) += damping;
    }
    // Fix oldest pose in window as anchor (gauge fix)
    for (int k = 0; k < 3; ++k) { H.coeffRef(k, k) += 1e12; b[k] = 0.0; }

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(H);
    if (solver.info() != Eigen::Success) return false;
    Eigen::VectorXd dx = solver.solve(-b);
    for (int i = 0; i < N; ++i) applyIncrement(W[i], dx.segment<3>(3*i));
    return true;
}

// --- ROS 2 Node ---

class PgoNode : public rclcpp::Node {
public:
    PgoNode() : Node("pgo_backend") {
        // STEP 1: Change subscription from "/odom" to "/odometry/filtered"
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odometry/filtered", 10, std::bind(&PgoNode::odom_callback, this, std::placeholders::_1));
        
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        
        // Initialize the first pose at the origin
        W_.push_back({0.0, 0.0, 0.0});
        
        RCLCPP_INFO(this->get_logger(), "PGO Backend Initialized. Listening to Fused EKF Odometry.");
    }

private:
    void marginalize() {
        // Remove the sequential edge (0,1) — the only edge incident to local index 0
        W_edges_.erase(
            std::remove_if(W_edges_.begin(), W_edges_.end(),
                [](const Edge2& e){ return e.i == 0 || e.j == 0; }),
            W_edges_.end());
        // Drop oldest pose from front of window
        W_.pop_front();
        // Shift all window-local indices down by 1
        for (auto& e : W_edges_) { e.i--; e.j--; }
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        if (W_.empty()) return;

        double cur_x = msg->pose.pose.position.x;
        double cur_y = msg->pose.pose.position.y;
        double qz = msg->pose.pose.orientation.z;
        double qw = msg->pose.pose.orientation.w;
        double cur_th = std::atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz);

        Pose2 last = W_.back();

        double travel_dist   = std::hypot(cur_x - last.x, cur_y - last.y);
        double rotation_dist = std::abs(wrapAngle(cur_th - last.th));

        if (travel_dist > 0.3 || rotation_dist > 0.26) {
            Eigen::Vector3d z(cur_x - last.x, cur_y - last.y, wrapAngle(cur_th - last.th));
            W_.push_back({cur_x, cur_y, cur_th});

            Eigen::Matrix3d info = Eigen::Matrix3d::Identity();
            info(0,0) = 500.0;
            info(1,1) = 500.0;
            info(2,2) = 3000.0;

            // Window-local indices: safe because push_back guarantees W_.size() >= 2
            W_edges_.push_back({(int)W_.size()-2, (int)W_.size()-1, z, info});

            // Slide the window: marginalize oldest pose when over capacity
            if ((int)W_.size() > WINDOW_SIZE) {
                marginalize();
            }

            // Optimize the bounded window
            for (int it = 0; it < 5; ++it) {
                gaussNewtonStep(W_, W_edges_, 1e-6);
            }

            RCLCPP_INFO(this->get_logger(),
                "Keyframe added. Window: %zu poses, %zu edges", W_.size(), W_edges_.size());
        }

        // Always publish the map->odom transform to keep the TF tree linked
        publish_correction(cur_x, cur_y, cur_th);
    }

    void publish_correction(double fused_x, double fused_y, double fused_th) {
        // The correction is the difference between our Optimized map pose
        // and the EKF's current fused odometry pose.
        Pose2 opt = W_.back();
        
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = this->get_clock()->now();
        t.header.frame_id = "map";
        t.child_frame_id = "odom";

        // Math: T_map_to_odom = T_map_to_base * inv(T_odom_to_base)
        t.transform.translation.x = opt.x - fused_x;
        t.transform.translation.y = opt.y - fused_y;
        t.transform.translation.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0, 0, wrapAngle(opt.th - fused_th));
        t.transform.rotation.x = q.x();
        t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z();
        t.transform.rotation.w = q.w();

        tf_broadcaster_->sendTransform(t);
    }

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::deque<Pose2>  W_;
    std::vector<Edge2> W_edges_;
};


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PgoNode>());
    rclcpp::shutdown();
    return 0;
}