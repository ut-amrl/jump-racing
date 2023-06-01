//========================================================================
//  This software is free: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License Version 3,
//  as published by the Free Software Foundation.
//
//  This software is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
//
//  You should have received a copy of the GNU Lesser General Public License
//  Version 3 in the file COPYING that came with this distribution.
//  If not, see <http://www.gnu.org/licenses/>.
//========================================================================
/*!
\file    navigation.cc
\brief   Starter code for navigation.
\author  Joydeep Biswas, (C) 2019
*/
//========================================================================

/*
Milestone 1 objective: Drive forward while avoiding obstacles.

run:
  best_score = -infinity
  best_path = infinity
  for each path (curvature between -1 and 1): incrementing by .2 including 0,-1 and 1
    score path
    if score > best_score:
      best_score = score
      best_path = path
  Find speed and rotation for path // pretty much done
  execute  // pretty much done
  Update goal //needs to be done locally

// Note: Path is just a curvature value
scoring path:
  score = free_path_length(path) + w_1 * clearance_of_path(path)+ w2 * distance_to_goal(path) # Need to find w_1 and w_2 (w_1 positive w_2 negative)


Clearance computation:
  Conditions:
    straight: if x > 0 and x < free_path_length + dist to front of car + safe_margin and abs(y) < max_clearance
    curved: if abs((c - p).norm() - c_r) < c_max and 0 < angle of point from center < theta_max
  of single point going straight: l = abs(p[y])-(w+margin(?))
  of single point on an arc:

  // Note: These don't need a special condition for a point in the car path
  if ((centerOfTurning - PointCoordinates).norm() > r_c){
    Clearance = (centerOfTurning - PointCoordinates).norm() - radius_outer;
  else
    Clearance = radius_inner - (centerOfTurning - PointCoordinates).norm();
  }


pseudocode:
  given curvature:
  Get free path length (closest point along path)
  clearance = c_max
  for each point in cloud:
    if p is not going to affect clearance based on c_max and free path length, continue
    clearance = min(clearance, clearance_from_point(p))


Distance to goal path:

distanceTravelled = velocity * timeStep;
if (curvature == 0){
  distanceFromGoal = (goalLocation - [distanceTravelled, 0]).norm();
} else{
  R = 1/curvature;
  angle = distanceTravelled / R;
  dx = R * sin(angle);
  dy = R * sin(pi/2 - angle);
  distanceFromGoal = (goalLocation - [dx, dy]).norm();
}


Free path length (remaining distance):
  (effectively distance to first point along path)
*/


#include <cmath>
#include "gflags/gflags.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "amrl_msgs/AckermannCurvatureDriveMsg.h"
#include "amrl_msgs/Pose2Df.h"
#include "amrl_msgs/VisualizationMsg.h"
#include "glog/logging.h"
#include "ros/ros.h"
#include "ros/package.h"

#include "shared/math/math_util.h"
#include "shared/util/timer.h"
#include "shared/ros/ros_helpers.h"
#include "navigation.h"
#include "visualization/visualization.h"

using Eigen::Vector2f;
using amrl_msgs::AckermannCurvatureDriveMsg;
using amrl_msgs::VisualizationMsg;
using std::string;
using std::vector;

using namespace math_util;
using namespace ros_helpers;

#define VERY_BIG_NUMBER 9999999.0f
#define MAX_CLEARANCE 0.5
#define x 0
#define y 1



DEFINE_double(cp1_distance, 2.5, "Distance to travel for 1D TOC (cp1)");
DEFINE_double(cp1_curvature, 0.5, "Curvature for arc path (cp1)");

DEFINE_double(cp2_curvature, 0.5, "Curvature for arc path (cp2)");

//define constant
const float max_speed = 1.0f;
const float max_accel = 3.0f;
const float max_decel = -3.0f;

const float vehicle_len = 0.535;
const float vehicle_width = 0.281;
const Eigen::Vector2f lidar_offset = {0.2, 0};
const float base_link_to_front = 0.4;
const float base_link_to_back = 0.135;

const float base_link_to_front_axle = 0.324;

const float safe_margin = 0.1;


float free_path_length = 10; //random starting value

float Rc;
float Rmin;
float Rmax;
bool IS_TURNING;
float theta_min;
float free_path_angle;
float curvature;


namespace {
ros::Publisher drive_pub_;
ros::Publisher viz_pub_;
VisualizationMsg local_viz_msg_;
VisualizationMsg global_viz_msg_;
AckermannCurvatureDriveMsg drive_msg_;
// Epsilon value for handling limited numerical precision.
const float kEpsilon = 1e-5;
} //namespace

namespace navigation {

string GetMapFileFromName(const string& map) {
  string maps_dir_ = ros::package::getPath("amrl_maps");
  return maps_dir_ + "/" + map + "/" + map + ".vectormap.txt";
}

Navigation::Navigation(const string& map_name, ros::NodeHandle* n) :
    odom_initialized_(false),
    localization_initialized_(false),
    robot_loc_(0, 0),
    robot_angle_(0),
    robot_vel_(0, 0),
    robot_omega_(0),
    nav_complete_(true),
    nav_goal_loc_(0, 0),
    nav_goal_angle_(0) {
  map_.Load(GetMapFileFromName(map_name));
  drive_pub_ = n->advertise<AckermannCurvatureDriveMsg>(
      "ackermann_curvature_drive", 1);
  viz_pub_ = n->advertise<VisualizationMsg>("visualization", 1);
  local_viz_msg_ = visualization::NewVisualizationMessage(
      "base_link", "navigation_local");
  global_viz_msg_ = visualization::NewVisualizationMessage(
      "map", "navigation_global");
  InitRosHeader("base_link", &drive_msg_.header);

  nav_goal_loc_(x) = robot_loc_(x) + cos(robot_angle_) * FLAGS_cp1_distance;
  nav_goal_loc_(y) = robot_loc_(y) + sin(robot_angle_) * FLAGS_cp1_distance;
  IS_TURNING = false;

  goal = Vector2f(3, 0);

}

void Navigation::computeTurnParameters(float proposed_curvature) {

  curvature = proposed_curvature;
  if(proposed_curvature != 0) {
    Rc = curvature != 0 ? abs(1.0f / curvature) : VERY_BIG_NUMBER;
    Rmin = sqrt(pow(Rc - vehicle_width/2 - safe_margin, 2) + pow(base_link_to_front + safe_margin, 2));
    Rmax = sqrt(pow(Rc + vehicle_width/2 + safe_margin, 2) + pow(base_link_to_front + safe_margin, 2));
    theta_min = atan2(base_link_to_front, Rc+vehicle_width/2 + safe_margin * 2);
    IS_TURNING = true;
  } else {
    IS_TURNING = false;
  }
  free_path_length = calculateFreePathLength();
}

float Navigation::scorePath(float clearance, float distance_to_goal) {
//TODO: FIND GOOD WEIGHTS
  return free_path_length + 3 * clearance - 3.5 * distance_to_goal;
}

float Navigation::getClearanceOfPointCurved(Vector2f point, float dist_from_center) {

  if(dist_from_center > Rc) {
    return dist_from_center - Rmax;
  } else {
    return Rmin - dist_from_center;
  }
}

float Navigation::getClearanceOfPointLinear(Vector2f point) {
  return abs(point[y]) - vehicle_width / 2 - safe_margin;
}

float Navigation::getClearanceOfPath() {
  Vector2f center_of_turning = {0, Rc};
  float smallest_clearance = MAX_CLEARANCE;
  if(curvature != 0) {
    for(Vector2f point : point_cloud_) {
      float dist_from_center = (center_of_turning - point).norm();
      float cos_point_theta = (pow(center_of_turning.norm(), 2) + pow((center_of_turning - point).norm(), 2) - pow(point.norm(), 2)) / (2 * center_of_turning.norm() * (point - center_of_turning).norm());
      float point_theta = acos(cos_point_theta);
      if(abs(dist_from_center - Rc) < MAX_CLEARANCE && point_theta > 0 && point_theta < free_path_angle) {
        float clearance = getClearanceOfPointCurved(point, dist_from_center);
        smallest_clearance = std::min(smallest_clearance, clearance);
      }
    }
  } else {
    for(Vector2f point : point_cloud_) {
      if(point[x] > 0 && point[x] < free_path_length + base_link_to_front + safe_margin && point[y] < MAX_CLEARANCE) {
        smallest_clearance = std::min(smallest_clearance, getClearanceOfPointLinear(point));
      }
    }
  }
  return smallest_clearance;
}

float Navigation::getDistanceFromGoal() {
  float timeStep = 0.2; // Could be source of error
  float distanceTravelled = drive_msg_.velocity * timeStep;
  if (curvature == 0) {
    return 0;
    //distanceFromGoal = (goal - Vector2f(distanceTravelled, 0)).norm();
    //return distanceFromGoal;
  } else {
    float R = 1/curvature;
    float angle = distanceTravelled / R;
    float dx = R * sin(angle);
    float dy = R - R * cos(angle);
    // float distanceFromGoal = (goal - Vector2f(dx, dy)).norm();
    return (Vector2f(dx, dy) - Vector2f(distanceTravelled, 0)).norm();
  }
}


float Navigation::distanceAlongPathLinear(Vector2f point) {
  if(abs(point[y]) < vehicle_width / 2 + safe_margin) {
    return point[x];
  }
  return VERY_BIG_NUMBER;
}

float Navigation::distanceAlongPathCurved(Vector2f point) {
  if(curvature < 0) {
    point[y] *= -1;
  }
  float dist_from_center_of_turning = (point - Vector2f(0, Rc)).norm();

  float dist_from_car = point.norm();

  if(point[x] <= 0) {
    return VERY_BIG_NUMBER;
  }

  if(dist_from_center_of_turning >= Rmin && dist_from_center_of_turning <= Rmax) {
    float denominator = (2*dist_from_center_of_turning*Rc);
    float ratio = pow(dist_from_center_of_turning, 2)/denominator - pow(dist_from_car, 2)/denominator + pow(Rc, 2)/denominator;
    float theta_point = acos(ratio);
    float distance_along_path = theta_point * dist_from_center_of_turning;
    return distance_along_path;
  }
  return VERY_BIG_NUMBER;
}


float Navigation::calculateFreePathLength() {
  // free_path_length = max_speed * 0.05 + base_link_to_front + vehicle_len;
  free_path_length = 5;
  free_path_angle = free_path_length/Rc;
  if(IS_TURNING) {
    for(Vector2f point : point_cloud_) {
      float distance_along_path = distanceAlongPathCurved(point);
      if(distance_along_path < free_path_length) {
        free_path_length = distance_along_path;
        free_path_angle = distance_along_path / (point - Vector2f(0, Rc)).norm();
      }
      free_path_length = std::min(free_path_length, distanceAlongPathCurved(point));
    }
  } else {
    for(Vector2f point : point_cloud_) {
      free_path_length = std::min(free_path_length, distanceAlongPathLinear(point));
    }
  }

  return free_path_length;
}

float Navigation::CalculateCurrentAcceleration(float safe_deceleration_distance, float robot_speed) {

  //Find safe stopping distance
  //If greater than or equal to dist to target/obstacle: stop.
  //Otherwise, if current speed greater than or equal to  ~cruising speed (max car speed)~ don't do anything.
  //Otherwise, if distance remaining, less than max  accelerate.
  if(free_path_length < safe_deceleration_distance || free_path_length <= safe_margin + base_link_to_front) {
    // return float of max deceleration
    return max_decel;
  }

  if(robot_speed >= max_speed) {
    // return float of 0
    return 0.0;
  }

  // // return acceleration of max accel
  // if(robot_speed + max_accel / 20 >= max_speed) {
  //   return (max_speed - robot_speed) * 20;
  // }
  return max_accel;

}


void Navigation::SetNavGoal(const Vector2f& loc, float angle) {
}

void Navigation::UpdateLocation(const Eigen::Vector2f& loc, float angle) {
  localization_initialized_ = true;
  robot_loc_ = loc;
  robot_angle_ = angle;
}

void Navigation::UpdateOdometry(const Vector2f& loc,
                                float angle,
                                const Vector2f& vel,
                                float ang_vel) {
  robot_omega_ = ang_vel;
  robot_vel_ = vel;
  if (!odom_initialized_) {
    odom_start_angle_ = angle;
    odom_start_loc_ = loc;
    odom_initialized_ = true;
    odom_loc_ = loc;
    starting_odom_ = odom_loc_;
    odom_angle_ = angle;
    return;
  }
  odom_loc_ = loc;
  odom_angle_ = angle;
}

void Navigation::ObservePointCloud(const vector<Vector2f>& cloud,
                                   double time) {
  point_cloud_ = cloud;
}

void Navigation::Visualizations(float max_safe_deceleration_distance, float clearance) {
  // check if there is any points in safe distance and change viz color to red
  uint32_t color = 0x00FF00;
  if(free_path_length < safe_deceleration_distance_) {
    color = 0xFF0000;
  }

    // draw the visualization box around the car showing the safe stopping distance


  // left safe boundary of car
  Vector2f left_boundary_start_car_location = Vector2f(0, vehicle_width / 2 + safe_margin);
  Vector2f left_boundary_end_car_location = left_boundary_start_car_location + Vector2f(base_link_to_front  + safe_margin + safe_deceleration_distance_, 0);

  // right safe boundary of car
  Vector2f right_boundary_start_car_location = Vector2f(0, -vehicle_width / 2 - safe_margin);
  Vector2f right_boundary_end_car_location = right_boundary_start_car_location + Vector2f(base_link_to_front + safe_margin + safe_deceleration_distance_, 0);

  visualization::DrawLine(left_boundary_start_car_location, left_boundary_end_car_location, color, local_viz_msg_);
  visualization::DrawLine(right_boundary_start_car_location, right_boundary_end_car_location, color, local_viz_msg_);

  // front safe boundary of car
  visualization::DrawLine(left_boundary_end_car_location, right_boundary_end_car_location, color, local_viz_msg_);

  // back boundary of car
  // visualization::DrawLine(left_boundary_start_car_location - (Vector2f()), right_boundary_start_car_location, 0xFF0000, local_viz_msg_);


  //draw the axles of the car
  Vector2f back_axle_midpoint = Vector2f(0, 0);
  Vector2f front_axle_midpoint = Vector2f(back_axle_midpoint[x] + base_link_to_front_axle, back_axle_midpoint[y]);

  visualization::DrawCross(back_axle_midpoint, 0.01, 0x00FFFF, local_viz_msg_);
  visualization::DrawCross(front_axle_midpoint, 0.01, 0x00FFFF, local_viz_msg_);

  Vector2f back_axle_left = back_axle_midpoint + Vector2f(0, vehicle_width / 2);
  Vector2f back_axle_right = back_axle_midpoint + Vector2f(0, -vehicle_width / 2);
  visualization::DrawLine(back_axle_left, back_axle_right, 0x00FFFF, local_viz_msg_);

  Vector2f front_axle_left = front_axle_midpoint + Vector2f(0, vehicle_width / 2);
  Vector2f front_axle_right = front_axle_midpoint + Vector2f(0, -vehicle_width / 2);
  visualization::DrawLine(front_axle_left, front_axle_right, 0x00FFFF, local_viz_msg_);

  // // draw the car
  Vector2f back_left_car_location = Vector2f(-base_link_to_back, vehicle_width / 2);
  Vector2f front_left_car_location = back_left_car_location + Vector2f(vehicle_len, 0);

  Vector2f back_right_car_location = Vector2f(-base_link_to_back, -vehicle_width / 2);
  Vector2f front_right_car_location = back_right_car_location + Vector2f(vehicle_len, 0);

  visualization::DrawLine(back_left_car_location, front_left_car_location, 0x000000, local_viz_msg_);
  visualization::DrawLine(back_right_car_location, front_right_car_location, 0x000000, local_viz_msg_);
  visualization::DrawLine(back_left_car_location, back_right_car_location, 0x000000, local_viz_msg_);
  visualization::DrawLine(front_left_car_location, front_right_car_location, 0x000000, local_viz_msg_);

  // draw the angle of turning
  // Vector2f turning_point = Vector2f(base_link_to_front_axle, 0);
  //visualization::DrawArc(turning_point, FLAGS_cp1_curvature, 0, 0.5, 0xFF00FF, local_viz_msg_);

  // draw the path
  // visualization::DrawPathOption(FLAGS_cp1_curvature, free_path_length, -safe_margin, 0x0000FF, true, local_viz_msg_);

  // if(curvature < 0) {

  // } else if(curvature > 0) {
  //   Vector2f center_of_turning = {0, Rc};
  //   float arc_length = free_path_length;
  //   float base_link_radius = Rc;
  //   float start_angle = 0;
  //   float end_angle = arc_length / base_link_radius;

  //   // arc
  //   visualization::DrawArc(center_of_turning, base_link_radius, start_angle, end_angle, 0x31AFD4, local_viz_msg_);
  // } else {
  //   visualization::DrawLine(Vector2f(0, 0), Vector2f(free_path_length, 0), 0, local_viz_msg_);
  // }

  visualization::DrawPathOption(curvature, free_path_length, 0,  0x31AFD4, false, local_viz_msg_);
  visualization::DrawPathOption(curvature, free_path_length, clearance,  0x902D41, true, local_viz_msg_);



}

void Navigation::Run() {
  // This function gets called 20 times a second to form the control loop.

  // Clear previous visualizations.
  visualization::ClearVisualizationMsg(local_viz_msg_);
  visualization::ClearVisualizationMsg(global_viz_msg_);

  // If odometry has not been initialized, we can't do anything.
  if (!odom_initialized_) return;

  // The control iteration goes here.
  // Feel free to make helper functions to structure the control appropriately.

  // The latest observed point cloud is accessible via "point_cloud_"

  // Eventually, you will have to set the control values to issue drive commands:
  // drive_msg_.curvature = ...;robot_vel_
  // drive_msg_.velocity = ...;

  float robot_speed = robot_vel_.norm();
  float safe_deceleration_distance = -1*pow(robot_speed, 2) / (2 * max_decel);
  float dist_in_one_frame = robot_speed / 20;
  safe_deceleration_distance +=  dist_in_one_frame;
  safe_deceleration_distance_ = safe_deceleration_distance;

  float best_path_score = -VERY_BIG_NUMBER;
  float best_path = -1.0;
  float best_clearance = VERY_BIG_NUMBER;
  // float max_free_path_length = 0;
  // float max_free_path = 0;
  for(float proposed_path = -1.0; proposed_path < 1.0; proposed_path += 0.1) {
    computeTurnParameters(proposed_path);
    // if (free_path_length > max_free_path_length){
    //   max_free_path_length = free_path_length;
    //   max_free_path = proposed_path;
    // }
    float clearance = getClearanceOfPath();
    float distance_from_goal = getDistanceFromGoal();

    float score = scorePath(clearance, distance_from_goal);
    // printf("%f %f %f %f\n", free_path_length, clearance, distance_from_goal, score);
    if(score > best_path_score) {
      best_path_score = score;
      best_path = proposed_path;
      best_clearance = clearance;
    }
  }

  // printf("\n\n\n\n\n\n\n");
  // printf("(%f %f)\n", best_path_score, best_path);
  // if (max_free_path_length <= minStoppingDistance + safetyValue){
  //  free_path_length = max_free_path_length;
  //  best_path = max_free_path;
  // }


  computeTurnParameters(best_path); // curvature now = best_path

  drive_msg_.velocity = robot_vel_.norm() + CalculateCurrentAcceleration(safe_deceleration_distance, robot_speed) / 20;
  if(drive_msg_.velocity > max_speed) {
    drive_msg_.velocity = max_speed;
  } if(drive_msg_.velocity < 0) {
    drive_msg_.velocity = 0;
  }
  drive_msg_.curvature = curvature;

  // Draw new visualizations.
  Visualizations(safe_deceleration_distance, best_clearance);

  // Applying the CurrentAcceleration function to output actual velocities to the car


  // Add timestamps to all messages.
  local_viz_msg_.header.stamp = ros::Time::now();
  global_viz_msg_.header.stamp = ros::Time::now();
  drive_msg_.header.stamp = ros::Time::now();
  // Publish messages.
  viz_pub_.publish(local_viz_msg_);
  viz_pub_.publish(global_viz_msg_);
  drive_pub_.publish(drive_msg_);

}

}  // namespace navigation