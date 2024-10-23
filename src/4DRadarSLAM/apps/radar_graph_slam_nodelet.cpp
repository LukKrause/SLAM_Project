// SPDX-License-Identifier: BSD-2-Clause

#include <ctime>
#include <mutex>
#include <atomic>
#include <memory>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/octree/octree_search.h>
#include <pcl/filters/passthrough.h>

#include <ros/ros.h>
#include <geodesy/utm.h>
#include <geodesy/wgs84.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <std_msgs/Time.h>
#include <std_msgs/String.h>
#include <nav_msgs/Odometry.h>
#include <nmea_msgs/Sentence.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <geographic_msgs/GeoPointStamped.h>
#include <visualization_msgs/MarkerArray.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <radar_graph_slam/SaveMap.h>
#include <radar_graph_slam/DumpGraph.h>
#include <radar_graph_slam/ros_utils.hpp>
#include <radar_graph_slam/ros_time_hash.hpp>
#include <radar_graph_slam/FloorCoeffs.h>
#include <radar_graph_slam/graph_slam.hpp>
#include <radar_graph_slam/keyframe.hpp>
#include <radar_graph_slam/keyframe_updater.hpp>
#include <radar_graph_slam/loop_detector.hpp>
#include <radar_graph_slam/information_matrix_calculator.hpp>
#include <radar_graph_slam/map_cloud_generator.hpp>
#include <radar_graph_slam/nmea_sentence_parser.hpp>
#include "radar_graph_slam/polynomial_interpolation.hpp"
#include <radar_graph_slam/registrations.hpp>

#include "scan_context/Scancontext.h"

#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/edge_se3_plane.hpp>
#include <g2o/edge_se3_priorxy.hpp>
#include <g2o/edge_se3_priorxyz.hpp>
#include <g2o/edge_se3_priorz.hpp>
#include <g2o/edge_se3_priorvec.hpp>
#include <g2o/edge_se3_priorquat.hpp>

#include <barometer_bmp388/Barometer.h>

#include "utility_radar.h"

using namespace std;

namespace radar_graph_slam {

class RadarGraphSlamNodelet : public nodelet::Nodelet, public ParamServer {
public:
  typedef pcl::PointXYZI PointT;
  typedef PointXYZIRPYT  PointTypePose;
  typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::PointCloud2> ApproxSyncPolicy;

  RadarGraphSlamNodelet() {}
  virtual ~RadarGraphSlamNodelet() {}

  virtual void onInit() {
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    // init parameters
    map_cloud_resolution = private_nh.param<double>("map_cloud_resolution", 0.05);
    trans_odom2map.setIdentity();
    trans_aftmapped.setIdentity();
    trans_aftmapped_incremental.setIdentity();
    initial_pose.setIdentity();

    max_keyframes_per_update = private_nh.param<int>("max_keyframes_per_update", 10);
    subsurface_removal_filter = private_nh.param<bool>("subsurface_removal_filter", true);

    anchor_node = nullptr;
    anchor_edge = nullptr;
    floor_plane_node = nullptr;
    graph_slam.reset(new GraphSLAM(private_nh.param<std::string>("g2o_solver_type", "lm_var")));
    keyframe_updater.reset(new KeyframeUpdater(private_nh));
    loop_detector.reset(new LoopDetector(private_nh));
    map_cloud_generator.reset(new MapCloudGenerator());
    inf_calclator.reset(new InformationMatrixCalculator(private_nh));
    nmea_parser.reset(new NmeaSentenceParser());

    gps_edge_intervals = private_nh.param<int>("gps_edge_intervals", 10);
    gps_time_offset = private_nh.param<double>("gps_time_offset", 0.0);
    gps_edge_stddev_xy = private_nh.param<double>("gps_edge_stddev_xy", 10000.0);
    gps_edge_stddev_z = private_nh.param<double>("gps_edge_stddev_z", 10.0);
    max_gps_edge_stddev_xy = private_nh.param<double>("max_gps_edge_stddev_xy", 1.0);
    max_gps_edge_stddev_z = private_nh.param<double>("max_gps_edge_stddev_z", 2.0);

    // Preintegration Parameters
    enable_preintegration = private_nh.param<bool>("enable_preintegration", false);
    use_egovel_preinteg_trans = private_nh.param<bool>("use_egovel_preinteg_trans", false);
    preinteg_trans_stddev = private_nh.param<double>("preinteg_trans_stddev", 1.0);
    preinteg_orient_stddev = private_nh.param<double>("preinteg_orient_stddev", 2.0);

    enable_barometer = private_nh.param<bool>("enable_barometer", false);
    barometer_edge_type = private_nh.param<int>("barometer_edge_type", 2);
    barometer_edge_stddev = private_nh.param<double>("barometer_edge_stddev", 0.5);

    points_topic = private_nh.param<std::string>("points_topic", "/radar_enhanced_pcl");

    show_sphere = private_nh.param<bool>("show_sphere", false);

    dataset_name = private_nh.param<std::string>("dataset_name", "");

    registration = select_registration_method(private_nh);

    // subscribers
    odom_sub.reset(new message_filters::Subscriber<nav_msgs::Odometry>(mt_nh, odomTopic, 256));
    cloud_sub.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(mt_nh, "/filtered_points", 32));
    sync.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(32), *odom_sub, *cloud_sub));
    sync->registerCallback(boost::bind(&RadarGraphSlamNodelet::cloud_callback, this, _1, _2));
    /*
    if(private_nh.param<bool>("enable_gps", true)) {
      gps_sub = mt_nh.subscribe("/gps/geopoint", 1024, &RadarGraphSlamNodelet::gps_callback, this);
      nmea_sub = mt_nh.subscribe("/gpsimu_driver/nmea_sentence", 1024, &RadarGraphSlamNodelet::nmea_callback, this);
      navsat_sub = mt_nh.subscribe(gpsTopic, 1024, &RadarGraphSlamNodelet::navsat_callback, this);
    }
    if(private_nh.param<bool>("enable_barometer", true)) {
      barometer_sub = mt_nh.subscribe("/barometer/filtered", 16, &RadarGraphSlamNodelet::barometer_callback, this);
    }
    if (enable_preintegration)
      imu_odom_sub = nh.subscribe("/imu_pre_integ/imu_odom_incre", 1024, &RadarGraphSlamNodelet::imu_odom_callback, this);
    imu_sub = nh.subscribe("/imu", 1024, &RadarGraphSlamNodelet::imu_callback, this);
    */
    command_sub = nh.subscribe("/command", 10, &RadarGraphSlamNodelet::command_callback, this);

    //***** publishers ******
    markers_pub = mt_nh.advertise<visualization_msgs::MarkerArray>("/radar_graph_slam/markers", 16);
    // Transform RadarOdom_to_base
    odom2base_pub = mt_nh.advertise<geometry_msgs::TransformStamped>("/radar_graph_slam/odom2base", 16);
    aftmapped_odom_pub = mt_nh.advertise<nav_msgs::Odometry>("/radar_graph_slam/aftmapped_odom", 16);
    aftmapped_odom_incremenral_pub = mt_nh.advertise<nav_msgs::Odometry>("/radar_graph_slam/aftmapped_odom_incremental", 16);
    map_points_pub = mt_nh.advertise<sensor_msgs::PointCloud2>("/radar_graph_slam/map_points", 1, true);
    read_until_pub = mt_nh.advertise<std_msgs::Header>("/radar_graph_slam/read_until", 32);
    odom_frame2frame_pub = mt_nh.advertise<nav_msgs::Odometry>("/radar_graph_slam/odom_frame2frame", 16);

    dump_service_server = mt_nh.advertiseService("/radar_graph_slam/dump", &RadarGraphSlamNodelet::dump_service, this);
    save_map_service_server = mt_nh.advertiseService("/radar_graph_slam/save_map", &RadarGraphSlamNodelet::save_map_service, this);

    graph_updated = false;
    double graph_update_interval = private_nh.param<double>("graph_update_interval", 3.0);
    double map_cloud_update_interval = private_nh.param<double>("map_cloud_update_interval", 10.0);
    optimization_timer = mt_nh.createWallTimer(ros::WallDuration(graph_update_interval), &RadarGraphSlamNodelet::optimization_timer_callback, this);
    map_publish_timer = mt_nh.createWallTimer(ros::WallDuration(map_cloud_update_interval), &RadarGraphSlamNodelet::map_points_publish_timer_callback, this);
  
    if (dataset_name == "loop3")
    utm_to_world << 
     -0.057621,       0.996222,      -0.064972, -128453.624105,
     -0.998281,      -0.058194,      -0.006954,  361869.958099,
     -0.010708,       0.064459,       0.997863,   -5882.237973,
      0.000000,       0.000000,       0.000000,       1.000000;
    else if (dataset_name == "loop2")
    utm_to_world <<
     -0.085585,       0.995774,      -0.033303, -117561.214476,
     -0.996323,      -0.085401,       0.006904,  364927.287181,
      0.004031,       0.033772,       0.999421,   -6478.377842,
      0.000000,       0.000000,       0.000000,       1.000000;
  }


private:
  /**
   * @brief received point clouds are pushed to #keyframe_queue
   * @param odom_msg
   * @param cloud_msg
   */
  void cloud_callback(const nav_msgs::OdometryConstPtr& odom_msg, const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
    const ros::Time& stamp = cloud_msg->header.stamp;
    Eigen::Isometry3d odom_now = odom2isometry(odom_msg);
    Eigen::Matrix4d matrix_map2base;
    // Publish TF between /map and /base_link
    if(keyframes.size() > 0)
    {
      const KeyFrame::Ptr& keyframe_last = keyframes.back();
      Eigen::Isometry3d lastkeyframe_odom_incre =  keyframe_last->odom_scan2scan.inverse() * odom_now;
      Eigen::Isometry3d keyframe_map2base_matrix = keyframe_last->node->estimate();
      // map2base = odom^(-1) * base
      matrix_map2base = (keyframe_map2base_matrix * lastkeyframe_odom_incre).matrix();
    }
    geometry_msgs::TransformStamped map2base_trans = matrix2transform(cloud_msg->header.stamp, matrix_map2base, mapFrame, baselinkFrame);
    if (pow(map2base_trans.transform.rotation.w,2)+pow(map2base_trans.transform.rotation.x,2)+
      pow(map2base_trans.transform.rotation.y,2)+pow(map2base_trans.transform.rotation.z,2) < pow(0.9,2)) 
      {map2base_trans.transform.rotation.w=1; map2base_trans.transform.rotation.x=0; map2base_trans.transform.rotation.y=0; map2base_trans.transform.rotation.z=0;}
    map2base_broadcaster.sendTransform(map2base_trans);
   
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if(baselinkFrame.empty()) {
      baselinkFrame = cloud_msg->header.frame_id;
    }
    
    // Push ego velocity to queue
    geometry_msgs::TwistStamped::Ptr twist_(new geometry_msgs::TwistStamped);
    twist_->header.stamp = cloud_msg->header.stamp;
    twist_->twist.linear = odom_msg->twist.twist.linear;
    {
      std::lock_guard<std::mutex> lock(keyframe_queue_mutex);
      twist_queue.push_back(twist_);
    }

    //********** Decided whether to accept the frame as a key frame or not **********
    if(!keyframe_updater->decide(odom_now, stamp)) {
      std::lock_guard<std::mutex> lock(keyframe_queue_mutex);
      if(keyframe_queue.empty()) {
        std_msgs::Header read_until;
        read_until.stamp = stamp + ros::Duration(10, 0);
        read_until.frame_id = points_topic;
        read_until_pub.publish(read_until);
        read_until.frame_id = "/filtered_points";
        read_until_pub.publish(read_until);
      }
      return;
    }
    // Get time of this key frame for Intergeration, to integerate between two key frames
    thisKeyframeTime = cloud_msg->header.stamp.toSec();
    
    double accum_d = keyframe_updater->get_accum_distance();
    // Construct keyframe
    KeyFrame::Ptr keyframe(new KeyFrame(keyframe_index, stamp, odom_now, accum_d, cloud));
    keyframe_index ++;

    /*if (enable_preintegration){
      // Intergerate translation of ego velocity, add rotation
      geometry_msgs::Transform transf_integ = preIntegrationTransform();
      static uint32_t sequ = 0;
      nav_msgs::Odometry odom_frame2frame;
      odom_frame2frame.pose.pose.orientation = transf_integ.rotation;
      odom_frame2frame.pose.pose.position.x = transf_integ.translation.x;
      odom_frame2frame.pose.pose.position.y = transf_integ.translation.y;
      odom_frame2frame.pose.pose.position.z = transf_integ.translation.z;
      odom_frame2frame.header.frame_id = "map";
      odom_frame2frame.header.stamp = cloud_msg->header.stamp;
      odom_frame2frame.header.seq = sequ; sequ ++;
      odom_frame2frame_pub.publish(odom_frame2frame);
      keyframe->trans_integrated = transf_integ;
    }*/
    
    
    std::lock_guard<std::mutex> lock(keyframe_queue_mutex);
    keyframe_queue.push_back(keyframe);

    // Scan Context loop detector - giseop
    // - SINGLE_SCAN_FULL: using downsampled original point cloud (/full_cloud_projected + downsampling)
    // - SINGLE_SCAN_FEAT: using surface feature as an input point cloud for scan context (2020.04.01: checked it works.)
    // - MULTI_SCAN_FEAT: using NearKeyframes (because a MulRan scan does not have beyond region, so to solve this issue ... )
    const SCInputType sc_input_type = SCInputType::SINGLE_SCAN_FULL; // change this 

    if( sc_input_type == SCInputType::SINGLE_SCAN_FULL ) {
        loop_detector->scManager->makeAndSaveScancontextAndKeys(*cloud);
    }
    // else if (sc_input_type == SCInputType::SINGLE_SCAN_FEAT) { 
    //     scManager.makeAndSaveScancontextAndKeys(*thisSurfKeyFrame); 
    // }
    // else if (sc_input_type == SCInputType::MULTI_SCAN_FEAT) { 
    //     pcl::PointCloud<PointT>::Ptr multiKeyFrameFeatureCloud(new pcl::PointCloud<PointT>());
    //     loopFindNearKeyframes(multiKeyFrameFeatureCloud, cloudKeyPoses6D->size() - 1, historyKeyframeSearchNum);
    //     scManager.makeAndSaveScancontextAndKeys(*multiKeyFrameFeatureCloud); 
    // }
    

    lastKeyframeTime = thisKeyframeTime;
  }


  /**
   * @brief this method adds all the keyframes_ in #keyframe_queue to the pose graph (odometry edges)
   * @return if true, at least one keyframe_ was added to the pose graph
   */
  bool flush_keyframe_queue() {
    std::lock_guard<std::mutex> lock(keyframe_queue_mutex);

    if(keyframe_queue.empty()) {
      return false;
    }

    trans_odom2map_mutex.lock();
    Eigen::Isometry3d odom2map(trans_odom2map.cast<double>());
    trans_odom2map_mutex.unlock();

    // Filtering keyframes: Points under the baselink frame are removed (Points under groundlevel)
    /*static tf::TransformListener listener;
    bool output_once = false;

    int num_processed = 0;
    cout << "keyframe_queue size: " << keyframe_queue.size() << endl;
    // ********** Select number of keyframes to be optimized **********
    for(int i = 0; i < std::min<int>(keyframe_queue.size(), max_keyframes_per_update); i++) {
      num_processed = i;

      const auto& keyframe = keyframe_queue[i];
      cout << "num_processed: " << num_processed << endl;
  
      // Skip already filtered keyframes
      if (!keyframe->filtered) {
        cout << "testtest: " << endl;
        tf::StampedTransform transform;
        try {
          listener.lookupTransform(mapFrame, baselinkFrame, ros::Time(0), transform);
        } catch (tf::TransformException &ex) {
          ROS_WARN("%s", ex.what());
          continue;
        }

        // Get the z-coordinate of the baselink frame
        double baselink_z = transform.getOrigin().z();
        if (!output_once) {
          cout << "Removing Points under z: " << baselink_z - 0.1 << endl;
          output_once = true;
        }
        size_t num_points_before = keyframe->cloud->size();

        // Filter the point cloud of the keyframe based on the height of the baselink frame
        pcl::PassThrough<PointT> pass;
        pass.setInputCloud(keyframe->cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(baselink_z - 0.1, std::numeric_limits<float>::max()); // Set the lower limit to baselink_z
        pcl::PointCloud<PointT>::Ptr filtered_cloud(new pcl::PointCloud<PointT>());
        pass.filter(*filtered_cloud);

        // Update the keyframe's cloud with the filtered cloud
        keyframe->cloud = filtered_cloud;
        size_t num_points_after = keyframe->cloud->size();
        if (num_points_before != num_points_after)
          cout << "Number of Keyframe points filtered: " << num_points_before - num_points_after << endl;

        // Mark the keyframe as filtered
        keyframe->filtered = true;
      }*/

    int num_processed = 0;
    // ********** Select number of keyframess to be optimized **********
    for(int i = 0; i < std::min<int>(keyframe_queue.size(), max_keyframes_per_update); i++) {
      num_processed = i;

      const auto& keyframe = keyframe_queue[i];
      // new_keyframess will be tested later for loop closure
      new_keyframes.push_back(keyframe);

      //cout << "new_keyframes size: " << new_keyframes.size() << "num processed: " << num_processed << "keyframe empty: " << keyframes.empty() << endl;

      // add pose node
      Eigen::Isometry3d odom = odom2map * keyframe->odom_scan2scan;
      // ********** Vertex of keyframess is contructed here ***********
      keyframe->node = graph_slam->add_se3_node(odom);
      keyframe_hash[keyframe->stamp] = keyframe;

      // fix the first node
      if(keyframes.empty() && new_keyframes.size() == 1) {
        //cout << "aaaaa" << endl;
        if(private_nh.param<bool>("fix_first_node", false)) {
          //cout << "bbbbb" << endl;
          Eigen::MatrixXd inf = Eigen::MatrixXd::Identity(6, 6);
          std::stringstream sst(private_nh.param<std::string>("fix_first_node_stddev", "1 1 1 1 1 1"));
          for(int i = 0; i < 6; i++) {
            double stddev = 1.0;
            sst >> stddev;
            inf(i, i) = 1.0 / stddev;
          }
          anchor_node = graph_slam->add_se3_node(Eigen::Isometry3d::Identity());
          anchor_node->setFixed(true);
          anchor_edge = graph_slam->add_se3_edge(anchor_node, keyframe->node, Eigen::Isometry3d::Identity(), inf);
        }
      }
      
      if(i == 0 && keyframes.empty()) {
        continue;
      }

      /***** Scan-to-Scan Add edge to between consecutive keyframes *****/
      const auto& prev_keyframe = i == 0 ? keyframes.back() : keyframe_queue[i - 1];
      // relative pose between odom of previous frame and this frame R2=R12*R1 => R12 = inv(R2) * R1
      Eigen::Isometry3d relative_pose = keyframe->odom_scan2scan.inverse() * prev_keyframe->odom_scan2scan;
      // calculate fitness score as information 
      Eigen::MatrixXd information = inf_calclator->calc_information_matrix(keyframe->cloud, prev_keyframe->cloud, relative_pose);
      auto edge = graph_slam->add_se3_edge(keyframe->node, prev_keyframe->node, relative_pose, information);
      // cout << information << endl;
      graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("odometry_edge_robust_kernel", "NONE"), private_nh.param<double>("odometry_edge_robust_kernel_size", 1.0));
      

      /*if (enable_preintegration){
        // Add Preintegration edge
        geometry_msgs::Transform relative_trans = keyframe->trans_integrated;
        g2o::SE3Quat relative_se3quat ( Eigen::Quaterniond(relative_trans.rotation.w, relative_trans.rotation.x, relative_trans.rotation.y, relative_trans.rotation.z), 
                                        Eigen::Vector3d(relative_trans.translation.x, relative_trans.translation.y, relative_trans.translation.z));
        Eigen::Isometry3d relative_isometry = transform2isometry(relative_trans);
        Eigen::MatrixXd information_integ = Eigen::MatrixXd::Identity(6, 6);
        information_integ <<  1.0 / preinteg_trans_stddev, 0, 0, 0, 0, 0,
                              0, 1.0 / preinteg_trans_stddev, 0, 0, 0, 0,
                              0, 0, 1.0 / preinteg_trans_stddev, 0, 0, 0,
                              0, 0, 0, 1.0 / preinteg_orient_stddev, 0, 0,
                              0, 0, 0, 0, 1.0 / preinteg_orient_stddev, 0,
                              0, 0, 0, 0, 0, 1.0 / preinteg_orient_stddev; 
        auto edge_integ = graph_slam->add_se3_edge(prev_keyframe->node, keyframe->node, relative_isometry, information_integ);
        graph_slam->add_robust_kernel(edge_integ, private_nh.param<std::string>("integ_edge_robust_kernel", "NONE"), private_nh.param<double>("integ_edge_robust_kernel_size", 1.0));
      }*/
    }

    std_msgs::Header read_until;
    read_until.stamp = keyframe_queue[num_processed]->stamp + ros::Duration(10, 0);
    read_until.frame_id = points_topic;
    read_until_pub.publish(read_until);
    read_until.frame_id = "/filtered_points";
    read_until_pub.publish(read_until);

    keyframe_queue.erase(keyframe_queue.begin(), keyframe_queue.begin() + num_processed + 1);
    return true;
  }


  /**
   * @brief Back-end Optimization. This methods adds all the data in the queues to the pose graph, and then optimizes the pose graph
   * @param event
   */
  void optimization_timer_callback(const ros::WallTimerEvent& event) {
    std::lock_guard<std::mutex> lock(main_thread_mutex);

    // add keyframes_ and floor coeffs in the queues to the pose graph
    bool keyframe_updated = flush_keyframe_queue();

    if(!keyframe_updated) {
      std_msgs::Header read_until;
      read_until.stamp = ros::Time::now() + ros::Duration(30, 0);
      read_until.frame_id = points_topic;
      read_until_pub.publish(read_until);
      read_until.frame_id = "/filtered_points";
      read_until_pub.publish(read_until);
    }

    //if(!keyframe_updated & !flush_gps_queue() & !flush_barometer_queue()) {
    //  return;
    //}
    
    // loop detection
    if(private_nh.param<bool>("enable_loop_closure", false)){
      std::vector<Loop::Ptr> loops = loop_detector->detect(keyframes, new_keyframes, *graph_slam);
    }

    // Copy "new_keyframes_" to vector  "keyframes_", "new_keyframes_" was used for loop detaction 
    std::copy(new_keyframes.begin(), new_keyframes.end(), std::back_inserter(keyframes));
    new_keyframes.clear();

    if(private_nh.param<bool>("enable_loop_closure", false))
      addLoopFactor();

    // move the first node / position to the current estimate of the first node pose
    // so the first node moves freely while trying to stay around the origin
    if(anchor_node && private_nh.param<bool>("fix_first_node_adaptive", true)) {
      Eigen::Isometry3d anchor_target = static_cast<g2o::VertexSE3*>(anchor_edge->vertices()[1])->estimate();
      anchor_node->setEstimate(anchor_target);
    }

    // optimize the pose graph
    int num_iterations = private_nh.param<int>("g2o_solver_num_iterations", 1024);
    clock_t start_ms = clock();
    graph_slam->optimize(num_iterations);
    clock_t end_ms = clock();
    double time_used = double(end_ms - start_ms) / CLOCKS_PER_SEC;
    opt_time.push_back(time_used);

    //********** publish tf **********
    const auto& keyframe = keyframes.back();
    // RadarOdom_to_base = map_to_base * map_to_RadarOdom^(-1)
    Eigen::Isometry3d trans = keyframe->node->estimate() * keyframe->odom_scan2scan.inverse();
    Eigen::Isometry3d map2base_trans = keyframe->node->estimate();

    trans_odom2map_mutex.lock();
    trans_odom2map = trans.matrix();
    // map2base_incremental = map2base_last^(-1) * map2base_this 
    trans_aftmapped_incremental = trans_aftmapped.inverse() * map2base_trans;
    trans_aftmapped = map2base_trans;
    trans_odom2map_mutex.unlock();

    std::vector<KeyFrameSnapshot::Ptr> snapshot(keyframes.size());
    std::transform(keyframes.begin(), keyframes.end(), snapshot.begin(), [=](const KeyFrame::Ptr& k) { return std::make_shared<KeyFrameSnapshot>(k); });

    keyframes_snapshot_mutex.lock();
    keyframes_snapshot.swap(snapshot);
    keyframes_snapshot_mutex.unlock();
    graph_updated = true;

    // Publish After-Mapped Odometry
    nav_msgs::Odometry aft = isometry2odom(keyframe->stamp, trans_aftmapped, mapFrame, odometryFrame);
    aftmapped_odom_pub.publish(aft);

    // Publish After-Mapped Odometry Incrementation
    nav_msgs::Odometry aft_incre = isometry2odom(keyframe->stamp, trans_aftmapped_incremental, mapFrame, odometryFrame);
    aftmapped_odom_incremenral_pub.publish(aft_incre);

    // Publish /odom to /base_link
    if(odom2base_pub.getNumSubscribers()) {  // Returns the number of subscribers that are currently connected to this Publisher
      geometry_msgs::TransformStamped ts = matrix2transform(keyframe->stamp, trans.matrix(), mapFrame, odometryFrame);
      odom2base_pub.publish(ts);
    }

    if(markers_pub.getNumSubscribers()) {
      auto markers = create_marker_array(ros::Time::now());
      markers_pub.publish(markers);
    }
  }

  void addLoopFactor()
  {
    //cout << "Add Loop Factor HERE" << endl;
    if (loop_detector->loopIndexQueue.empty())
      return;
    for (int i = 0; i < (int)loop_detector->loopIndexQueue.size(); ++i){
      int indexFrom = loop_detector->loopIndexQueue[i].first;
      int indexTo = loop_detector->loopIndexQueue[i].second;
      Eigen::Isometry3d poseBetween = loop_detector->loopPoseQueue[i];
      Eigen::MatrixXd information_matrix = loop_detector->loopInfoQueue[i];
      auto edge = graph_slam->add_se3_edge(keyframes[indexFrom]->node, keyframes[indexTo]->node, poseBetween, information_matrix);
      graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("loop_closure_edge_robust_kernel", "NONE"), private_nh.param<double>("loop_closure_edge_robust_kernel_size", 1.0));
    }
    // loopIndexQueue.clear();
    // loopPoseQueue.clear();
    // loopInfoQueue.clear();
    // aLoopIsClosed = true;
  }

  /**
   * @brief generate map point cloud and publish it
   * @param event
   */
  void map_points_publish_timer_callback(const ros::WallTimerEvent& event) {
    if(!map_points_pub.getNumSubscribers() || !graph_updated) {
      return;
    }
    std::vector<KeyFrameSnapshot::Ptr> snapshot;
    keyframes_snapshot_mutex.lock();
    snapshot = keyframes_snapshot;
    keyframes_snapshot_mutex.unlock();

    auto cloud = map_cloud_generator->generate(snapshot, map_cloud_resolution);
    if(!cloud) {
      return;
    }

    /*// Initialize TransformListener
    static tf::TransformListener listener;
    tf::StampedTransform transform;
    try {
      listener.lookupTransform(mapFrame, baselinkFrame, ros::Time(0), transform);
    } catch (tf::TransformException &ex) {
      ROS_WARN("%s", ex.what());
      return;
    }

    // Get the z-coordinate of the baselink frame
    double baselink_z = transform.getOrigin().z();

    // Filter points below the height of the baselink frame
    cout << "Filter Points below z: " << baselink_z - 1.0 << endl;
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits( -7.0, std::numeric_limits<float>::max()); // Set the lower limit to baselink_z
    pass.filter(*cloud); // Apply the filter directly to the existing cloud*/

    cloud->header.frame_id = mapFrame;
    cloud->header.stamp = snapshot.back()->cloud->header.stamp;

    sensor_msgs::PointCloud2Ptr cloud_msg(new sensor_msgs::PointCloud2());
    pcl::toROSMsg(*cloud, *cloud_msg);

    map_points_pub.publish(cloud_msg);
  }

  /**
   * @brief create visualization marker
   * @param stamp
   * @return
   */
  visualization_msgs::MarkerArray create_marker_array(const ros::Time& stamp) const {
    visualization_msgs::MarkerArray markers;
    if (show_sphere)
      markers.markers.resize(5);
    else
      markers.markers.resize(4);

    // loop edges
    visualization_msgs::Marker& loop_marker = markers.markers[0];
    loop_marker.header.frame_id = "map";
    loop_marker.header.stamp = stamp;
    loop_marker.action = visualization_msgs::Marker::ADD;
    loop_marker.type = visualization_msgs::Marker::LINE_LIST;
    loop_marker.ns = "loop_edges";
    loop_marker.id = 1;
    loop_marker.pose.orientation.w = 1;
    loop_marker.scale.x = 0.1; loop_marker.scale.y = 0.1; loop_marker.scale.z = 0.1;
    loop_marker.color.r = 0.9; loop_marker.color.g = 0.9; loop_marker.color.b = 0;
    loop_marker.color.a = 1;
    for (auto it = loop_detector->loopIndexContainer.begin(); it != loop_detector->loopIndexContainer.end(); ++it)
    {
      int key_cur = it->first;
      int key_pre = it->second;
      geometry_msgs::Point p;
      Eigen::Vector3d pos = keyframes[key_cur]->node->estimate().translation();
      p.x = pos.x();
      p.y = pos.y();
      p.z = pos.z();
      loop_marker.points.push_back(p);
      pos = keyframes[key_pre]->node->estimate().translation();
      p.x = pos.x();
      p.y = pos.y();
      p.z = pos.z();
      loop_marker.points.push_back(p);
    }

    // node markers
    visualization_msgs::Marker& traj_marker = markers.markers[1];
    traj_marker.header.frame_id = "map";
    traj_marker.header.stamp = stamp;
    traj_marker.ns = "nodes";
    traj_marker.id = 0;
    traj_marker.type = visualization_msgs::Marker::SPHERE_LIST;

    traj_marker.pose.orientation.w = 1.0;
    traj_marker.scale.x = traj_marker.scale.y = traj_marker.scale.z = 0.3;

    visualization_msgs::Marker& imu_marker = markers.markers[2];
    imu_marker.header = traj_marker.header;
    imu_marker.ns = "imu";
    imu_marker.id = 1;
    imu_marker.type = visualization_msgs::Marker::SPHERE_LIST;

    imu_marker.pose.orientation.w = 1.0;
    imu_marker.scale.x = imu_marker.scale.y = imu_marker.scale.z = 0.75;

    traj_marker.points.resize(keyframes.size());
    traj_marker.colors.resize(keyframes.size());
    for(size_t i = 0; i < keyframes.size(); i++) {
      Eigen::Vector3d pos = keyframes[i]->node->estimate().translation();
      traj_marker.points[i].x = pos.x();
      traj_marker.points[i].y = pos.y();
      traj_marker.points[i].z = pos.z();

      double p = static_cast<double>(i) / keyframes.size();
      traj_marker.colors[i].r = 0.0;//1.0 - p;
      traj_marker.colors[i].g = 1.0;//p;
      traj_marker.colors[i].b = 0.0;
      traj_marker.colors[i].a = 1.0;

      if(keyframes[i]->acceleration) {
        Eigen::Vector3d pos = keyframes[i]->node->estimate().translation();
        geometry_msgs::Point point;
        point.x = pos.x();
        point.y = pos.y();
        point.z = pos.z();

        std_msgs::ColorRGBA color;
        color.r = 0.0;
        color.g = 0.0;
        color.b = 1.0;
        color.a = 0.1;

        imu_marker.points.push_back(point);
        imu_marker.colors.push_back(color);
      }
    }

    // edge markers
    visualization_msgs::Marker& edge_marker = markers.markers[3];
    edge_marker.header.frame_id = "map";
    edge_marker.header.stamp = stamp;
    edge_marker.ns = "edges";
    edge_marker.id = 2;
    edge_marker.type = visualization_msgs::Marker::LINE_LIST;

    edge_marker.pose.orientation.w = 1.0;
    edge_marker.scale.x = 0.05;

    edge_marker.points.resize(graph_slam->graph->edges().size() * 2);
    edge_marker.colors.resize(graph_slam->graph->edges().size() * 2);

    auto edge_itr = graph_slam->graph->edges().begin();
    for(int i = 0; edge_itr != graph_slam->graph->edges().end(); edge_itr++, i++) {
      g2o::HyperGraph::Edge* edge = *edge_itr;
      g2o::EdgeSE3* edge_se3 = dynamic_cast<g2o::EdgeSE3*>(edge);
      if(edge_se3) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_se3->vertices()[0]);
        g2o::VertexSE3* v2 = dynamic_cast<g2o::VertexSE3*>(edge_se3->vertices()[1]);
        
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2 = v2->estimate().translation();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z();
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        double p1 = static_cast<double>(v1->id()) / graph_slam->graph->vertices().size();
        double p2 = static_cast<double>(v2->id()) / graph_slam->graph->vertices().size();
        edge_marker.colors[i * 2].r = 0.0;//1.0 - p1;
        edge_marker.colors[i * 2].g = 1.0;//p1;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 0.0;//1.0 - p2;
        edge_marker.colors[i * 2 + 1].g = 1.0;//p2;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        if(std::abs(v1->id() - v2->id()) > 2) {
          // edge_marker.points[i * 2].z += 0.5;
          // edge_marker.points[i * 2 + 1].z += 0.5;
          edge_marker.colors[i * 2].r = 0.9;
          edge_marker.colors[i * 2].g = 0.9;
          edge_marker.colors[i * 2].b = 0.0;
          edge_marker.colors[i * 2 + 1].r = 0.9;
          edge_marker.colors[i * 2 + 1].g = 0.9;
          edge_marker.colors[i * 2 + 1].b = 0.0;
          edge_marker.colors[i * 2].a = 0.0;
          edge_marker.colors[i * 2 + 1].a += 0.0;
        }
        continue;
      }

      g2o::EdgeSE3Plane* edge_plane = dynamic_cast<g2o::EdgeSE3Plane*>(edge);
      if(edge_plane) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_plane->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2(pt1.x(), pt1.y(), 0.0);

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z();
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        edge_marker.colors[i * 2].b = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].b = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }

      g2o::EdgeSE3PriorXY* edge_priori_xy = dynamic_cast<g2o::EdgeSE3PriorXY*>(edge);
      if(edge_priori_xy) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_priori_xy->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2 = Eigen::Vector3d::Zero();
        pt2.head<2>() = edge_priori_xy->measurement();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z() + 0.5;
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z() + 0.5;

        edge_marker.colors[i * 2].r = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }

      g2o::EdgeSE3PriorXYZ* edge_priori_xyz = dynamic_cast<g2o::EdgeSE3PriorXYZ*>(edge);
      if(edge_priori_xyz) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_priori_xyz->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2 = edge_priori_xyz->measurement();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z() + 0.5;
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        edge_marker.colors[i * 2].r = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }
    }

    if (show_sphere)
    {
      // sphere
      visualization_msgs::Marker& sphere_marker = markers.markers[4];
      sphere_marker.header.frame_id = "map";
      sphere_marker.header.stamp = stamp;
      sphere_marker.ns = "loop_close_radius";
      sphere_marker.id = 3;
      sphere_marker.type = visualization_msgs::Marker::SPHERE;

      if(!keyframes.empty()) {
        Eigen::Vector3d pos = keyframes.back()->node->estimate().translation();
        sphere_marker.pose.position.x = pos.x();
        sphere_marker.pose.position.y = pos.y();
        sphere_marker.pose.position.z = pos.z();
      }
      sphere_marker.pose.orientation.w = 1.0;
      sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = loop_detector->get_distance_thresh() * 2.0;

      sphere_marker.color.r = 1.0;
      sphere_marker.color.a = 0.3;
    }

    return markers;
  }

  /**
   * @brief dump all data to the current directory
   * @param req
   * @param res
   * @return
   */
  bool dump_service(radar_graph_slam::DumpGraphRequest& req, radar_graph_slam::DumpGraphResponse& res) {
    std::lock_guard<std::mutex> lock(main_thread_mutex);

    std::string directory = req.destination;

    if(directory.empty()) {
      std::array<char, 64> buffer;
      buffer.fill(0);
      time_t rawtime;
      time(&rawtime);
      const auto timeinfo = localtime(&rawtime);
      strftime(buffer.data(), sizeof(buffer), "%d-%m-%Y %H:%M:%S", timeinfo);
    }

    if(!boost::filesystem::is_directory(directory)) {
      boost::filesystem::create_directory(directory);
    }

    std::cout << "all data dumped to:" << directory << std::endl;

    graph_slam->save(directory + "/graph.g2o");
    for(size_t i = 0; i < keyframes.size(); i++) {
      std::stringstream sst;
      sst << boost::format("%s/%06d") % directory % i;

      keyframes[i]->save(sst.str());
    }

    if(zero_utm) {
      std::ofstream zero_utm_ofs(directory + "/zero_utm");
      zero_utm_ofs << boost::format("%.6f %.6f %.6f") % zero_utm->x() % zero_utm->y() % zero_utm->z() << std::endl;
    }

    std::ofstream ofs(directory + "/special_nodes.csv");
    ofs << "anchor_node " << (anchor_node == nullptr ? -1 : anchor_node->id()) << std::endl;
    ofs << "anchor_edge " << (anchor_edge == nullptr ? -1 : anchor_edge->id()) << std::endl;
    ofs << "floor_node " << (floor_plane_node == nullptr ? -1 : floor_plane_node->id()) << std::endl;

    res.success = true;
    return true;
  }

  /**
   * @brief save map data as pcd
   * @param req
   * @param res
   * @return
   */
  bool save_map_service(radar_graph_slam::SaveMapRequest& req, radar_graph_slam::SaveMapResponse& res) {
    std::vector<KeyFrameSnapshot::Ptr> snapshot;

    keyframes_snapshot_mutex.lock();
    snapshot = keyframes_snapshot;
    keyframes_snapshot_mutex.unlock();

    auto cloud = map_cloud_generator->generate(snapshot, req.resolution);
    if(!cloud) {
      res.success = false;
      return true;
    }

    if(zero_utm && req.utm) {
      for(auto& pt : cloud->points) {
        pt.getVector3fMap() += (*zero_utm).cast<float>();
      }
    }

    cloud->header.frame_id = mapFrame;
    cloud->header.stamp = snapshot.back()->cloud->header.stamp;

    if(zero_utm) {
      std::ofstream ofs(req.destination + ".utm");
      ofs << boost::format("%.6f %.6f %.6f") % zero_utm->x() % zero_utm->y() % zero_utm->z() << std::endl;
    }

    int ret = pcl::io::savePCDFileBinary(req.destination, *cloud);
    res.success = ret == 0;

    return true;
  }

  
  
  void command_callback(const std_msgs::String& str_msg) {
    if (str_msg.data == "output_aftmapped") {
      ofstream fout;
      fout.open("/home/zhuge/stamped_pose_graph_estimate.txt", ios::out);
      fout << "# timestamp tx ty tz qx qy qz qw" << endl;
      fout.setf(ios::fixed, ios::floatfield);  // fixed mode，float
      fout.precision(8);  // Set precision 8
      for(size_t i = 0; i < keyframes.size(); i++) {
        Eigen::Vector3d pos_ = keyframes[i]->node->estimate().translation();
        Eigen::Matrix3d rot_ = keyframes[i]->node->estimate().rotation();
        Eigen::Quaterniond quat_(rot_);
        double timestamp = keyframes[i]->stamp.toSec();
        double tx = pos_(0), ty = pos_(1), tz = pos_(2);
        double qx = quat_.x(), qy = quat_.y(), qz = quat_.z(), qw = quat_.w();

        fout << timestamp << " "
          << tx << " " << ty << " " << tz << " "
          << qx << " " << qy << " " << qz << " " << qw << endl;
      }
      fout.close();
      ROS_INFO("Optimized edges have been output!");
    }
    else if (str_msg.data == "time") {
      if (loop_detector->pf_time.size() > 0) {
        std::sort(loop_detector->pf_time.begin(), loop_detector->pf_time.end());
        double median = loop_detector->pf_time.at(floor((double)loop_detector->pf_time.size() / 2));
        cout << "Pre-filtering Matching time cost (median): " << median << endl;
      }
      if (loop_detector->sc_time.size() > 0) {
        std::sort(loop_detector->sc_time.begin(), loop_detector->sc_time.end());
        double median = loop_detector->sc_time.at(floor((double)loop_detector->sc_time.size() / 2));
        cout << "Scan Context time cost (median): " << median << endl;
      }
      if (loop_detector->oc_time.size() > 0) {
        std::sort(loop_detector->oc_time.begin(), loop_detector->oc_time.end());
        double median = loop_detector->oc_time.at(floor((double)loop_detector->oc_time.size() / 2));
        cout << "Odometry Check time cost (median): " << median << endl;
      }
      if (opt_time.size() > 0) {
        std::sort(opt_time.begin(), opt_time.end());
        double median = opt_time.at(floor((double)opt_time.size() / 2));
        cout << "Optimization time cost (median): " << median << endl;
      }
    }
  }


private:
  // ROS
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;
  ros::WallTimer optimization_timer;
  ros::WallTimer map_publish_timer;

  std::unique_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub;
  std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> cloud_sub;
  std::unique_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync;

  //ros::Subscriber barometer_sub;
  //ros::Subscriber gps_sub;
  //ros::Subscriber nmea_sub;
  //ros::Subscriber navsat_sub;

  //ros::Subscriber imu_odom_sub;
  //ros::Subscriber imu_sub;
  ros::Subscriber command_sub;

  //ros::Publisher imu_odom_pub;
  ros::Publisher markers_pub;

  std::mutex trans_odom2map_mutex;
  Eigen::Matrix4d trans_odom2map; // keyframe->node->estimate() * keyframe->odom.inverse();
  Eigen::Isometry3d trans_aftmapped;  // Odometry from /map to /base_link
  Eigen::Isometry3d trans_aftmapped_incremental;
  ros::Publisher odom2base_pub;
  ros::Publisher aftmapped_odom_pub;
  ros::Publisher aftmapped_odom_incremenral_pub;
  ros::Publisher odom_frame2frame_pub;

  std::string points_topic;
  ros::Publisher read_until_pub;
  ros::Publisher map_points_pub;

  tf::TransformListener tf_listener;
  tf::TransformBroadcaster map2base_broadcaster; // odom_frame => base_frame

  ros::ServiceServer dump_service_server;
  ros::ServiceServer save_map_service_server; 

  // keyframe queue
  std::mutex keyframe_queue_mutex;
  std::deque<KeyFrame::Ptr> keyframe_queue;
  std::deque<geometry_msgs::TwistStampedConstPtr> twist_queue;
  std::deque<nav_msgs::OdometryConstPtr> imu_odom_queue;
  double thisKeyframeTime;
  double lastKeyframeTime;
  size_t keyframe_index = 0;
  bool subsurface_removal_filter;

  // IMU / Ego Velocity Integration
  bool enable_preintegration;
  double preinteg_orient_stddev;
  double preinteg_trans_stddev;
  bool enable_imu_orientation;
  bool use_egovel_preinteg_trans;
  Eigen::Matrix4d initial_pose;

  // barometer queue
  bool enable_barometer;
  int barometer_edge_type;
  double barometer_edge_stddev;
  boost::optional<Eigen::Vector1d> zero_alt;
  std::mutex barometer_queue_mutex;
  std::deque<barometer_bmp388::BarometerConstPtr> barometer_queue;

  // gps queue
  int gps_edge_intervals;
  int last_gps_edge_index;
  double gps_time_offset;
  double gps_edge_stddev_xy;
  double gps_edge_stddev_z;
  double max_gps_edge_stddev_xy;
  double max_gps_edge_stddev_z;
  boost::optional<Eigen::Vector3d> zero_utm;
  std::mutex gps_queue_mutex;
  std::deque<geographic_msgs::GeoPointStampedConstPtr> gps_geopoint_queue;
  std::deque<sensor_msgs::NavSatFix>           gps_navsat_queue;
  Eigen::Matrix4d utm_to_world;
  std::string dataset_name;

  // Marker coefficients
  bool show_sphere;

  // for map cloud generation
  std::atomic_bool graph_updated;
  double map_cloud_resolution;
  std::mutex keyframes_snapshot_mutex;
  std::vector<KeyFrameSnapshot::Ptr> keyframes_snapshot;
  std::unique_ptr<MapCloudGenerator> map_cloud_generator;

  // graph slam
  // all the below members must be accessed after locking main_thread_mutex
  std::mutex main_thread_mutex;

  int max_keyframes_per_update;
  //  Used for Loop Closure detection source, 
  //  pushed form keyframe_queue at "flush_keyframe_queue()", 
  //  inserted to "keyframes_" before optimization
  std::deque<KeyFrame::Ptr> new_keyframes;
  //  Previous keyframes_
  std::vector<KeyFrame::Ptr> keyframes;
  std::unordered_map<ros::Time, KeyFrame::Ptr, RosTimeHash> keyframe_hash;
  g2o::VertexSE3* anchor_node;
  g2o::EdgeSE3* anchor_edge;
  g2o::VertexPlane* floor_plane_node;

  std::unique_ptr<GraphSLAM> graph_slam;
  std::unique_ptr<LoopDetector> loop_detector;
  std::unique_ptr<KeyframeUpdater> keyframe_updater;
  std::unique_ptr<NmeaSentenceParser> nmea_parser;
  std::unique_ptr<InformationMatrixCalculator> inf_calclator;

  // Registration Method
  pcl::Registration<PointT, PointT>::Ptr registration;
  pcl::KdTreeFLANN<PointT>::Ptr kdtreeHistoryKeyPoses;

  std::vector<double> opt_time;
};

}  // namespace radar_graph_slam

PLUGINLIB_EXPORT_CLASS(radar_graph_slam::RadarGraphSlamNodelet, nodelet::Nodelet)
