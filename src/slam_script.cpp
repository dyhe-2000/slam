#include <iostream>
#include <chrono>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <stdlib.h> 
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "slam/buffer_subscriber.hpp"
#include "slam/Timer.hpp"
#include <cv_bridge/cv_bridge.h>
#include <cmath>
#include <math.h>
#include <algorithm>
#include <Eigen/LU>
#include <Eigen/QR>
#include <mutex>
#include <tuple>
#include "livox_interfaces/msg/custom_msg.hpp"
#include "livox_interfaces/msg/custom_point.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include <fstream>
#include <random>
#include <time.h>
using std::placeholders::_1;
std::mutex mtx;

#define MAP_SIZE 400 // will +1 for making origin
#define MAP_RESOLUTION 0.05
#define NUM_PARTICLES 1 //175
#define X_VARIANCE 0.0 //0.01
#define Y_VARIANCE 0.0 //0.01
#define THETA_VARIANCE 0.0 //0.0325
#define NEFF_THRESH 140.0 // less than this will trigger resample

// <x, y>
void bresenham2d(std::pair<std::pair<double, double>, std::pair<double, double>> theData, std::vector<std::pair<int, int>> *theVector) {
	int y1, x1, y2, x2;
	int dx, dy, sx, sy;
	int e2;
	int error;

	x1 = int(theData.first.first);
	y1 = int(theData.first.second);
	x2 = int(theData.second.first);
	y2 = int(theData.second.second);

	dx = abs(x1 - x2);
	dy = -abs(y1 - y2);

	sx = x1 < x2 ? 1 : -1;
	sy = y1 < y2 ? 1 : -1;

	error = dx + dy;

	while (1) {
		theVector->push_back(std::pair<int, int>(x1, y1));
		if (x1 == x2 && y1 == y2)
			break;
		e2 = 2 * error;

		if (e2 >= dy) {
			if (x2 == x1) break;
			error = error + dy;
			x1 = x1 + sx;
		}
		if (e2 <= dx) {
			if (y2 == y1) break;
			error = error + dx;
			y1 = y1 + sy;
		}
	}
}

Eigen::MatrixXd frame_transformation(Eigen::MatrixXd& T, Eigen::MatrixXd& x){
	// T is 4 by 4
	// x is 2 by n
	//std::cout << "T: \n" << T << std::endl;
	//std::cout << "x: \n" << x << std::endl;
	int n = int(x.cols());
	Eigen::MatrixXd homo_vecs(4, n); //4 by n matrix
	homo_vecs = Eigen::MatrixXd::Zero(4,n);
	Eigen::MatrixXd homo_outs(4, n); //4 by n matrix
	homo_outs = Eigen::MatrixXd::Zero(4,n);
	Eigen::MatrixXd ones(1,n);
	ones = Eigen::MatrixXd::Ones(1,n);
	homo_vecs.block(0, 0, 2, n) = x;
	homo_vecs.block(2, 0, 1, n) = Eigen::MatrixXd::Zero(1,n);
	homo_vecs.block(3, 0, 1, n) = ones;
	//std::cout << "homo_vecs: \n" << homo_vecs << std::endl;
	homo_outs = T*homo_vecs;
	//std::cout << "homo_outs: \n" << homo_outs << std::endl;
	return homo_outs;
	//return homo_outs.block(0, 0, 2, n);
}

class slam_node : public rclcpp::Node{
protected:
	// publisher
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub_;
	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr trajectory_map_pub_;
	rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr coord_pub_; // x and y are in integer coord
        
	// timer
	rclcpp::TimerBase::SharedPtr map_sending_timer_;
	rclcpp::TimerBase::SharedPtr step_timer_;
	Timer check_duration_timer;
        
	// parameter
	uint32_t map_publisher_freq;
	double dt_;
	rclcpp::Time last_step_time_;
	bool Lidar_Valid;
	bool firstScan;
	cv::Mat Map = cv::Mat(MAP_SIZE+1, MAP_SIZE+1, CV_32FC1, cv::Scalar(0)); // each grid length MAP_RESOLUTION m
	cv::Mat Trajectory_Map = cv::Mat(MAP_SIZE+1, MAP_SIZE+1, CV_8UC3, cv::Scalar(0, 0, 0)); // each grid length MAP_RESOLUTION m
	double log_odds_map[MAP_SIZE + 1][MAP_SIZE + 1] = {};
	double Map_resolution = MAP_RESOLUTION;
	Eigen::MatrixXd x = Eigen::MatrixXd::Zero(3,NUM_PARTICLES); // x,y,theta in world frame, initially 0
	Eigen::MatrixXd mapTworld = Eigen::MatrixXd::Zero(4,4);
	Eigen::MatrixXd bodyTlidar = Eigen::MatrixXd::Zero(4,4);
	Eigen::MatrixXd worldTbody = Eigen::MatrixXd::Zero(4,4);
	rclcpp::Time previous_read_imu_message_time_stamp;
	long long int step_counter;
	geometry_msgs::msg::Vector3 coord_message; 

	// msg struct
	livox_interfaces::msg::CustomMsg Lidar_measurement;
    std_msgs::msg::Float64 Lat_measurement;

	// buffer subscriber
	MsgSubscriber<livox_interfaces::msg::CustomMsg>::UniquePtr lidar_sub;
    MsgSubscriber<std_msgs::msg::Float64>::UniquePtr lat_sub;
public:
	explicit slam_node(const rclcpp::NodeOptions & options): Node("slam_node", options) {
		std::cout << "this is boat project slam node" << std::endl;

		// initialize mapTworld
		mapTworld(0,0) = 1/Map_resolution;
		mapTworld(1,1) = 1/Map_resolution;
		mapTworld(2,2) = 1;
		mapTworld(3,3) = 1;
		mapTworld(0,3) = MAP_SIZE/2;
		mapTworld(1,3) = MAP_SIZE/2;
		std::cout << "mapTworld: \n" << mapTworld << std::endl;

		// initialize bodyTlidar
		bodyTlidar(0,0) = cos(3.1415926535897932384626433832795);
		bodyTlidar(0,1) = -sin(3.1415926535897932384626433832795);
		bodyTlidar(1,0) = sin(3.1415926535897932384626433832795);
		bodyTlidar(1,1) = cos(3.1415926535897932384626433832795);
		bodyTlidar(2,2) = 1;
		bodyTlidar(3,3) = 1;
		std::cout << "bodyTlidar: \n" << bodyTlidar << std::endl;

		// initialize worldTbody
		worldTbody(0,0) = 1;
		worldTbody(1,1) = worldTbody(2,2) = worldTbody(3,3) = worldTbody(0,0);
		std::cout << "worldTbody: \n" << worldTbody << std::endl;

		// initialize coord_message
		coord_message.x = 0;
		coord_message.y = 0;
		coord_message.z = 0;

		// initialize x
        this->x(0,0) = 0;
        this->x(1,0) = 0;
        this->x(2,0) = 0;
		std::cout << "x: \n" << this->x << std::endl;

		declare_parameter("map_publisher_freq", 10);
        get_parameter("map_publisher_freq", this->map_publisher_freq);
		declare_parameter("dt", 0.005);
		get_parameter("dt", this->dt_);

		// initialize the member publisher
		this->img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("slam_map", 10);
		this->trajectory_map_pub_ = this->create_publisher<sensor_msgs::msg::Image>("slam_trajectory_map", 10);
		this->coord_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>("slam_coord", 10);
		
		// initialize the buffer subscribers
		subscribe_from(this, lidar_sub, "/livox/lidar");
        subscribe_from(this, lat_sub, "/wamv/sensors/gps/lat");

		// project first lidar measurement, in this case wTb is identity matrix because initially body frame is world frame
		this->map_sending_timer_ = this->create_wall_timer(std::chrono::milliseconds((uint32_t)(1.0/this->map_publisher_freq*1000)), std::bind(&slam_node::publish_map, this));
		// initialize the timer
		this->step_timer_ = rclcpp::create_timer(this, get_clock(), std::chrono::duration<float>(this->dt_), [this] {step();});
	}

	~slam_node(){ // destructor save the occupancy map and log odds map

	}

    void publish_map() { // running repeatedly with the timer set frequency
		mtx.lock();
		
		//std::cout << "Particle Filter SLAM publishing map" << std::endl;
		// OpenCV example (matplotlib equivalent)
		// cv::Mat single_channel_image(5, 5, CV_32FC1, cv::Scalar(0));
		// float* single_channel_pixel_ptr = (float*)single_channel_image.data;
		// single_channel_pixel_ptr[2*single_channel_image.cols + 2] = 0.8;
		// single_channel_pixel_ptr[0*single_channel_image.cols + 0] = 0.6;
		// cv::threshold(single_channel_image, single_channel_image, 0.5, 255.0, 0);
		cv::Mat single_channel_image = this->Map;
		single_channel_image.convertTo(single_channel_image, CV_8UC1);

		// example of publishing image out
		// Avoid copying image message if possible
		sensor_msgs::msg::Image::UniquePtr image_msg(new sensor_msgs::msg::Image());
		auto stamp = now();
		// Convert OpenCV Mat to ROS Image
		image_msg->header.stamp = stamp;
		// image_msg->header.frame_id = cxt_.camera_frame_id_;
		image_msg->height = single_channel_image.rows;
		image_msg->width = single_channel_image.cols;
		image_msg->encoding = "8UC1";
		image_msg->is_bigendian = false;
		image_msg->step = static_cast<sensor_msgs::msg::Image::_step_type>(single_channel_image.step);
		image_msg->data.assign(single_channel_image.datastart,single_channel_image.dataend);
		img_pub_->publish(std::move(image_msg));

		//this->Trajectory_Map; // 8UC3
		sensor_msgs::msg::Image::UniquePtr trajectory_map_msg(new sensor_msgs::msg::Image());
		stamp = now();
		trajectory_map_msg->header.stamp = stamp;
		trajectory_map_msg->height = this->Trajectory_Map.rows;
		trajectory_map_msg->width = this->Trajectory_Map.cols;
		trajectory_map_msg->encoding = "8UC3";
		trajectory_map_msg->is_bigendian = false;
		trajectory_map_msg->step = static_cast<sensor_msgs::msg::Image::_step_type>(this->Trajectory_Map.step);
		trajectory_map_msg->data.assign(this->Trajectory_Map.datastart,this->Trajectory_Map.dataend);
		trajectory_map_pub_->publish(std::move(trajectory_map_msg));


		// publish x and y coord in map
		coord_pub_->publish(this->coord_message);
		mtx.unlock();
    }

    void step(){
		this->check_duration_timer.reset();
		mtx.lock();
		
        if(this->lidar_sub->has_msg()){
            std::cout << "got a lidar measurement" << std::endl;
            this->Lidar_measurement = *(this->lidar_sub->take());
            std::cout << "point_num: " << Lidar_measurement.point_num << std::endl;

            float* Map_pixel_ptr = (float*)Map.data;
            for(int j = 0; j < MAP_SIZE+1; ++j){
                for(int k = 0; k < MAP_SIZE+1; ++k){
                    Map_pixel_ptr[j*Map.cols + k] = 0.0;
                }
            }

            for(int i = 0; i < Lidar_measurement.point_num; ++i){
                if(Lidar_measurement.points[i].x != 0 || Lidar_measurement.points[i].y != 0 || Lidar_measurement.points[i].z != 0){
                    if(Lidar_measurement.points[i].z <= 1 && Lidar_measurement.points[i].z >= -1){
                        if(Lidar_measurement.points[i].x <= 10 && Lidar_measurement.points[i].x >= -10 && Lidar_measurement.points[i].y <= 10 && Lidar_measurement.points[i].y >= -10){
                            //std::cout << "x: " << Lidar_measurement.points[i].x << std::endl;
                            //std::cout << "y: " << Lidar_measurement.points[i].y << std::endl;
                            //std::cout << "z: " << Lidar_measurement.points[i].z << std::endl;

                            Eigen::MatrixXd x_coord = Eigen::MatrixXd::Zero(2,1);
                            x_coord(0,0) = Lidar_measurement.points[i].x;
		                    x_coord(1,0) = Lidar_measurement.points[i].y;
		                    Eigen::MatrixXd x_map_frame = frame_transformation(this->mapTworld,x_coord);
                            int x_index = int(round(x_map_frame(0,0)));
		                    int y_index = int(round(x_map_frame(1,0)));

                            Map_pixel_ptr[y_index*Map.cols + x_index] = 255.0;
                        }
                    }
                }
            }
        }

        if(this->lat_sub->has_msg()){
            std::cout << "got a lat measurement" << std::endl;
            this->Lat_measurement = *(this->lat_sub->take());
            std::cout << std::setprecision(15) << this->Lat_measurement.data << std::endl;
        }

		mtx.unlock();
		std::cout << "step duration: " << this->check_duration_timer.elapsed() << std::endl;
		return;
	}
};

int main(int argc, char** argv){
    // Eigen(linear algebra) example
	Eigen::MatrixXd T(4,4); // zero initialized matrix
	T = Eigen::MatrixXd::Zero(4,4);
	T(0,0) = 2;
	T(2,2) = T(1,1) = T(0,0);
	T(3,3) = 1;
	std::cout << "T: \n" << T << std::endl;

	// Eigen::MatrixXd x(4,1); // zero initialized vector
	// std::cout << x << std::endl;
	// x(0,0) = 1;
	// x(1,0) = 2;
	// x(2,0) = 0;
	// x(3,0) = 1;
	// std::cout << x << std::endl;

	// std::cout << T * x << std::endl;

	Eigen::MatrixXd x(2,3);
	x = Eigen::MatrixXd::Zero(2,3);
	x(0,0) = 1;
	x(1,0) = 2;
	x(0,1) = 3.5;
	x(1,1) = 4.3;
	x(0,2) = 8.8;
	x(1,2) = 9.9;
	std::cout << "x: \n" << x << std::endl;

	std::cout << "result: \n" << frame_transformation(T,x) << std::endl;

    rclcpp::init(argc, argv);
	rclcpp::NodeOptions options{};
	auto node = std::make_shared<slam_node>(options);
	rclcpp::spin(node);
	rclcpp::shutdown();

    return 0;
}
