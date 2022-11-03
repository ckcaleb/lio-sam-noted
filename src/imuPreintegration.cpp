/************************************************* 
GitHub: https://github.com/smilefacehh/LIO-SAM-DetailedNote
Author: lutao2014@163.com
Date: 2021-02-21 
--------------------------------------------------
TransformFusion类
功能简介：
    主要功能是订阅回环后激光里程计信息（来自MapOptimization）和IMU里程计，根据前一时刻激光里程计，和该时刻到当前时刻的IMU里程计变换增量，计算当前时刻IMU里程计；rviz展示IMU里程计轨迹（局部）。

订阅：
    1、订阅激光里程计，来自MapOptimization；
    2、订阅imu里程计，来自ImuPreintegration。


发布：
    1、发布IMU里程计，用于rviz展示；
    2、发布IMU里程计轨迹，仅展示最近一帧激光里程计时刻到当前时刻之间的轨迹。
--------------------------------------------------
IMUPreintegration类
功能简介：
    1、用原始未经回环的激光里程计，两帧激光里程计之间的IMU预计分量构建因子图，优化当前帧的状态（包括位姿、速度、偏置）;
    2、以优化后的状态为基础，施加IMU预计分量，得到每一时刻的IMU里程计。

订阅：
    1、订阅IMU原始数据，以因子图优化后的激光里程计为基础，施加两帧之间的IMU预计分量，预测每一时刻（IMU频率）的IMU里程计；
    2、订阅只图优化，未回环的激光里程计（来自MapOptimization），用两帧之间的IMU预计分量构建因子图，优化当前帧位姿（这个位姿仅用于更新每时刻的IMU里程计，以及下一次因子图优化）。     

发布：
    1、发布在未回环的激光里程计基础上的imu里程计；
**************************************************/ 

#include "utility.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

class TransformFusion : public ParamServer
{
public:
    std::mutex mtx;

    ros::Subscriber subImuOdometry;// 通过imu积分估计的雷达里程计信息订阅器
    ros::Subscriber subLaserOdometry;// 最终优化后的里程计信息订阅器

    ros::Publisher pubImuOdometry;// imu里程计信息发布器
    ros::Publisher pubImuPath;// imu路径发布器

    Eigen::Affine3f lidarOdomAffine; // lidar里程计对应变换矩阵
    Eigen::Affine3f imuOdomAffineFront; // 最老的IMU里程计信息
    Eigen::Affine3f imuOdomAffineBack;  // 最新的IMU里程计信息


    tf::TransformListener tfListener; // tf树监听者
    tf::StampedTransform lidar2Baselink; // 监听从雷达到基坐标系的坐标变换

    double lidarOdomTime = -1; // 激光雷达里程计帧时间戳
    deque<nav_msgs::Odometry> imuOdomQueue; // IMU里程计信息队列

    //构造函数
    TransformFusion()
    {
        // 如果lidar帧和baselink帧不是同一个坐标系
        // 通常baselink指车体系
        if(lidarFrame != baselinkFrame)
        {
            try
            {
                // 查询一下lidar和baselink之间的tf变换
                //Time(0)代表最新的时间
                tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0)); 
                // 查询到的赋值给lidar2Baselink
                tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), lidar2Baselink);
            }
            catch (tf::TransformException ex)
            {
                ROS_ERROR("%s",ex.what());
            }
        }
        // 订阅地图优化节点的全局位姿和预积分节点的增量位姿
        // 订阅激光里程计，来自mapOptimization
        //两次调用不太一样预积分模块调用的是增量性质的lidar 里程计，这个模块调用的是回环性质的里程计
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry", 5, &TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
         // 订阅imu里程计，来自IMUPreintegration
        subImuOdometry   = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental",   2000, &TransformFusion::imuOdometryHandler,   this, ros::TransportHints().tcpNoDelay());
        // 发布imu里程计，用于rviz展示
        pubImuOdometry   = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
        // 发布回环后imu里程计轨迹，用于rviz展示
        pubImuPath       = nh.advertise<nav_msgs::Path>    ("lio_sam/imu/path", 1);
    }
    /**
     * 里程计对应变换矩阵 
     * 把里程计转成eigen 形式
    */
    Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
    {
        double x, y, z, roll, pitch, yaw;
        x = odom.pose.pose.position.x;
        y = odom.pose.pose.position.y;
        z = odom.pose.pose.position.z;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        return pcl::getTransformation(x, y, z, roll, pitch, yaw);
    }

    // 将全局位姿保存下来
    void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        // 激光里程计对应旋转变换矩阵
       /**
     * 里程计对应变换矩阵,位姿
     * 把里程计转成eigen 形式
    */
        lidarOdomAffine = odom2affine(*odomMsg);
        // 激光里程计时间戳
        lidarOdomTime = odomMsg->header.stamp.toSec();
    }

        /**
     * 订阅imu里程计，来自IMUPreintegration
     * 1、以最近一帧激光里程计位姿为基础，计算该时刻与当前时刻间imu里程计增量位姿变换，相乘得到当前时刻imu里程计位姿
     * 2、发布当前时刻里程计位姿，用于rviz展示；发布imu里程计路径，注：只是最近一帧激光里程计时刻与当前时刻之间的一段
    */

    void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
         //// 1. 创建一个tf发布对象   发布tf，map与odom系设为同一个系
        static tf::TransformBroadcaster tfMap2Odom;// odom坐标系一般以上电时刻为原点
        // 2. 创建一个tf对象
        static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
        // 发送静态tf，odom系和map系将重合
        tfMap2Odom. (tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame));

        std::lock_guard<std::mutex> lock(mtx);
        // imu得到的里程记结果送入这个队列中
        imuOdomQueue.push_back(*odomMsg);

        // get latest odometry (at current IMU stamp)
        // 如果没有收到lidar位姿就return
        if (lidarOdomTime == -1)
            return;
        // 弹出时间戳小于最新lidar位姿时刻之前的imu里程记数据
        while (!imuOdomQueue.empty())
        {
            if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
                imuOdomQueue.pop_front();
            else
                break;
        }

        // 计算最新队列里imu里程记的增量
        Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front());
        Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());
        Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
        // 增量补偿到lidar的位姿上去，就得到了最新的预测的位姿
        Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;
        float x, y, z, roll, pitch, yaw;
        // 分解成平移+欧拉角的形式
        pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);
        
        // publish latest odometry
        // 发送全局一致位姿的最新位姿
        nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubImuOdometry.publish(laserOdometry);

        // publish tf
        // 发布tf，当前时刻odom与baselink系变换关系
        static tf::TransformBroadcaster tfOdom2BaseLink;
        tf::Transform tCur;
        tf::poseMsgToTF(laserOdometry.pose.pose, tCur);
        if(lidarFrame != baselinkFrame)
            tCur = tCur * lidar2Baselink;

        // 更新odom到baselink的tf
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
        tfOdom2BaseLink.sendTransform(odom_2_baselink);

        // publish IMU path
        // 发布imu里程计路径，注：只是最近一帧激光里程计时刻与当前时刻之间的一段
        static nav_msgs::Path imuPath;
        static double last_path_time = -1;
        double imuTime = imuOdomQueue.back().header.stamp.toSec();
        // 控制一下更新频率，不超过10hz
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            // 将最新的位姿送入轨迹中
            imuPath.poses.push_back(pose_stamped);
            // 把lidar时间戳之前的轨迹全部擦除
            while(!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0)
                imuPath.poses.erase(imuPath.poses.begin());
            // 发布轨迹，这个轨迹实际上是可视化imu预积分节点输出的预测值 根据是否有节点订阅所需话题，决定是否发布对应话题,getNumSubscribers判断订阅者是否连接
            if (pubImuPath.getNumSubscribers() != 0)
            {
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = odometryFrame;
                pubImuPath.publish(imuPath);
            }
        }
    }
};


class IMUPreintegration : public ParamServer
{
    public:

    std::mutex mtx; // 互斥锁
    //变量声明
    //订阅发布
    ros::Subscriber subImu;
    ros::Subscriber subOdometry;
    ros::Publisher pubImuOdometry;
    // 系统初始化标志位（判断是否是开机或重置后第一次进入雷达订阅者回调函数）
    bool systemInitialized = false;

    // 噪声协方差
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    //初始先验噪声速度置信度 
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    //
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;// 非退化先验位姿噪声
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;// 退化先验位姿噪声
    //
    gtsam::Vector noiseModelBetweenBias;// 帧间偏置变化噪声

    // imu预积分器
    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;// 优化预积分器（雷达帧数据优化处理使用）
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;// 信息处理预积分器（IMU帧数据处理使用）

    // 
    //imu数据队列，执行预积分和位姿的优化
    std::deque<sensor_msgs::Imu> imuQueOpt;
    //imu数据队列更新最新imu状态，保留原始Imu 数据
    std::deque<sensor_msgs::Imu> imuQueImu;


    gtsam::Pose3 prevPose_;// 上一帧估计imu的位姿信息
    gtsam::Vector3 prevVel_;// 上一帧估计imu的速度信息
    gtsam::NavState prevState_;// 上一帧估计的IMU状态信息（包括速度和位姿）初始帧为零
    gtsam::imuBias::ConstantBias prevBias_;// 上一帧估计的IMU零偏信息

    gtsam::NavState prevStateOdom;// 上一帧估计的IMU里程计状态信息（包括速度和位姿）
    gtsam::imuBias::ConstantBias prevBiasOdom;// 优化后上一帧odom 估计的IMU里程计零偏信息

    bool doneFirstOpt = false;//是否经过首次优化标志位
    double lastImuT_imu = -1;// 上一帧IMU数据的时间（-1代表为从odom 第一次优化成功后的   首帧IMU数据  ）在odom 优化中，如果优化失败就会被赋值为=-1，代表重新开始，

    // 里程计初始时间之前一帧IMU数据的时间（-1代表为首帧IMU数据时间大于初始时刻odom 时间，
    //没有上一帧以供求差）在第二帧odom 数据中会用来求差计算预积分,最后记录的是小于当前帧的最近的imu队列的imu时间值
    double lastImuT_opt = -1;

    gtsam::ISAM2 optimizer;// isam2优化器（可以理解为因子图）
    gtsam::NonlinearFactorGraph graphFactors;// 因子集合
    gtsam::Values graphValues;// 变量集合

    const double delta_t = 0; // 判断雷达帧时间所用的偏置，可认为雷达帧实际时间为 获取时间 - delta_t

    int key = 1;// 参与优化的odom 数目（完成优化odom数，达到100时启动重置）

    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));// 从IMU到雷达的平移变换
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));// 从雷达到IMU的平移变换

    IMUPreintegration()
    {

        // 订阅imu原始数据，用下面因子图优化的结果，施加两帧之间的imu预计分量，预测每一时刻（imu频率）的imu里程计
        subImu      = nh.subscribe<sensor_msgs::Imu>  (imuTopic,                   2000, &IMUPreintegration::imuHandler,      this, ros::TransportHints().tcpNoDelay());
        // 订阅后端优化节点发布的 无回环修正的增量式lidar里程计位姿的信息  来自mapOptimization，用两帧之间的imu预计分量构建因子图，优化当前帧位姿（这个位姿仅用于更新每时刻的imu里程计，以及下一次因子图优化）
        subOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry_incremental", 5,    &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        
        // 发布IMU预积分最新位姿信息
        pubImuOdometry = nh.advertise<nav_msgs::Odometry> (odomTopic+"_incremental", 2000);


        // 定义进行imu预积分的imu传感器信息
        //配置IMU 重力方向，根据IMU 数据的方向确定
        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity); // 配置重力加速度

        p->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuAccNoise, 2);  // 加速度计的观测白噪声
        p->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuGyrNoise, 2);        // 陀螺仪的观测白噪声
        //???
        p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2);                      // 通过速度积分位置信息引入的噪声（描述积分不确定性的连续时间协方差）

        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished()); // 初始化imu零偏信息

        // 初始位姿先验噪声置信度设置比较高
        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        // 初始先验噪声速度置信度就设置差一些 第二种表示方式 方差大，置信度小
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // m/s
        // 零偏的先验噪声也设置高一些
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good

        // 激光里程计scan-to-map协方差矩阵
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
       // 激光里程计scan-to-map优化过程中发生退化，则选择一个较大的协方差
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
        //帧间零偏噪声
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
        
        
        // 根据上面的参数，定义两个imu预积分器，一个用于imu信息处理线程，一个用于优化线程
        // imu预积分器，用于预测每一时刻（imu频率）的imu里程计（转到lidar系了，与激光里程计同一个系）
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        // imu预积分器，用于因子图优化
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization        
    }

    //gtsam 复位
    void resetOptimization()
    {

        // 重置优化器
        gtsam::ISAM2Params optParameters;
        // 规定对变量进行重新线性化的阈值
        optParameters.relinearizeThreshold = 0.1;
        // 决定调用几次ISAM2::update才考虑对变量进行重新线性化
        optParameters.relinearizeSkip = 1;
        //优化器
        optimizer = gtsam::ISAM2(optParameters);

        // 重置初始化非线性因子图
        //重置因子graphFactors和
        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;
        //变量属性值 
        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }
// imu因子图优化结果，速度或者偏置过大，认为失败
    void resetParams()
    {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }

   /**
     * 订阅激光里程计，来自mapOptimization
     * 1、每隔100帧激光里程计，重置ISAM2优化器，添加里程计、速度、偏置先验因子，执行优化
     * 2、计算前一帧激光里程计与当前帧激光里程计之间的imu预积分量，用前一帧状态施加预积分量得到当前帧初始状态估计，添加来自mapOptimization的当前帧位姿，进行因子图优化，更新当前帧状态
     * 3、优化之后，执行重传播；优化更新了imu的偏置，用最新的偏置重新计算当前激光里程计时刻之后的imu预积分，这个预积分用于计算每时刻位姿
     * ！！！核心
    */
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        // 获取当前帧激光里程计时间戳
        double currentCorrectionTime = ROS_TIME(odomMsg);

        // make sure we have imu data to integrate
        // 确保imu队列中有数据
        if (imuQueOpt.empty())
            return;
        // 当前帧激光位姿，来自scan-to-map匹配、因子图优化后的位姿
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;

        //判断是否存在退化 （在后端优化里有） 有退化风险，里程计准确性下降
        bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;

        //把里程计雷达位姿转成GTSAM 格式 先旋转再平移
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

        // 0. initialize system
        // 首先初始化系统
        if (systemInitialized == false)
        {
            // 优化问题进行复位
            resetOptimization();

            // pop old IMU message
            // 将这个里程记消息之前的imu信息全部扔掉
            while (!imuQueOpt.empty())
            {
                if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t)
                {
                    lastImuT_opt = ROS_TIME(&imuQueOpt.front());
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }

            // initial pose
            // 将lidar的位姿转移到imu坐标系下 only 平移
            prevPose_ = lidarPose.compose(lidar2Imu);

            // 设置其初始位姿和置信度协方差矩阵，X(0)表示对第一个位姿有先验约束，约束为上一帧算出来的当前帧起始位置imu位姿
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
            // 约束加入到因子中
            graphFactors.add(priorPose);
            // initial velocity
            // 初始化速度，这里就直接赋0了
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            // 将对速度的约束也加入到因子图中
            graphFactors.add(priorVel);

            // initial bias
            // //初始化零偏，初始值设置为0
            prevBias_ = gtsam::imuBias::ConstantBias();
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            // 零偏加入到因子图中
            graphFactors.add(priorBias);

            // 以上把约束加入完毕，下面开始添加状态量
            // add values
            // 将各个状态量赋成初始值
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            // 约束和状态量更新进isam优化器
            optimizer.update(graphFactors, graphValues);

            // 进优化器之后保存约束和状态量的变量就清零
            graphFactors.resize(0);
            graphValues.clear();
            // 预积分的接口，使用初始零偏进行初始化 
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            key = 1;//第一次循环结束
            //系统初始化完成
            systemInitialized = true;
            return;
        }


        // reset graph for speed
        /**
         * @brief 加入约束过多时候，进行isam 复位，复位整个优化问题，只保留最新的位姿，速度以及零偏和状态值（位姿）进入因子图，作为先验
         * 
         */

        if (key == 100)
        {
            // get updated noise before reset
            // 取出第100个优化帧前一帧的位姿、速度、偏置噪声模型
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));

            // reset graph
            // 复位整个优化问题
            resetOptimization();
            // add pose
            // 将最新的位姿，速度，零偏以及对应的协方差矩阵加入到因子图中
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // add velocity
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // add bias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);

            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1;
        }


             /**
              *  1. 计算前一帧与当前帧之间的imu预积分量，两帧odom间imu预积分完成之后，就将其转换成预积分约束
              * 2. 从第2帧开始把从odom 获得的lidar坐标系下位姿转换成imu坐标系下lidar位姿 同时根据是否退化选择不同的置信度，作为这一帧的先验估计
              * 3. 根据上一时刻的状态，结合上一帧的预积分结果，对当前状态进行预测 
              * 4. 预测量作为初始值插入因子图中
              * 5. 获取优化后的当前状态作为当前帧的最佳估计，更新为下次优化所用的上帧优化值
              * 6. 优化之后，对剩余的imu数据计算预积分，传播最新的imu状态
             */
        while (!imuQueOpt.empty())
        {
            // 系统初始化后，收到第二帧的odom,将两帧之间的imu做积分
            // 将imu消息取出来
            sensor_msgs::Imu *thisImu = &imuQueOpt.front();
            double imuTime = ROS_TIME(thisImu);

            // 时间上小于当前lidar位姿的都取出来
            if (imuTime < currentCorrectionTime - delta_t)
            {
                // 计算两个imu量之间的时间差
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);

                // 调用预积分接口将imu数据送进去处理
                // imu预积分数据输入：加速度、角速度、dt
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);

                // 记录当前imu时间
                lastImuT_opt = imuTime;
                imuQueOpt.pop_front();
            }
            else
                break;
        }

        // add imu factor to graph
        // 两帧odom间imu预积分完成之后，就将其转换成预积分约束
        const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);

        // 预积分约束对相邻两帧之间的位姿 速度 零偏形成约束 
        // 参数：前一帧位姿，前一帧速度，当前帧位姿，当前帧速度，前一帧偏置，预计分量    既约束当前帧，也约束前一阵
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        
        // 加入因子图中
        graphFactors.add(imu_factor);


        // add imu bias between factor
        // IMU零偏的约束，两帧间零偏相差不会太大，因此使用常量约束 前一帧bias，当前帧bias，观测值，噪声协方差；deltaTij()是积分段的时间 根据时间游走计算：
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
   
        // 添加位姿因子
        // 从第2帧开始把从odom 获得的lidar坐标系下位姿转换成imu坐标系下lidar位姿
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);

        // 同时根据是否退化选择不同的置信度，作为这一帧的先验估计
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        // 加入因子图中去
        graphFactors.add(pose_factor);

       
        // 根据上一时刻的状态，结合上一帧的预积分结果，对当前状态进行预测 
         // 定义prevState_的时候会调用默认构造函数并赋0初值
         //第一帧估计的零偏为零
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);

        // 预测量作为初始值插入因子图中
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        // optimize
        // 执行优化
        //两次优化
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        //清零
        graphFactors.resize(0);
        graphValues.clear();

        // Overwrite the beginning of the preintegration for the next step.
        //获取优化结果，即本帧优化后的位姿
        gtsam::Values result = optimizer.calculateEstimate();

        // 获取优化后的当前状态作为当前帧的最佳估计，更新为下次优化所用的上帧优化值
        //位姿
        prevPose_  = result.at<gtsam::Pose3>(X(key));
        prevVel_   = result.at<gtsam::Vector3>(V(key));
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        //零偏
        prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));
        // Reset the optimization preintegration object.
        // 当前约束任务已经完成，预积分约束复位，同时需要设置一下零偏作为下一次积分的先决条件
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

        /**
         * imu因子图优化结果，速度或者偏置过大，认为失败
        */
        if (failureDetection(prevVel_, prevBias_))
        {
            // 状态异常就直接复位了
            resetParams();
            return;
        }


        // 2. after optiization, re-propagate imu odometry preintegration
        // 优化之后，对剩余的imu数据计算预积分，传播最新的imu状态
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;
        // 非预积分imu队列中时间戳（-1代表imu 队列中imu 数据大于本帧odom 时间，
        //没有上一帧以供求差）在预测当前帧往后imu 预积分数据时计算dt
        double lastImuQT = -1;
        // 首先把lidar帧之前的imu状态全部弹出去
        while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
        {
            lastImuQT = ROS_TIME(&imuQueImu.front());
            imuQueImu.pop_front();
        }
        // repropogate
        // 如果有晚于lidar状态时刻的imu，对剩余的imu数据计算预积分
        if (!imuQueImu.empty())
        {
           
            // 这个预积分变量复位
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);

            // 然后把剩下的imu状态重新积分
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::Imu *thisImu = &imuQueImu[i];
                double imuTime = ROS_TIME(thisImu);
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) :(imuTime - lastImuQT);

                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        ++key;
        doneFirstOpt = true;
    }

    /**
     * imu因子图优化结果，速度或者偏置过大，认为失败
    */
    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        // 如果当前速度大于30m/s，108km/h就认为是异常状态，
        if (vel.norm() > 30)
        {
            ROS_WARN("Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        // 如果零偏太大，那也不太正常
        if (ba.norm() > 1.0 || bg.norm() > 1.0)
        {
            ROS_WARN("Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

   /**
     * 订阅imu原始数据
     * 1、用上一帧激光里程计时刻对应的状态、偏置，施加从该时刻开始到当前时刻的imu预计分量，得到当前时刻的状态，也就是imu里程计
     * 2、imu里程计位姿转到lidar系，发布里程计
    */
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw)
    {
        std::lock_guard<std::mutex> lock(mtx);

        // 首先把imu的状态做一个简单的转换
        // 将imu信息转换到雷达坐标系（前左上）下表达,其实也就是获得雷达运动的加速度、角速度和姿态信息
        //IMU数据不包括位移，所以这个坐标变换只涉及到旋转。在odometryHandler()里，我们可以看到gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu)，把lidarpose转换到IMU系，但是这里又是把IMU数据转到lidar系，岂不是冲突了吗？我们再看看lidar2Imu：
        //lidar2Imu只涉及到平移的变换，不涉及旋转变换。所以说，整个类，是在lidar坐标系下进行的，原点却是body坐标系的原点。
        sensor_msgs::Imu thisImu = imuConverter(*imu_raw);

        // 注意这里有两个imu的队列，作用不相同，一个用来执行预积分和位姿的优化，一个用来更新最新imu状态
        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        // 要求上一次imu因子图优化执行成功，确保更新了上一帧（激光里程计帧）的状态、偏置，预积分重新计算了
        // 如果没有发生过优化就return 
        //只有当里程计第一帧优化成功才会有
        if (doneFirstOpt == false)
            return;

        double imuTime = ROS_TIME(&thisImu);
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
        lastImuT_imu = imuTime;

        // integrate this single imu message
        // 每来一个imu值就加入预积分状态中
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

        // predict odometry
        // 根据这个值预测最新的状态
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        nav_msgs::Odometry odometry;
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odometryFrame;
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        // 将这个状态转到lidar坐标系下去发送出去
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

         .pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        
        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdo123112311m.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry.publish(odometry);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "roboat_loam");
    
    IMUPreintegration ImuP;

    TransformFusion TF;

    ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");
    
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();
    
    return 0;
}
