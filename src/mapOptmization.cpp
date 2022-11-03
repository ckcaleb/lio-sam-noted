/************************************************* 
功能简介:
    1、scan-to-map匹配：提取当前激光帧特征点（角点、平面点），局部关键帧map的特征点，执行scan-to-map迭代优化，更新当前帧位姿；
    2、关键帧因子图优化：关键帧加入因子图，添加激光里程计因子、GPS因子、闭环因子，执行因子图优化，更新所有关键帧位姿；
    3、闭环检测：在历史关键帧中找距离相近，时间相隔较远的帧设为匹配帧，匹配帧周围提取局部关键帧map，同样执行scan-to-map匹配，得到位姿变换，构建闭环因子数据，加入因子图优化。

订阅：
    1、订阅当前激光帧点云信息，来自FeatureExtraction；
    2、订阅GPS里程计；
    3、订阅来自外部闭环检测程序提供的闭环数据，本程序没有提供，这里实际没用上。


发布：
    1、发布历史关键帧里程计；
    2、发布局部关键帧map的特征点云；
    3、发布激光里程计，rviz中表现为坐标轴；
    4、发布激光里程计；
    5、发布激光里程计路径，rviz中表现为载体的运行轨迹；
    6、发布地图保存服务；
    7、发布闭环匹配关键帧局部map；
    8、发布当前关键帧经过闭环优化后的位姿变换之后的特征点云；
    9、发布闭环边，rviz中表现为闭环帧之间的连线；
    10、发布局部map的降采样平面点集合；
    11、发布历史帧（累加的）的角点、平面点降采样集合；
    12、发布当前帧原始点云配准之后的点云。
**************************************************/ 
#include "utility.h"
#include "lio_sam/cloud_info.h"
#include "lio_sam/save_map.h"

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

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/**
 * 6D位姿点云结构定义
*/
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;


class mapOptimization : public ParamServer
{

    public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;// 因子集合
    Values initialEstimate;//变量属性值 
    Values optimizedEstimate;
    ISAM2 *isam; // isam2优化器（可以理解为因子图）

    Values isamCurrentEstimate; // 优化结果 
    Eigen::MatrixXd poseCovariance; //当前位姿的置信度

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal; //发布回环后里程计
    ros::Publisher pubLaserOdometryIncremental; //发布平滑里程计，类似未经回环
    ros::Publisher pubKeyPoses;//发布位置
    ros::Publisher pubPath;//发送全局位置信息

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames; //发布局部地图的点云
    ros::Publisher pubRecentKeyFrame; //当前关键帧的平面点和角点
    ros::Publisher pubCloudRegisteredRaw; //发布原始配准后点云
    ros::Publisher pubLoopConstraintEdge;

    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subLoop;

    ros::ServiceServer srvSaveMap;

    std::deque<nav_msgs::Odometry> gpsQueue;
    lio_sam::cloud_info cloudInfo;

    // 历史所有关键帧的角点集合（降采样）
    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    // 历史所有关键帧的平面点集合（降采样）
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    //  存储关键帧的位置信息的点云
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;    
    // 存储关键帧的6D位姿信息的点云
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;   //到世界坐标系变换 
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    // 当前激光帧角点集合
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; 
    // 当前激光帧平面点集合
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; 
    // 当前激光帧角点集合，降采样后，DS: DownSize
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; 
    // 当前激光帧平面点集合，降采样
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; 

    // 当前帧与局部map匹配上了的角点、平面点，加入同一集合；后面是对应点的参数
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    // 当前帧与局部map匹配上了的角点、参数、标记
    std::vector<PointType> laserCloudOriCornerVec; 
    // 残差和雅克比
    std::vector<PointType> coeffSelCornerVec;
    //标记
    std::vector<bool> laserCloudOriCornerFlag;
    // 当前帧与局部map匹配上了的平面点、参数、标记
    std::vector<PointType> laserCloudOriSurfVec; 
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    // 局部地图的一个容器 int 存放第几个关键帧 第一个pair 存放角点 第二个存放面点 是转移到世界坐标系下的点
    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer; 
    // 局部map的角点集合
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    // 局部map的平面点集合
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    // 局部map的角点集合，降采样
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    // 局部map的平面点集合，降采样
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;   // 角点局部地图的kdtree
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap; // 面点局部地图的kdtree

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses; //历史关键帧集合

    //降采样
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    
    ros::Time timeLaserInfoStamp;//lidar时间戳
    double timeLaserInfoCur;//当前帧时间戳sec
    //初始是回环优化后的lidar最佳位姿,然后是本帧的位姿，从先验到后验，第一帧时候为零
    float transformTobeMapped[6];

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    cv::Mat matP;

    int laserCloudCornerFromMapDSNum = 0;   // 当前局部地图下采样后的角点数目
    int laserCloudSurfFromMapDSNum = 0; // 当前局部地图下采样后的面点数目
    int laserCloudCornerLastDSNum = 0;  // 当前帧下采样后的角点的数目
    int laserCloudSurfLastDSNum = 0;    // 当前帧下采样后的面点数目

    bool aLoopIsClosed = false;//回环优化标志位
    map<int, int> loopIndexContainer; // 形成回环约束对
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    deque<std_msgs::Float64MultiArray> loopInfoVec;
    //里程计位姿,rviz 展示用
    nav_msgs::Path globalPath;

    // 当前帧位姿
    Eigen::Affine3f transPointAssociateToMap;
    // Eigen前一帧回环优化后位姿
    Eigen::Affine3f incrementalOdometryAffineFront;
    // 通过lidar 帧间约束和imu 优化后的位姿
    Eigen::Affine3f incrementalOdometryAffineBack;

    /**
     * 构造函数
     * 1
    */
    mapOptimization()
    {
        // ISM2参数
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1; // 规定对变量进行重新线性化的阈值
        parameters.relinearizeSkip = 1; // 决定调用几次ISAM2::update才考虑对变量进行重新线性化
        isam = new ISAM2(parameters);

        // 发布当前帧位置信息
        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/trajectory", 1);
        // 发布局部关键帧map的特征点云
        pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_global", 1);
        // 发布回环位姿激光里程计，
        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry", 1);
        // 发布非回环光滑激光里程计，
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry_incremental", 1);
        // 发布激光里程计路径，rviz中表现为载体的运行轨迹
        pubPath                     = nh.advertise<nav_msgs::Path>("lio_sam/mapping/path", 1);

        

        // 订阅当前激光帧点云信息，来自featureExtraction
        subCloud = nh.subscribe<lio_sam::cloud_info>("lio_sam/feature/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅GPS里程计
        subGPS   = nh.subscribe<nav_msgs::Odometry> (gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅来自外部闭环检测程序提供的闭环数据，本程序没有提供，这里实际没用上
        subLoop  = nh.subscribe<std_msgs::Float64MultiArray>("lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());

        // 发布一个保存地图功能的服务
        srvSaveMap  = nh.advertiseService("lio_sam/save_map", &mapOptimization::saveMapService, this);

        // 发布闭环匹配关键帧局部map
        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
        // 发布闭环边，rviz中表现为闭环帧之间的连线
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1);

        // 发布局部map的降采样平面点集合
        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_local", 1);
        // 发布当前帧回环优化后的角点、平面点降采样集合
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered", 1);
        // 发布当前帧原始点云配准之后的点云
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);

        // 体素滤波设置珊格大小
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization
        //初始化 内存分配
        allocateMemory();
    }
    // 预先分配内存
    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
    }


/**
     * 订阅当前激光帧点云信息，来自featureExtraction
     * 1、当前帧位姿初始化
     *   1) 如果是第一帧，用原始imu数据的RPY初始化当前帧位姿（旋转部分）
     *   2) 后续帧，用imu里程计计算两帧之间的增量位姿变换，作用于前一帧的激光位姿，得到当前帧激光位姿
     * 2、提取局部角点、平面点云集合，加入局部map
     *   1) 对最近的一帧关键帧，搜索时空维度上相邻的关键帧集合，降采样一下
     *   2) 对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部map中
     * 3、当前激光帧角点、平面点集合降采样
     * 4、scan-to-map优化当前帧位姿
     *   (1) 要求当前帧特征点数量足够多，且匹配的点数够多，才执行优化
     *   (2) 迭代30次（上限）优化
     *      1) 当前激光帧角点寻找局部map匹配点
     *          a.更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
     *          b.计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
     *      2) 当前激光帧平面点寻找局部map匹配点
     *          a.更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
     *          b.计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
     *      3) 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
     *      4) 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
     *   (3)用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
     * 5、设置当前帧为关键帧并执行因子图优化
     *   1) 计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
     *   2) 添加激光里程计因子、GPS因子、闭环因子
     *   3) 执行因子图优化
     *   4) 得到当前帧优化后位姿，位姿协方差
     *   5) 添加cloudKeyPoses3D，cloudKeyPoses6D，更新transformTobeMapped，添加当前关键帧的角点、平面点集合
     * 6、更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
     * 7、发布激光里程计
     * 8、发布里程计、点云、轨迹
    */
    void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn)
    {
        // extract time stamp
        // 提取当前时间戳
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = msgIn->header.stamp.toSec();

        // extract info and feature cloud
        // 提取cloudinfo中的角点和面点
        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerLast);
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

        std::lock_guard<std::mutex> lock(mtx);
        //上次处理时间戳，第一帧一定处理
        static double timeLastProcessing = -1;

       
        // 控制后端频率，两帧处理一帧
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
            timeLastProcessing = timeLaserInfoCur;
            /**
             * 当前帧位姿初始化
             * 1、先提取上一帧的优化后位姿 incrementalOdometryAffineFront
             * 2、如果是第一帧，相当于没有imu预积分结果，那么初始的位姿由imu 数据提供，可以不用yaw,同时保存IMU 原始数据的位姿
             * 3、后续帧，有预积分
             *          第一帧预积分，那么先提取预积分信息作为下一帧求增量用，然后用imu 信息去进行本帧先验
             *          第二帧和往后预积分，那么先提取预积分信息作为下一帧求增量用，计算预积分增量，赋值给上一帧的最佳位姿，得到先验
             * 4、后续帧、没有预积分
             *          用imu计算两帧之间的增量位姿变换，作用于前一帧的激光位姿，得到当前帧激光位姿，当前帧先验估计
            */
            updateInitialGuess();

               /**
                 * 判断有没有关键帧，没有的话就返回
                 * 只提取关键帧信息，构建局部地图
                * 提取局部角点、平面点云集合，加入局部map
                    * 1.提取关键帧位姿，kdtree查找，放到周围关键帧集合中，下采样
                    * 2.搜索时间维度上相邻的关键帧集合，降采样一下
                    * 3.索引为其是poses3D中第几个关键帧，赋值intensity 好像没啥用
                    * 4.构建局部地图 同时存入容器
                */
            extractSurroundingKeyFrames();

            /**
             * 当前激光帧角点、平面点集合降采样
            */
            downsampleCurrentScan();
                /**
                 * 第一帧直接跳过
                 * scan-to-map优化当前帧位姿
                 * 1、要求当前帧特征点数量足够多，且匹配的点数够多，才执行优化
                 * 2、迭代30次（上限）优化
                 *   1) 当前激光帧角点寻找局部map匹配点
                 *      a.更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
                 *      b.计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
                 *   2) 当前激光帧平面点寻找局部map匹配点
                 *      a.更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
                 *      b.计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
                 *   3) 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
                 *   4) 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
                 * 3、用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，
                 * 值约束、约束z坐标
                 * 4. 得到的位姿用来进行imu预积分的平滑里程计的转换
                */
            scan2MapOptimization();
                /**
                 * 设置当前帧为关键帧并执行因子图优化
                 * 1、计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧，第一帧设置为关键帧
                 * 2、添加激光里程计因子、GPS因子、闭环因子，第一帧和第二帧不一样
                 * 3、执行因子图优化
                 * 4、得到当前帧优化后位姿，位姿协方差
                 * 第一帧在这个位置才加入关键帧中
                 * 5、添加关键帧cloudKeyPoses3D// 平移信息取出来保存进cloudKeyPoses3D这个结构中，其中索引作为intensity值，含义为第几个关键帧
                 * 6、cloudKeyPoses6D // 6D姿态同样保留下来
                 * 7、更新transformTobeMapped，添加当前关键帧的角点、平面点集合
                 * 6、更新里程计位姿
                */
            saveKeyFramesAndFactor();

            /**
             * 第一帧和后续帧不一样，第一帧直接返回
             * 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
             * 含有回环信息
            */
            correctPoses();
            /**
             * 发布标准优化的激光里程计 位姿
             * 发布tf变换 发送激光里程计坐标系
             * 发布平滑的里程计位姿
            */
            publishOdometry();
            /**
             * 发布里程计、点云、轨迹
             * 1、发布历史关键帧位姿集合
             * 2、发布局部map的降采样平面点集合
             * 3、发布历史帧（累加的）的角点、平面点降采样集合
             * 4、发布原始点云的配准后点云
             * 5、发布里程计轨迹
            */
            publishFrames();
        }
    }
    // 收集gps信息
    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        gpsQueue.push_back(*gpsMsg);
    }

    /**
     * 激光坐标系下的激光点，通过激光帧位姿，变换到世界坐标系下
    */
    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

   /**
     * 对点云cloudIn进行变换transformIn，返回结果点云
    */
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        // 使用openmp进行并行加速，for循环中可以使用 
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            // 每个点都施加RX+t这样一个过程
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }
    // 转成gtsam的数据结构
    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }


    /**
     * Eigen格式的位姿变换
    */
    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    











    /**
     * 保存全局关键帧特征点集合
    */
    bool saveMapService(lio_sam::save_mapRequest& req, lio_sam::save_mapResponse& res)
    {
      string saveMapDirectory;

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files ..." << endl;
      // 如果是空，说明是程序结束后的自动保存，否则是中途调用ros的service发布的保存指令
      if(req.destination.empty()) saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
      else saveMapDirectory = std::getenv("HOME") + req.destination;
      cout << "Save destination: " << saveMapDirectory << endl;
      // create directory and remove old files;
      // 删掉之前有可能保存过的地图
      int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
      unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
      // save key frame transformations
      // 首先保存关键帧轨迹
      pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
      // extract global point cloud map
      pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
      // 遍历所有关键帧，将点云全部转移到世界坐标系下去
      for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
          *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
          *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
          // 类似进度条的功能
          cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
      }
        // 如果没有指定分辨率，就是直接保存
      if(req.resolution != 0)
      {
        cout << "\n\nSave resolution: " << req.resolution << endl;

        // down-sample and save corner cloud
        // 使用指定分辨率降采样，分别保存角点地图和面点地图
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        downSizeFilterCorner.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterCorner.filter(*globalCornerCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
        // down-sample and save surf cloud
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
      }
      else
      {
        // save corner cloud
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
        // save surf cloud
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
      }

      // save global point cloud map
      *globalMapCloud += *globalCornerCloud;
      *globalMapCloud += *globalSurfCloud;
        // 保存全局地图
      int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
      res.success = ret == 0;

      downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
      downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files completed\n" << endl;

      return true;
    }





    /**
     * 展示线程
     * 1、发布局部关键帧map的特征点云
     * 2、保存全局关键帧特征点集合
    */

    // 全局可视化线程
    void visualizeGlobalMapThread()
    {
        // 更新频率设置为0.2hz
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            /**
             * 发布局部关键帧map的特征点云
             * kdtree 选取附近关键帧
             * 根据距离排除
             * 转到世界坐标系下，发布
            */
            publishGlobalMap();
        }
        // 当ros被杀死之后，执行保存地图功能
        if (savePCD == true)
            return;

        lio_sam::save_mapRequest  req;
        lio_sam::save_mapResponse res;

        if(!saveMapService(req, res)){
            cout << "Fail to save map" << endl;
        }
    }

    /**
     * 发布局部关键帧map的特征点云
     * kdtree 选取附近关键帧
     * 根据距离排除
     * 转到世界坐标系下，发布
    */
    void publishGlobalMap()
    {
        // 如果没有订阅者就不发布，节省系统负载
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;
        // 没有关键帧自然也没有全局地图了
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>()); //最新关键帧附近的关键帧
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kdtree查找最近一帧关键帧相邻的关键帧集合
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        // 把所有关键帧送入kdtree
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        // 寻找具体最新关键帧一定范围内的其他关键帧
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();
        // 把这些找到的关键帧的位姿保存起来
        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        // 最一个简单的下采样
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

         // 提取局部相邻关键帧对应的特征点云
        for(auto& pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            // 找到这些下采样后的关键帧的索引，并保存下来
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }

       // 提取局部相邻关键帧对应的特征点云
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            //距离过远
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            // 将每一帧的点云通过位姿转到世界坐标系下
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        // 转换后的点云也进行一个下采样
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        // 最终发布出去
        publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }











    // 回环检测线程
    void loopClosureThread()
    {
        // 如果不需要进行回环检测，那么就退出这个线程
        if (loopClosureEnableFlag == false)
            return;
        // 设置回环检测的频率
        ros::Rate rate(loopClosureFrequency);
        while (ros::ok())
        {
            // 执行完一次就必须sleep一段时间，否则该线程的cpu占用会非常高
            rate.sleep();
            // 执行回环检测
                /**
         * 闭环scan-to-map，icp优化位姿
         * 1、在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
         * 2、提取当前关键帧特征点集合，降采样；提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
         * 3、执行scan-to-map优化，调用icp方法，得到优化后位姿，
         * 4.把修正后的当前点云发布供可视化使用
         * 5.闭环优化得到的当前关键帧与回环变换后当前关键帧之间的位姿变换
         * 6.将两帧索引，两帧相对位姿和噪声作为回环约束送入队列
         * 7.保存已存在的约束对
         * 构造闭环因子需要的数据，在因子图优化中一并加入更新位姿
         * 注：闭环的时候没有立即更新当前帧的位姿，而是添加闭环因子，让图优化去更新位姿
        */
            performLoopClosure();
            visualizeLoopClosure();
        }
    }
    // 接收外部告知的回环信息
    void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        // 回环信息必须是配对的，因此大小检查一下是不是2
        if (loopMsg->data.size() != 2)
            return;
        // 把当前回环信息送进队列
        loopInfoVec.push_back(*loopMsg);
        // 如果队列里回环信息太多没有处理，就把老的回环信息扔掉
        while (loopInfoVec.size() > 5)
            loopInfoVec.pop_front();
    }

    /**
     * 闭环scan-to-map，icp优化位姿
     * 1、在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
     * 2、提取当前关键帧特征点集合，降采样；提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
     * 3、执行scan-to-map优化，调用icp方法，得到优化后位姿，
     * 4.把修正后的当前点云发布供可视化使用
     * 5.闭环优化得到的当前关键帧与闭环关键帧之间的位姿变换
     * 6.将两帧索引，两帧相对位姿和噪声作为回环约束送入队列
     * 7.保存已存在的约束对
     * 构造闭环因子需要的数据，在因子图优化中一并加入更新位姿
     * 注：闭环的时候没有立即更新当前帧的位姿，而是添加闭环因子，让图优化去更新位姿
    */
    void performLoopClosure()
    {
        //1.寻找回环对
        // 如果没有关键帧，就没法进行回环检测了
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        mtx.lock();
        // 把存储关键帧的位姿的点云copy出来，避免线程冲突
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        int loopKeyCur;//优化帧
        int loopKeyPre;//回环帧

            /**
         * @description: 外部提供的回环，还没用过
         * @param {int} *latestID
         * @param {int} *closestID
         * @return {*}
         */    
        if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
                /**
         *  在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
         * 1.检查一下较晚帧是否和别的形成了回环，如果有就算了
         * 2.kdtree 查找近邻帧  必须满足时间上超过一定阈值，才认为是一个有效的回环
         * 3.添加距离判断
         */ 
            if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
                return;
        
        // 2. 检出回环之后开始计算两帧位姿变换
        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>()); //回环帧
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());//匹配帧
        {
            // 稍晚的帧就把自己取了出来
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            // 稍早一点的就把自己和周围一些点云取出来，也就是构成一个帧到局部地图的一个匹配问题
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
            // 如果点云数目太少就算了
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                // 把局部地图发布出去供rviz可视化使用
                publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // ICP Settings
        // 3. 使用简单的icp来进行帧到局部地图的配准pcl icp

        static pcl::IterativeClosestPoint<PointType, PointType> icp;  
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);//设置对应点对之间的最大距离（此值对配准结果影响较大）。
        icp.setMaximumIterations(100);//最大迭代次数
        icp.setTransformationEpsilon(1e-6);//前一个变换矩阵和当前变换矩阵的差异小于阈值时，就认为已经收敛了，是一条收敛条件
        icp.setEuclideanFitnessEpsilon(1e-6); // 还有一条收敛条件是均方误差和小于阈值， 停止迭代。
        icp.setRANSACIterations(0);

        // Align clouds
        // 设置两个点云
        icp.setInputSource(cureKeyframeCloud); //回环帧
        icp.setInputTarget(prevKeyframeCloud);//匹配帧
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        // 执行点云配准
        icp.align(*unused_result);//执行配准  存储变换后的源点云
        // 检查icp是否收敛且得分是否满足要求
        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)//如果两个点云匹配正确的话 则 icp.hasConverged()为1 
            return;

        // publish corrected cloud
        // 4.把修正后的当前点云发布供可视化使用
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());//变换矩阵Tpre2cure
            publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // 5.闭环优化得到的当前关键帧与闭环关键帧之间的位姿变换
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        // 获得当前帧点云变换前后位姿变换 相对于世界坐标系
        correctionLidarFrame = icp.getFinalTransformation();


        // 闭环优化前当前帧位姿
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        //  闭环优化后回环帧位姿
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame %https://blog.csdn.net/qsunj/article/details/123426801


        // 将回环帧的位姿转成平移+欧拉角
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        // from是修正后的稍微帧的点云位姿
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        // to是修正前的稍早帧的点云位姿
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        // 使用icp的得分作为他们的约束的噪声项
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        mtx.lock();
        // 6.将两帧索引，两帧相对位姿和噪声作为回环约束送入队列
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        // add loop constriant
        // 7.保存已存在的约束对
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    /**
     *  在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
     * 1.检查一下较晚帧是否和别的形成了回环，如果有就算了
     * 2.kdtree 查找近邻帧  必须满足时间上超过一定阈值，才认为是一个有效的回环
     * 3.添加距离判断
     * 
     */   
    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        // 检测最新帧是否和其他帧形成回环，所以后面一帧的id就是最后一个关键帧
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        // 检查一下较晚帧是否和别的形成了回环，如果有就算了
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合
        std::vector<int> pointSearchIndLoop; //关键帧下标
        std::vector<float> pointSearchSqDisLoop;//关键帧距离
        // 把只包含关键帧位移信息的点云填充kdtree
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        // 根据最后一个关键帧的平移信息，寻找离他一定距离内的其他关键帧
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        // 遍历找到的候选关键帧
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            // 必须满足时间上超过一定阈值，才认为是一个有效的回环
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                // 此时就退出了
                loopKeyPre = id;
                break;
            }
        }
        // 如果没有找到回环或者回环找到自己身上去了，就认为是此次回环寻找失败
        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }


    /**
     * @description: 外部提供的回环，还没用过
     * @param {int} *latestID
     * @param {int} *closestID
     * @return {*}
     */    
    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // this function is not used yet, please ignore it
        // 作者表示这个功能还没有使用过，可以忽略它
        int loopKeyCur = -1;//回环帧
        int loopKeyPre = -1;  //previous key

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        // 外部的先验回环消息队列
        if (loopInfoVec.empty())
            return false;
        // 取出回环信息，这里是把时间戳作为回环信息
        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();
        // 如果两个回环帧之间的时间差小于30s就算了
        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;
        // 如果现存的关键帧小于2帧那也没有意义
        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return false;

        // latest key
        loopKeyCur = cloudSize - 1;
        // 遍历所有的关键帧，找到离后面一个时间戳更近的关键帧的id
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        // 同理，找到距离前一个时间戳更近的关键帧的id
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }
        // 两个是同一个关键帧，就没什么意思了吧
        if (loopKeyCur == loopKeyPre)
            return false;
        // 检查一下较晚的这个帧有没有被其他时候检测了回环约束，如果已经和别的帧形成了回环就算了
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;
        // 两个帧的索引输出
        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }
    /**
     * 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合，降采样
     * 回环帧是只取自己
     * 匹配帧是取出周围临近几帧数据
    */
    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        // searchNum是搜索范围
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            // 找到这个idx
            int keyNear = key + i;
            // 如果超出范围就算了
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            // 否则把对应角点和面点的点云转到世界坐标系下去
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }
        // 如果没有有效的点云就算了
        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        // 把点云下采样
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    /**
     * rviz展示闭环边
    */
    void visualizeLoopClosure()
    {
        // 如果没有回环约束就算了
        if (loopIndexContainer.empty())
            return;
        
        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        // 回环约束的两帧作为node添加
        // 先定义一下node的信息
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        // 两帧之间约束作为edge添加
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;
        // 遍历所有回环约束
        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            // 添加node和edge
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        // 最后发布出去供可视化使用
        pubLoopConstraintEdge.publish(markerArray);
    }

    

    /**
     * 当前帧位姿初始化
     * 1、先提取上一帧的优化后位姿 incrementalOdometryAffineFront
     * 2、如果是第一帧，相当于没有imu预积分结果，那么初始的位姿由imu 数据提供，可以不用yaw,同时保存IMU 原始数据的位姿
     * 3、后续帧，有预积分
     *          第一帧预积分，那么先提取预积分信息作为下一帧求增量用，然后用imu 信息去进行本帧先验
     *          第二帧和往后预积分，那么先提取预积分信息作为下一帧求增量用，计算预积分增量，赋值给上一帧的最佳位姿，得到先验
     * 4、后续帧、没有预积分
     *          用imu里程计计算两帧之间的增量位姿变换，作用于前一帧的激光位姿，得到当前帧激光位姿，当前帧先验估计
    */
    void updateInitialGuess()
    {

        /**
         * Eigen格式的位姿变换
        */
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);
         // 前一帧的初始化姿态角（来自原始imu数据），用于估计第一帧的位姿（旋转部分）
        static Eigen::Affine3f lastImuTransformation;
        // initialization
        // 没有关键帧，也就是系统刚刚初始化完成
        if (cloudKeyPoses3D->points.empty())
        {
            // 初始的位姿就由磁力计提供
            transformTobeMapped[0] = cloudInfo.imuRollInit;
            transformTobeMapped[1] = cloudInfo.imuPitchInit;
            transformTobeMapped[2] = cloudInfo.imuYawInit;
            // 无论vio还是lio 系统的不可观都是4自由度，平移+yaw角，这里虽然有磁力计将yaw对齐，但是也可以考虑不使用yaw？？？
           //如果不使用IMU
            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;
            // 保存磁力计得到的位姿，平移置0
            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }

        static bool lastImuPreTransAvailable = false;
        //上一帧IMU 里程计信息
        static Eigen::Affine3f lastImuPreTransformation;

        // 如果有预积分节点提供的里程记
        if (cloudInfo.odomAvailable == true)
        {
            // 将提供的初值转成eigen的数据结构保存下来
            //这一帧imu里程计信息
            Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX,    cloudInfo.initialGuessY,     cloudInfo.initialGuessZ, 
                                                               cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
            // 这个标志位表示是否收到过第一帧预积分里程记信息
            if (lastImuPreTransAvailable == false)
            {
                
                // 将当前里程记结果记录下来，作为上一针预积分数据以供下一帧使用
                lastImuPreTransformation = transBack;
                // 收到第一个里程记数据以后这个标志位就是true
                lastImuPreTransAvailable = true;
            } else {
                // 计算上一个里程记的结果和当前里程记结果之间的delta pose
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                //上一帧优化后的最佳位姿
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                // 将这个增量加到上一帧最佳位姿上去，就是当前帧位姿的一个先验估计位姿
                Eigen::Affine3f transFinal = transTobe * transIncre;//当前帧位姿的一个先验估计位姿
                // 将eigen变量转成当前帧优化后结果的欧拉角和平移的形式
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                //  将当前里程记结果记录下来，作为上一针数据下一帧使用
                lastImuPreTransformation = transBack;
                // 虽然有里程记信息，仍然需要把imu磁力计得到的旋转记录下来
                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
                return;
            }
        }

        // use imu incremental estimation for pose guess (only rotation)
        // 如果没有里程记信息，就是用imu的旋转信息来更新，因为单纯使用imu无法得到靠谱的平移信息，因此，平移直接置0
        if (cloudInfo.imuAvailable == true)
        {
            // 初值计算方式和上面相同，只不过注意平移置0
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            // 计算上一个imu的结果和当前imu结果之间的delta pose
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;
            //上一帧优化后的最佳位姿
            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }
    }

    /**
     * not-used
    */
    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

    /**
     * 提取局部角点、平面点云集合，加入局部map
     * 1.提取关键帧位姿，kdtree查找，放到周围关键帧集合中，下采样
     * 2.搜索时间维度上相邻的关键帧集合，降采样一下
     * 3.索引为其是poses3D中第几个关键帧，赋值intensity 好像没啥用
     * 4.构建局部地图 同时存入容器
    */
    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>()); //kdtree和时间采样出来的关键帧集合


        std::vector<int> pointSearchInd;    // 保存kdtree提取出来的元素的索引
        std::vector<float> pointSearchSqDis;    // 保存距离查询位置的距离的数组

        // extract all the nearby key poses and downsample them
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
         // 对最近的一帧关键帧，在半径区域内搜索空间区域上相邻的关键帧集合  50m
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        // 根据查询的结果，把这些点的位置存进一个点云结构中
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        // 避免关键帧过多，因此对关键帧位置信息做一个下采样 1m范围
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);


        // also extract some latest key frames in case the robot rotates in one position
        int numPoses = cloudKeyPoses3D->size();
        // 刚刚是提取了一些空间上比较近的关键帧，然后再提取一些时间上比较近的关键帧
        for (int i = numPoses-1; i >= 0; --i)
        {
            // 最近十秒的关键帧也保存下来
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        // 确认每个下采样后的点的索引，就使用一个最近邻搜索，其索引为是第几帧关键帧，赋值给这个点的intensity数据位
        for(auto& pt : surroundingKeyPosesDS->points)
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

    /**
     * 构建局部地图
     * 1、对关键帧集合中的每一帧pose，提取对应的角点、平面点，转换到世界坐标系下，加入局部map中，
     * 同时加地图容器中
    */
        extractCloud(surroundingKeyPosesDS);
    }

    /**
     * 构建局部地图
     * 1、对关键帧集合中的每一帧pose，提取对应的角点、平面点，转换到世界坐标系下，加入局部map中，
     * 同时加地图容器中
    */
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
        // 分别存储角点和面点相关的局部地图
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            // 简单校验一下关键帧距离不能太远，这个实际上不太会触发
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;
            // 取出提出出来的关键帧的索引
            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            // 如果这个关键帧对应的点云信息已经存储在一个地图容器里
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) 
            {
                // transformed cloud available
                // 直接从容器中取出来加到局部地图中
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // transformed cloud not available
                // 如果这个点云没有实现存取，那就通过该帧对应的位姿，把该帧点云从当前帧的位姿转到世界坐标系下
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
                // 点云转换之后加到局部地图中
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
                // 把转换后的面点和角点存进这个容器中，方便后续直接加入点云地图，避免点云转换的操作，节约时间
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
            
        }

        // Downsample the surrounding corner key frames (or map)
        // 将提取的关键帧的点云转到世界坐标系下后，避免点云过度密集，因此对面点和角点的局部地图做一个下采样的过程
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // clear map cache if too large
        // 如果这个局部地图容器过大，就clear一下，避免占用内存过大
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

    /**
     * 判断有没有关键帧，没有的话就返回
    * 提取局部角点、平面点云集合，加入局部map
        * 1.提取关键帧位姿，kdtree查找，放到周围关键帧集合中，下采样
        * 2.搜索时间维度上相邻的关键帧集合，降采样一下
        * 3.索引为其是poses3D中第几个关键帧，赋值intensity 好像没啥用
        * 4.构建局部地图 同时存入容器
    */
    void extractSurroundingKeyFrames()
    {
        // 如果当前没有关键帧，就return了
        if (cloudKeyPoses3D->points.empty() == true)
            return; 
        
        // if (loopClosureEnableFlag == true)
        // {
        //     extractForLoopClosure();    
        // } else {
        //     extractNearby();
        // }

        extractNearby();
    }

    /**
     * 当前激光帧角点、平面点集合降采样
    */
    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        // 当前帧的角点和面点分别进行下采样，也就是为了减少计算量
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }
    // 更新当前帧的位姿转成eigen的对象
    void updatePointAssociateToMap()
    {
        // 将欧拉角转成eigen的对象
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }
    /**
     * 当前激光帧角点寻找局部map匹配点
     * 1、更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
     * 2、计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
    */
    void   cornerOptimization()
    {
    /**
     * 更新当前帧位姿 -transPointAssociateToMap
    */
        updatePointAssociateToMap();
        // 使用openmp并行加速
        #pragma omp parallel for num_threads(numberOfCores)  

        // 遍历当前帧的角点
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff; //通过本帧先验位姿把pointSel转到世界坐标系 //残差和雅克比 
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            /**
             * 激光坐标系下的激光点，通过激光帧位姿，变换到世界坐标系下
            */
            pointAssociateToMap(&pointOri, &pointSel);
            // 在角点地图里寻找距离当前点比较近的5个点
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0)); //协方差矩阵
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0)); //特征值
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
            // 计算找到的点中距离当前点最远的点，如果距离太大那说明这个约束不可信，就跳过  
            if (pointSearchSqDis[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;
                // 计算协方差矩阵
                // 首先计算均值
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                // 构建协方差矩阵
                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;
                // 特征值分解
                cv::eigen(matA1, matD1, mtV1);
                // 这是线特征性，要求最大特a征值大于3倍的次大特征值
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
                    // 当前帧角点坐标（map系下）
                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    // 特征向量对应的就是直线的方向向量
                    // 通过点的均值往两边拓展，构成一个线的两个端点
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);
                    // area_012，也就是三个点组成的三角形面积*2，叉积的模|axb|=a*b*sin(theta) 
                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));
                    // line_12，底边边长
                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
                    // 两次叉积，得到点到直线的垂线段单位向量，x分量，下面同理  先求线到点的雅克比 //https://zhuanlan.zhihu.com/p/548579394  https://blog.csdn.net/qq_32761549/article/details/126641834
                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;
                     // 三角形的高，也就是点到直线距离
                    float ld2 = a012 / l12;
                     // 距离越大，s越小，是个距离惩罚因子（权重）
                    float s = 1 - 0.9 * fabs(ld2); //绝对值

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;
                    // 如果ld2小于10cm，就认为是一个有效的约束
                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    /**
     * 当前激光帧平面点寻找局部map匹配点
     * 1、更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
     * 2、计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
    */
    void surfOptimization()
    {
        // 更新当前帧的位姿转成eigen的对象
        updatePointAssociateToMap();
        // 遍历当前帧平面点集合
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            // 平面点（坐标还是lidar系） 同样找5个面点
            pointOri = laserCloudSurfLastDS->points[i];
            // 根据当前帧位姿，变换到世界坐标系（map系）下
            pointAssociateToMap(&pointOri, &pointSel); 
            // 在局部平面点map中查找当前平面点相邻的5个平面点
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;
            // 平面方程Ax + By + Cz + 1 = 0
            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();
            // 同样最大距离不能超过1m
            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }
                // 求解Ax = b这个超定方程
                matX0 = matA0.colPivHouseholderQr().solve(matB0);
                // 求出来x的就是这个平面的法向量
                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                // 归一化，将法向量模长统一为1
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    // 每个点代入平面方程，计算点到平面的距离，如果距离大于0.2m认为这个平面曲率偏大，就是无效的平面
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }
                // 如果通过了平面的校验
                if (planeValid) {
                    // 计算当前点到平面的距离
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;
                    // 分母不是很明白，为了更多的面点用起来？
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        // 当前激光帧角点，加入匹配集合中
                        laserCloudOriSurfVec[i] = pointOri;
                         // 角点的参数
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }
    /**
     * 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
     * 标志位清零
    */
    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
            // 只有标志位为true的时候才是有效约束
            if (laserCloudOriCornerFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        // 标志位清零
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    /**
     * scan-to-map优化
     * 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，求解
     * 首次迭代式，判断是否退化，当海森矩阵的特征值小于100时候，判断为退化，将对应特征向量置零，然后用置零后的海森矩阵特征向量矩阵和原来的海森矩阵特征向量矩阵相乘，
     * 获得新的特征向量，将求解的deltaX和新的特征向量相乘，获得新的deltaX
     * 迭代优化当前位姿，存transformTobeMapped 判断是否收敛
     * 公式推导：todo
    */
    bool LMOptimization(int iterCount)
    {
        // 原始的loam代码是将lidar坐标系转到相机坐标系，这里把原先loam中的代码拷贝了过来，但是为了坐标系的统一，就先转到相机系优化，然后结果转回lidar系
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        // 将lidar系转到相机系
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        //特征点匹配太少，约束小于50
        if (laserCloudSelNum < 50) {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        for (int i = 0; i < laserCloudSelNum; i++) {
            // 首先将当前点以及点到线（面）的单位向量转到相机系
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            // 相机系下的旋转顺序是Y - X - Z对应lidar系下Z -Y -X
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            // lidar -> camera
            // 这里就是把camera转到lidar了
            matA.at<float>(i, 0) = arz; //对旋转求导
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z; //对位移求导
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            // 点到直线距离、平面距离，作为残差
            matB.at<float>(i, 0) = -coeff.intensity;
        }
        // 构造JTJ以及-JTe矩阵
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        // 求解增量
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);
     // 首次迭代，检查近似Hessian矩阵（J^T·J）是否退化，或者称为奇异，行列式值=0 todo
        if (iterCount == 0) {
            // 检查一下是否有退化的情况
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0)); //特征值
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));
            // 对JTJ进行特征值分解
            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                // 特征值从小到大遍历，如果小于阈值就认为退化
                if (matE.at<float>(0, i) < eignThre[i]) {
                    // 对应的特征向量全部置0
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }
        // 如果发生退化，就对增量进行修改，退化方向不更新
        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }
        
        // 增量更新
        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);
        // 计算更新的旋转和平移大小
        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));
        // 旋转和平移增量足够小，认为优化问题收敛了
        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        // 否则继续优化
        return false; // keep optimizing
    }
    /**
     * 第一帧直接跳过
     * scan-to-map优化当前帧位姿
     * 1、要求当前帧特征点数量足够多，且匹配的点数够多，才执行优化
     * 2、迭代30次（上限）优化
     *   1) 当前激光帧角点寻找局部map匹配点
     *      a.更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
     *      b.计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
     *   2) 当前激光帧平面点寻找局部map匹配点
     *      a.更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
     *      b.计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
     *   3) 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
     *   4) 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
     * 3、用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
     * 4. 得到的位姿用来进行imu预积分的平滑里程计的转换
    */
    void  scan2MapOptimization()
    {
        // 如果没有关键帧，那也没办法做当前帧到局部地图的匹配
        if (cloudKeyPoses3D->points.empty())
            return;
        // 判断当前帧的角点数和面点数是否足够
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            // 分别把角点面点局部地图构建kdtree
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);
            // 迭代求解 手写优化器
            //迭代完收到新的当前位姿，transformTobeMapper
            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                // 每次迭代清空特征点集合
                laserCloudOri->clear();
                coeffSel->clear();
                // 当前激光帧角点寻找局部map匹配点
                // 1、更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
                // 2、计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
                cornerOptimization();

                // 当前激光帧平面点寻找局部map匹配点
                // 1、更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
                // 2、计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数                
                surfOptimization();
                /**
                 * 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
                 * 标志位清零
                */
                combineOptimizationCoeffs();

                    /**
                     * scan-to-map优化
                     * 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，求解
                     * 首次迭代式，判断是否退化，当海森矩阵的特征值小于100时候，判断为退化，将对应特征向量置零，然后用置零后的海森矩阵特征向量矩阵和原来的海森矩阵特征向量矩阵相乘，
                     * 获得新的特征向量，将求解的deltaX和新的特征向量相乘，获得新的deltaX
                     * 迭代优化当前位姿，存transformTobeMapped 判断是否收敛，收敛 即 true
                     * scanmapping 更新后的位姿留下来仍到最后做帧间约束用
                     * 公式推导：todo
                    */                
                     if (LMOptimization(iterCount) == true)
                    break;              
            }
            /**
             * 用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，球插和协方差矩阵，更新当前帧位姿的roll、pitch，约束z坐标，roll、pitch
            *传递到transformTobeMapped
            *
            */
            transformUpdate( );
        } else {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

        /**
         * 用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，球插和协方差矩阵，更新当前帧位姿的roll、pitch，
         * 用值约束 约束z坐标，roll、pitch
        */
    void transformUpdate()
    {
        // 可以获取九轴imu的世界系下的姿态
        if (cloudInfo.imuAvailable == true)
        {
            // 因为roll 和 pitch原则上全程可观，因此这里把lidar推算出来的姿态和磁力计结果做一个加权平均
            // 首先判断车翻了没有，车翻了好像做slam也没有什么意义了，当然手持设备可以pitch很大，这里主要避免插值产生的奇异
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                //权重
                double imuWeight = imuRPYWeight;
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                // lidar匹配获得的roll角转成四元数
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                // imu获得的roll角
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);

                // 使用四元数球面插值
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                // 插值结果作为roll的最终结果
                transformTobeMapped[0] = rollMid;

                // 下面pitch角同理
                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                // 球插
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }
        // 用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，球插和协方差矩阵，更新当前帧位姿的roll、pitch，, z坐标；因为是小车，roll、pitch是相对稳定的，不会有很大变动，一定程度上可以信赖imu的数据，z是进行高度约束
        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);
        // 最终的结果也可以转成eigen的结构
        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    /**
     * 值约束
    */
    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    /**
     * 判断是否为关键帧
     * 第一帧默认关键帧
     * 计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
    */
    bool saveFrame()
    {
        // 如果没有关键帧，那直接认为是关键帧
        if (cloudKeyPoses3D->points.empty())
            return true;
        // 取出上一个关键帧的位姿
        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        // 当前帧的位姿转成eigen形式
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        // 计算两个位姿之间的delta pose
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        // 转成平移+欧拉角的形式
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);
        // 任何一个旋转大于给定阈值或者平移大于给定阈值就认为是关键帧
        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    
    /**
     * 1.第一帧设为关键帧
     *      1.1 添加先验约束，对第0点添加约束
     *       1.2添加第0点信息
     *      此时置信度低，尤其是不可观的平移和yaw角
     *  2.如果不是第一帧，添加帧间约束信息，这时帧间约束置信度就设置高一些
     *      
     */    
    void addOdomFactor()
    {
        // 第一帧直接往下走
        if (cloudKeyPoses3D->points.empty())
        {
            // 置信度就设置差一点，尤其是不可观的平移和yaw角
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            // 增加先验约束，对第0个节点增加约束
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            // 加入节点信息
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }else{
            // 如果不是第一帧，就增加帧间约束
            // 这时帧间约束置信度就设置高一些
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            // 转成gtsam的格式
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            // 这是一个帧间约束，分别输入两个节点的id，帧间约束大小以及置信度
            // 参数：前一帧id，当前帧id，前一帧与当前帧的位姿变换（作为观测值），噪声协方差
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            // 加入节点信息
            // 变量节点设置初始值
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }



    void addGPSFactor()
    {
        // 如果没有gps信息就算了
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        // 如果有gps但是没有关键帧信息也算了
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            // 第一个关键帧和最后一个关键帧相差很近也算了，要么刚起步，要么会触发回环
            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // pose covariance small, no need to correct
        // gtsam反馈的当前x，y的置信度，如果置信度比较高也不需要gps来打扰
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            // 把距离当前帧比较早的帧都抛弃
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            // 比较晚就索性再等等lidar计算
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                // 说明这个gps时间上距离当前帧已经比较近了,那就把这个数据取出来
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                // 如果gps的置信度不高，也没有必要使用了
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;
                // 取出gps的位置
                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                // 通常gps的z没有x y准，因此这里可以不使用z值
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                // gps的x或者y太小说明还没有初始化好
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                // 加入gps观测不宜太频繁，相邻不超过5m
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                // gps的置信度，标准差设置成最小1m，也就是不会特别信任gps信号
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                // 调用gtsam中集成的gps的约束
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);
                // 加入gps之后等同于回环，需要触发较多的isam update
                aLoopIsClosed = true;
                break;
            }
        }
    }

    /**
     * 添加闭环因子
    */
    void addLoopFactor()
    {
        // 有一个专门的回环检测线程会检测回环，检测到就会给这个队列塞入回环结果
        if (loopIndexQueue.empty())
            return;
        // 把队列里所有的回环约束都添加进来
        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            // 这是一个帧间约束
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            // 回环的置信度就是icp的得分
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            // 加入约束
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }
        // 清空回环相关队列
        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        // 标志位置true
        aLoopIsClosed = true;
    }

    /**
     * 设置当前帧为关键帧并执行因子图优化
     * 1、计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧，第一帧设置为关键帧
     * 2、添加激光里程计因子、GPS因子、闭环因子，第一帧和第二帧不一样
     * 3、执行因子图优化
     * 4、得到当前帧优化后位姿，位姿协方差
     * 5、添加关键帧cloudKeyPoses3D// 平移信息取出来保存进cloudKeyPoses3D这个结构中，其中索引作为intensity值，含义为第几个关键帧
     * 6、cloudKeyPoses6D // 6D姿态同样保留下来
     * 7、更新transformTobeMapped，添加当前关键帧的角点、平面点集合
     * 6、更新里程计位姿
    */
    void saveKeyFramesAndFactor()
    {
            /**
         * 判断是否为关键帧
         * 第一帧默认设为关键帧
         * 计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之认为可以当关键帧
        */
        if (saveFrame() == false)
            return;
        /**
         * 1.第一帧直接添加先验约束
         *      1.1 添加先验约束，对第0点添加约束
         *       1.2添加节点信息
         *  2.后续关键帧添加添加帧间信息
         */    
        addOdomFactor();
        // gps的因子
        // gps factor
        addGPSFactor();
        // 回环的因子
        // loop factor
        addLoopFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // 所有因子加完了，就调用isam接口更新图模型
        // update iSAM
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();
        // 如果加入了gps的约束或者回环约束，isam需要进行更多次的优化
        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }
        // 将约束和节点信息清空，他们已经被加入到isam中去了，因此这里清空不会影响整个优化
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        // 下面保存关键帧信息
        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate; // 优化后的最新关键帧位姿

        isamCurrentEstimate = isam->calculateEstimate();
        // 取出优化后的最新关键帧位姿
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");
        //添加关键帧
        // 平移信息取出来保存进cloudKeyPoses3D这个结构中，其中索引作为intensity值，含义为第几个关键帧
        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);
        // 6D姿态同样保留下来
        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        // 保存当前位姿的置信度
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // save updated transform
        // 将优化后的位姿更新到transformTobeMapped数组中，作为当前最佳估计值
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        // 当前帧的点云的角点和面点分别拷贝一下
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        // 关键帧的点云保存下来
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // save path for visualization
        // 根据当前最新位姿更新rviz可视化
        updatePath(thisPose6D);
    }

    /**
     * 跟回环和GPS相关
     *有回环因子再更新 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
    */
    void correctPoses()
    {
        // 没有关键帧，自然也没有什么意义
        if (cloudKeyPoses3D->points.empty())
            return;
        // 只有回环以及gps信息这些会触发全局调整信息才会触发
        if (aLoopIsClosed == true)
        {
            // clear map cache
            // 很多位姿会变化，因子之前的容器内转到世界坐标系下的很多点云就需要调整，因此这里索性清空
            laserCloudMapContainer.clear();
            // clear path
            // 清空path
            globalPath.poses.clear();
            // update key poses
            // 然后更新所有的位姿
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                // 更新所有关键帧的位姿
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();
                    /**
                     * 更新里程计轨迹
                    */
                updatePath(cloudKeyPoses6D->points[i]);
            }
            // 标志位置位
            aLoopIsClosed = false;
        }
    }

    /**
     * 更新里程计轨迹
    */
    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }
    /**
     * 发布标准优化的激光里程计 位姿
     * 发布tf变换 发送激光里程计坐标系
     * 发布平滑的里程计位姿
    */
    void publishOdometry()
    {
        // Publish odometry for ROS (global)
        // 发送当前帧的位姿
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping"; //优化后当前帧坐标系
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        pubLaserOdometryGlobal.publish(laserOdometryROS);
        
        // Publish TF
        // 发送lidar在odom坐标系下的tf 和odom_mapping 同一个坐标系
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
        br.sendTransform(trans_odom_to_lidar);

        // Publish odometry for ROS (incremental)
        // 发送增量位姿变换
        // 这里主要用于给imu预积分模块使用，需要里程计是平滑的？？？

        //第一帧优化后的也是平滑的
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental; // 平滑里程计

        static Eigen::Affine3f increOdomAffine; // 当前优化后位姿的仿射里程计 一直都是回环前的 由于初始帧就是回环前的，增量一直用的是scanmapping 的增量

        // 该标志位处理一次后始终为true
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            // 记录当前位姿
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } else {
            // 上一帧的最佳位姿（回环后）和当前帧最佳位姿（scan matching之后，而不是根据回环或者gps调整之后的位姿）之间的位姿增量
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            // 位姿增量叠加到上一帧非回环位姿上
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            // 分解成欧拉角+平移向量
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            // 如果有imu信号，同样对roll和pitch做插值
            if (cloudInfo.imuAvailable == true)
            {
                if (std::abs(cloudInfo.imuPitchInit) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf::Quaternion imuQuaternion;
                    tf::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // slerp roll
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // slerp pitch
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
            // 协方差这一位作为是否退化的标志位
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        pubLaserOdometryIncremental.publish(laserOdomIncremental);
    }

    /**
     * 发布关键帧、点云、轨迹
     * 1、发布历史关键帧位姿集合
     * 2、发布局部map的降采样平面点集合
     * 3、发布历史帧（累加的）的角点、平面点降采样集合
     * 4、发布原始点云的配准后点云
     * 5、发布里程计轨迹
    */
    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses
        // 1.发送关键帧位置信息
        publishCloud(&pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // Publish surrounding key frames
        // 2.发送周围局部地图的平面点云信息
        publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);

         // 3.发布当前帧的角点、平面点降采样集合
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            //当前关键帧的平面点和角点
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            // 把当前点云转换到世界坐标系下去
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            // //当前关键帧的点云
            publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // 4. 发布当前帧原始点云配准之后的点云
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            // 发送原始点云
            publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish path
        // 5. 发送path
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    mapOptimization MO;

    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");

    // std::thread 构造函数，将MO作为参 数传入构造的线程中使用
    // 进行闭环检测与闭环的功能
    //两个线程 一个回环 一个可视化
    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    // 该线程中进行的工作是publishGlobalMap(),将数据发布到ros中，可视化
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::spin();

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}
