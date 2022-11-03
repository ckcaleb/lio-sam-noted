/************************************************* 
功能简介:
    对经过运动畸变校正之后的当前帧激光点云，计算每个点的曲率，进而提取角点、平面点（用曲率的大小进行判定）。

订阅：
    1、订阅当前激光帧运动畸变校正后的点云信息，来自ImageProjection。

发布：
    1、发布当前激光帧提取特征之后的点云信息，包括的历史数据有：运动畸变校正，点云数据，初始位姿，姿态角，有效点云数据，角点点云，平面点点云等，发布给MapOptimization；
    2、发布当前激光帧提取的角点点云，用于rviz展示；
    3、发布当前激光帧提取的平面点点云，用于rviz展示。
**************************************************/  
#include "utility.h"
#include "lio_sam/cloud_info.h"

/**
 * 激光点曲率
*/
struct smoothness_t{ 
    float value;// 曲率值
    size_t ind;// 激光点一维索引
};

/**
 * 曲率比较函数，从小到大排序
 * 自定义排序顺序 
*/
struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:
    // 雷达点云信息订阅器
    ros::Subscriber subLaserCloudInfo;
    // 发布当前激光帧提取特征之后的点云信息
    ros::Publisher pubLaserCloudInfo;
    // 角点特征发布器
    ros::Publisher pubCornerPoints;
    // 平面点特征发布器
    ros::Publisher pubSurfacePoints;

    // 当前激光帧运动畸变校正后的有效点云
    pcl::PointCloud<PointType>::Ptr extractedCloud;
    //角点点云
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    //平面点点云
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    // 降采样滤波器(降低角点和平面点密度)
    pcl::VoxelGrid<PointType> downSizeFilter;
        // 当前激光帧点云信息，包括的历史数据有：运动畸变校正，点云数据，初始位姿，姿态角，有效点云数据，角点点云，平面点点云等
    lio_sam::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;

    // 点云顺滑性缓存器(每个元素包含点的曲率和索引)
    std::vector<smoothness_t> cloudSmoothness;
    // 点云中点的曲率
    float *cloudCurvature;
    // 特征提取标记，1表示遮挡、平行，或者已经进行特征提取的点，0表示还未进行特征提取处理
    int *cloudNeighborPicked;
    // 1表示角点，-1表示平面点
    int *cloudLabel;

    FeatureExtraction()
    {
        // 订阅当前激光帧运动畸变校正后的点云信息
        subLaserCloudInfo = nh.subscribe<lio_sam::cloud_info>("lio_sam/deskew/cloud_info", 1, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        // 发布当前激光帧提取特征之后的点云信息
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/feature/cloud_info", 1);
        // 发布当前激光帧的角点点云
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_corner", 1);
        // 发布当前激光帧的面点点云
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_surface", 1);
        // 初始化
        initializationValue();
    }

    void initializationValue()
    {
        // 重置缓存器大小为 雷达线数*水平扫描数
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);
        //设置降采样滤波器
        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        //点云分配内存
        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        //分配内存
        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }

    // 订阅上一个结点的消息
    void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn)
    {
        cloudInfo = *msgIn; // new cloud info
        cloudHeader = msgIn->header; // new cloud header
        // 把提取出来的有效的点转成pcl的格式
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // new cloud for extraction

        // 计算曲率
        calculateSmoothness();

        // 标记遮挡点和平行光束点，避免后面进行错误的特征提取
        markOccludedPoints();

        // 点云角点、平面点特征提取
        // 1、遍历扫描线，每根扫描线扫描一周的点云划分为6段，针对每段提取20个角点、不限数量的平面点，加入角点集合、平面点集合
        // 2、认为非角点的点都是平面点，加入平面点云集合，最后降采样
        extractFeatures();

        // 发布特征点信息
        publishFeatureCloud();
    }

    // 计算曲率
    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++)
        {
            // 计算当前点和周围十个点的距离差
            float diffRange = cloudInfo.pointRange[i-5] + cloudInfo.pointRange[i-4]
                            + cloudInfo.pointRange[i-3] + cloudInfo.pointRange[i-2]
                            + cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i] * 10
                            + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2]
                            + cloudInfo.pointRange[i+3] + cloudInfo.pointRange[i+4]
                            + cloudInfo.pointRange[i+5];            
            //计算方差
            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;
            // 下面两个值赋成初始值
            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting
            // 用来进行曲率排序
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    // 标记属于遮挡、平行两种情况的点，不做特征提取
    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            // 取出相邻两个点距离信息
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i+1];
        // 两个激光点之间的一维索引差值，如果在一条扫描线上，那么值为1；如果两个点之间有一些无效点被剔除了，可能会比1大，但不会特别大
            // 如果恰好前一个点在扫描一周的结束时刻，下一个点是另一条扫描线的起始时刻，那么值会很大
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));
            // 只有比较靠近才有意义
            if (columnDiff < 10){
                // 10 pixel diff in range image
            // 两个点在同一扫描线上，且距离相差大于0.3，认为存在遮挡关系（也就是这两个点不在同一平面上，如果在同一平面上，距离相差不会太大）
            // 远处的点会被遮挡，标记一下该点以及相邻的5个点，后面不再进行特征提取
                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){   // 这里同理
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));
            // 如果两点距离比较大 就很可能是平行的点，也很可能失去观测
            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }
   /**
     * 点云角点、平面点特征提取
     * 1、遍历扫描线，每根扫描线扫描一周的点云划分为6段，针对每段提取20个角点、不限数量的平面点，加入角点集合、平面点集合
     * 2、认为非角点的点都是平面点，加入平面点云集合，最后降采样
    */
    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();
        //每一scan 未进行下采样面点集合
        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        //每一scan采样后面点集合
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();
            // 把每一根scan等分成6份，每份分别提取特征点
            for (int j = 0; j < 6; j++)
            {
                // 根据之前得到的每个scan的起始和结束id来均分
                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;
                // 这种情况就不正常
                if (sp >= ep)
                    continue;

                // 按照曲率从大到小遍历
                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                // 当前得到的角点数目
                int largestPickedNum = 0;

                for (int k = ep; k >= sp; k--)
                {
                    // 找到这个点对应的原先的idx
                    int ind = cloudSmoothness[k].ind;
                   // 当前激光点还未被处理，且曲率大于阈值，则认为是角点
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        largestPickedNum++;
                        // 每段最多找20个角点
                        if (largestPickedNum <= 20){
                            // 标签置1表示是角点
                            cloudLabel[ind] = 1;
                            // 这个点收集进存储角点的点云中
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else {
                            break;
                        }
                        // 将这个点周围的几个点设置成已处理点，避免选取太集中
                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            // 列idx距离太远就算了，空间上也不会太集中
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        // 同理
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                // 开始收集面点
                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    // 同样要求不是遮挡点且曲率小于给定阈值
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {
                        // -1表示是面点
                        cloudLabel[ind] = -1;
                        // 同理 把周围的点都设置为遮挡点
                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                // 平面点和未被处理的点，都认为是平面点，加入平面点云集合
                    if (cloudLabel[k] <= 0){
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }

            surfaceCloudScanDS->clear();
            // 因为面点太多了，所以做一个下采样
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            *surfaceCloud += *surfaceCloudScanDS;
        }
    }
    /**
     * 清理，释放内存
    */
    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointRange.clear();
    }

    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        // 发布角点、面点点云，用于rviz展示
        cloudInfo.cloud_corner  = publishCloud(&pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(&pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
        // publish to mapOptimization
        // 发布当前激光帧点云信息，加入了角点、面点点云数据，发布给mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> Feature Extraction Started.\033[0m"); //https://blog.csdn.net/QLeelq/article/details/124518475?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-124518475-blog-107184684.pc_relevant_multi_platform_whitelistv1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-124518475-blog-107184684.pc_relevant_multi_platform_whitelistv1&utm_relevant_index=1
   
    ros::spin();

    return 0;
}













