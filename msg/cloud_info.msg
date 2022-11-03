# Cloud Info
Header header 

#key 为每条线上第几个有效点是起点
int32[] startRingIndex #每个线可以计算曲率的起点
int32[] endRingIndex #每个线可以计算曲率的终点

#key 对应的是count ,也就是一维的有效点的坐标，
int32[]  pointColInd # point column index in range image每个有效点对应的是第几根水平线 这个数组的key对应第几个有效点
float32[] pointRange # point range 每个有效点离原点的距离   这个数组的key对应第几个有效点

int64 imuAvailable
int64 odomAvailable

# Attitude for LOAM initialization
#提取比点云开始时间戳早的第一个imu数据姿态角RPYimu姿态角RPY，作为当前lidar帧初始姿态角
float32 imuRollInit
float32 imuPitchInit
float32 imuYawInit

# Initial guess from imu pre-integration，
#当前取出离雷达帧最近且时间大于等于雷达帧起始时间的IMU里程计信息，后面用于mapOptmization，存的是odom2imu
float32 initialGuessX 
float32 initialGuessY
float32 initialGuessZ
float32 initialGuessRoll
float32 initialGuessPitch
float32 initialGuessYaw

# Point cloud messages
#把所有有效点都进行一维化处理 跟count 一一对应   都是矫正后的点云
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed 提取出来的有效的点
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature //角点
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature//面点