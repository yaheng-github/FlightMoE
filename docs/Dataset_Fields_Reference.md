# FlightMoE 数据集字段参考文档

> 本文档对比 RflyMAD 官方数据处理工具提取字段 与 FlightMoE 项目所需字段，用于数据预处理阶段对齐。

---

## 一、FlightMoE 所需字段（来自 `数据集格式.xlsx`）

共 **41 维有效字段**（不含 Index / Timestamp / trueTime 三列索引时间）。

| 序号 | 字段名（处理后 CSV 列名）                      | 物理含义                | 异常类型 | 图像模态 | 备注                                |
| ---- | ---------------------------------------------- | ----------------------- | -------- | -------- | ----------------------------------- |
| 1    | `_actuator_controls_0_0_control[0]`          | 滚转控制指令 [-1,1]     | 突变     | ✓       | PX4 绕 X 轴旋转控制量               |
| 2    | `_actuator_controls_0_0_control[1]`          | 俯仰控制指令 [-1,1]     | —       | —       | PX4 绕 Y 轴旋转控制量               |
| 3    | `_actuator_controls_0_0_control[2]`          | 偏航控制指令 [-1,1]     | —       | —       | PX4 绕 Z 轴旋转控制量               |
| 4    | `_actuator_controls_0_0_control[3]`          | 油门/推力控制指令 [0,1] | —       | —       | 总推力控制量                        |
| 5    | `_actuator_outputs_0_output[0]`              | 电机 1 PWM              | 突变     | —       | 决定电机转速和升力                  |
| 6    | `_actuator_outputs_0_output[1]`              | 电机 2 PWM              | —       | —       | —                                  |
| 7    | `_actuator_outputs_0_output[2]`              | 电机 3 PWM              | —       | —       | —                                  |
| 8    | `_actuator_outputs_0_output[3]`              | 电机 4 PWM              | —       | —       | —                                  |
| 9    | `_sensor_combined_0_gyro_rad[0]`             | X 轴角速度 (rad/s)      | 突变     | ✓       | 陀螺仪                              |
| 10   | `_sensor_combined_0_gyro_rad[1]`             | Y 轴角速度 (rad/s)      | —       | —       | 陀螺仪                              |
| 11   | `_sensor_combined_0_gyro_rad[2]`             | Z 轴角速度 (rad/s)      | —       | —       | 陀螺仪                              |
| 12   | `_sensor_combined_0_accelerometer_m_s2[0]`   | X 轴线加速度 (m/s²)    | 突变     | ✓       | 加速度计                            |
| 13   | `_sensor_combined_0_accelerometer_m_s2[1]`   | Y 轴线加速度 (m/s²)    | —       | —       | 加速度计                            |
| 14   | `_sensor_combined_0_accelerometer_m_s2[2]`   | Z 轴线加速度 (m/s²)    | —       | —       | 加速度计                            |
| 15   | `_vehicle_air_data_0_baro_alt_meter`         | 气压高度 (m)            | 缓变     | —       | —                                  |
| 16   | `_vehicle_air_data_0_baro_pressure_pa`       | 气压 (Pa)               | 缓变     | —       | —                                  |
| 17   | `_vehicle_air_data_0_baro_temp_celcius`      | 温度 (°C)              | 缓变     | —       | —                                  |
| 18   | `_vehicle_attitude_0_q[0]`                   | 姿态四元数 w            | 突变     | —       | —                                  |
| 19   | `_vehicle_attitude_0_q[1]`                   | 姿态四元数 x            | —       | —       | —                                  |
| 20   | `_vehicle_attitude_0_q[2]`                   | 姿态四元数 y            | —       | —       | —                                  |
| 21   | `_vehicle_attitude_0_q[3]`                   | 姿态四元数 z            | —       | —       | —                                  |
| 22   | `_vehicle_local_position_0_x`                | 本地位置 X (m)          | 缓变     | —       | —                                  |
| 23   | `_vehicle_local_position_0_y`                | 本地位置 Y (m)          | —       | —       | —                                  |
| 24   | `_vehicle_local_position_0_z`                | 本地位置 Z (m)          | —       | —       | —                                  |
| 25   | `_vehicle_local_position_0_vx`               | 本地速度 X (m/s)        | 缓变     | —       | —                                  |
| 26   | `_vehicle_local_position_0_vy`               | 本地速度 Y (m/s)        | —       | —       | —                                  |
| 27   | `_vehicle_local_position_0_vz`               | 本地速度 Z (m/s)        | —       | —       | —                                  |
| 28   | `_vehicle_magnetometer_0_magnetometer_ga[0]` | X 轴地磁场强度 (Gauss)  | 缓变     | ✓       | 磁力计                              |
| 29   | `_vehicle_magnetometer_0_magnetometer_ga[1]` | Y 轴地磁场强度 (Gauss)  | —       | —       | 磁力计                              |
| 30   | `_vehicle_magnetometer_0_magnetometer_ga[2]` | Z 轴地磁场强度 (Gauss)  | —       | —       | 磁力计                              |
| 31   | `_battery_status_0_voltage_v`                | 电池电压 (V)            | 缓变     | —       | —                                  |
| 32   | `_battery_status_0_current_a`                | 电池总电流 (A)          | —       | —       | —                                  |
| 33   | `_battery_status_0_remaining`                | 电池剩余电量 (%)        | —       | —       | —                                  |
| 34   | `_battery_status_0_temperature`              | 电池温度 (°C)          | —       | —       | —                                  |
| 35   | `_vehicle_gps_position_0_lat`                | 纬度                    | 突变     | —       | 注意：原表拼写为 `posotion_0_let` |
| 36   | `_vehicle_gps_position_0_lon`                | 经度                    | —       | —       | —                                  |
| 37   | `_vehicle_gps_position_0_alt`                | 海拔 (m)                | —       | —       | —                                  |
| 38   | `_vehicle_gps_position_0_satellites_used`    | 参与定位卫星数          | —       | —       | 正常 ≥8                            |
| 39   | `_vehicle_gps_position_0_eph`                | 水平定位精度 (m)        | —       | —       | —                                  |
| 40   | `_vehicle_gps_position_0_epv`                | 垂直定位精度 (m)        | —       | —       | —                                  |
| 41   | `_vehicle_gps_position_0_fix_type`           | 定位状态码              | —       | —       | —                                  |
|      |                                                |                         |          |          |                                     |

**附加标注列**（由工具生成，非传感器字段）：

| 字段                               | 含义                 | 来源         |
| ---------------------------------- | -------------------- | ------------ |
| `Index`                          | 数据点序号           | 工具生成     |
| `Timestamp`                      | PX4 飞控时间戳 (μs) | PX4 ULog     |
| `trueTime` / `rosbagTimestamp` | 真实时间 / ROS 时间  | GTD / ROS    |
| `飞行模式`                       | —                   | 需后处理添加 |
| `故障模式`                       | —                   | 需后处理添加 |

---

## 二、数据处理工具包当前配置

工具包通过 6 个 JSON 文件控制字段提取：

| JSON 文件              | 适用数据类型 | 内容              |
| ---------------------- | ------------ | ----------------- |
| `data_SIL_PX4.json`  | SIL 仿真     | PX4 ULog 字段选择 |
| `data_HIL_PX4.json`  | HIL 仿真     | PX4 ULog 字段选择 |
| `data_SIL_GTD.json`  | SIL 仿真     | 地面真值字段选择  |
| `data_HIL_GTD.json`  | HIL 仿真     | 地面真值字段选择  |
| `data_real_PX4.json` | 真实飞行     | PX4 ULog 字段选择 |
| `data_real_ROS.json` | 真实飞行     | ROS 话题字段选择  |

### 2.1 当前默认已启用的字段（标记为 `1`）

以 `data_SIL_PX4.json` 为例，以下字段已启用：

- `_actuator_controls_0_0`: `control[0-3]` ✅
- `_actuator_outputs_0`: `output[0-3]` ✅
- `_sensor_combined_0`: `gyro_rad[0-2]`, `accelerometer_m_s2[0-2]` ✅
- `_vehicle_air_data_0`: `baro_alt_meter`, `baro_temp_celcius`, `baro_pressure_pa` ✅
- `_vehicle_attitude_0`: `q[0-3]` ✅
- `_vehicle_local_position_0`: `x, y, z, vx, vy, vz` ✅
- `_vehicle_magnetometer_0`: `magnetometer_ga[0-2]` ✅
- `_battery_status_0`: `voltage_v`, `current_a`, `remaining`, `temperature` ✅
- `_vehicle_gps_position_0`: `lat, lon, alt, eph, epv, fix_type, satellites_used` ✅
- `_rfly_ctrl_lxl_0`: `id, mode, controls[0-5]` ✅（含故障 ID）

### 2.2 当前默认未启用的字段（标记为 `0`）

以下字段在 JSON 中被关闭，但 FlightMoE 可能需要：

- `_vehicle_gps_position_0`: `time_utc_usec, s_variance_m_s, c_variance_rad, hdop, vdop, vel_m_s, vel_n_m_s, vel_e_m_s, vel_d_m_s, cog_rad, heading, heading_offset, jamming_state, vel_ned_valid` 等 — **对用户表格无用**
- `_sensor_gyro_0`, `_sensor_accel_0`, `_sensor_mag_0` — **原始传感器数据**，与 `_sensor_combined_0` 重复
- `_estimator_*` 系列（EKF 估计器状态、创新值、方差等）— **当前未选**，但对 Consistency Expert 可能有价值

---

## 三、字段对齐检查

### ✅ 完全匹配（工具已默认提取）

| 用户表格字段                                                             | 工具 JSON 中对应字段                                                | 状态 |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------- | ---- |
| `_actuator_controls_0_0_control[0-3]`                                  | `control[0-3]` = 1                                                | ✅   |
| `_actuator_outputs_0_output[0-3]`                                      | `output[0-3]` = 1                                                 | ✅   |
| `_sensor_combined_0_gyro_rad[0-2]`                                     | `gyro_rad[0-2]` = 1                                               | ✅   |
| `_sensor_combined_0_accelerometer_m_s2[0-2]`                           | `accelerometer_m_s2[0-2]` = 1                                     | ✅   |
| `_vehicle_air_data_0_baro_*`                                           | `baro_alt_meter`, `baro_pressure_pa`, `baro_temp_celcius` = 1 | ✅   |
| `_vehicle_attitude_0_q[0-3]`                                           | `q[0-3]` = 1                                                      | ✅   |
| `_vehicle_local_position_0_x/y/z/vx/vy/vz`                             | `x,y,z,vx,vy,vz` = 1                                              | ✅   |
| `_vehicle_magnetometer_0_magnetometer_ga[0-2]`                         | `magnetometer_ga[0-2]` = 1                                        | ✅   |
| `_battery_status_0_voltage_v/current_a/remaining/temperature`          | 对应字段 = 1                                                        | ✅   |
| `_vehicle_gps_position_0_lat/lon/alt/satellites_used/eph/epv/fix_type` | 对应字段 = 1                                                        | ✅   |
| `TrueState_data_motorRPMs[1-4]`                                        | 来自 `data_SIL_GTD.json` / `data_HIL_GTD.json`                  | ✅   |

### ⚠️ 需要注意的差异

| 问题                            | 说明                                                                                                   | 处理建议                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| **拼写差异**              | 用户表 `_sensor_conmbined_0`（多打了 n），`_vehicle_gps_posotion_0_let`（position 拼错，lat 拼错） | 以工具包标准命名为准                                                              |
| **飞行模式 / 故障模式列** | 用户表中有这两列，但工具包不输出                                                                       | 需要后处理脚本从 `TestInfo.csv` 或 `_rfly_ctrl_lxl_0.id` 中提取并添加为独立列 |
| **Real 数据的 motorRPMs** | 用户表备注"仅仿真有"，Real 飞行无地面真值                                                              | Real 数据处理时该 4 列为空，需在设计模型时处理（如补零或掩码）                    |
| **图像模态标记**          | 用户表标记了控制量、陀螺仪、加速度计、磁力计为图像模态来源                                             | 后续需写 STFT 代码将对应时序转为时频图像                                          |
| **缓变/突变分类**         | 用户表给出了每类异常的动态特征分类                                                                     | 对设计路由网络（Router）和专家（Expert）有直接指导意义                            |

---

## 四、时频图像通道映射（与用户表格对应）

用户表格中标记为 **图像模态（✓）** 的字段，将用于生成 STFT 时频图像：

| 图像组名               | R 通道                    | G 通道                    | B 通道                    | 检测目标                     |
| ---------------------- | ------------------------- | ------------------------- | ------------------------- | ---------------------------- |
| **Control**      | `control[0]` (Roll)     | `control[1]` (Pitch)    | `control[2]` (Yaw)      | 飞控算法异常、控制链路干扰   |
| **Gyro**         | `gyro_rad[0]`           | `gyro_rad[1]`           | `gyro_rad[2]`           | 飞行姿态震荡、平衡环异常     |
| **Accel**        | `accelerometer_m_s2[0]` | `accelerometer_m_s2[1]` | `accelerometer_m_s2[2]` | 结构松动、物理撞击、剧烈抖动 |
| **Magnetometer** | `magnetometer_ga[0]`    | `magnetometer_ga[1]`    | `magnetometer_ga[2]`    | 环境地磁场变化               |

> 注：此映射与项目汇报 PPT 中的"时频图像通道具体映射"表一致。

---

## 五、后续操作建议

### 5.1 JSON 配置是否需要修改？

**结论：基本不需要修改。** 当前工具包的默认 JSON 配置已经覆盖了 FlightMoE 所需的全部 41 维字段。

### 5.2 需要补充的工作

1. **添加飞行模式 / 故障模式列**

   - 从每个案例的 `TestInfo.csv` 读取故障类型和参数
   - 或从 `_rfly_ctrl_lxl_0.id` 解析故障 ID
   - 写入处理后 CSV 的最后两列
2. **Real 数据的 motorRPMs 处理**

   - 由于 Real 飞行没有 `TrueState_data`，motorRPMs[1-4] 为空
   - 方案 A：用 0 填充 + 模态掩码（Mask）告知模型此字段无效
   - 方案 B：从 `_actuator_outputs_0_output[0-3]` 间接估算（PWM→转速映射）
3. **STFT 时频图像生成**

   - 工具包只输出时序 CSV，不生成图像
   - 需额外编写脚本：对 control/gyro/accel/mag 四组字段做 STFT → RGB 伪彩色编码
   - 建议作为独立步骤，在 CSV 生成后执行
4. **阶段标注（起飞/悬停/巡航/降落）**

   - 工具包不自动输出飞行阶段标签
   - 可基于 `_vehicle_local_position_0_vx/vy/vz` 的速度阈值启发式划分
   - 或基于 RflyMAD 数据集目录结构（hover/waypoint/velocity 等）直接使用文件夹名作为阶段标签

---

## 六、快速生成数据的命令

```bash
# 激活环境
conda activate uav_anomaly

# 处理 SIL 悬停状态下的全部故障，20Hz
python Rflytool_main.py --sub_dataset 1 --flight_status 1 --fault_type 0 --trans_freq 20

# 处理 Real 悬停状态下的全部故障，20Hz
python Rflytool_main.py --sub_dataset 3 --flight_status 1 --fault_type 0 --trans_freq 20

# 处理 HIL 航点飞行状态下的电机和螺旋桨故障，50Hz
python Rflytool_main.py --sub_dataset 2 --flight_status 2 --fault_type 1 2 --trans_freq 50
```

---

## 七、参考文件位置

| 文件                 | 路径                                                         |
| -------------------- | ------------------------------------------------------------ |
| 数据集字段需求表     | `C:\wangyaheng\科研\无人机多模态\启研计划\数据集格式.xlsx` |
| 数据处理主程序       | `Data_processing_tools/Rflytool_main.py`                   |
| PX4 字段配置（SIL）  | `Data_processing_tools/data_SIL_PX4.json`                  |
| PX4 字段配置（Real） | `Data_processing_tools/data_real_PX4.json`                 |
| 地面真值配置（SIL）  | `Data_processing_tools/data_SIL_GTD.json`                  |
| 时间同步工具         | `Data_processing_tools/timetrans.py`                       |
| 核心处理类           | `Data_processing_tools/fileprocess.py`                     |
| 字段提取器           | `Data_processing_tools/data_extractor.py`                  |
| 项目字段参考         | `docs/Dataset_Fields_Reference.md`（本文档）               |

---

*文档生成时间: 2026-05-05*
*对应工具包版本: Data_processing_tools (RflyLab, Beihang)*
