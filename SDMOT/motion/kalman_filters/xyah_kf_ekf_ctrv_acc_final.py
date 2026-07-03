import numpy as np
from numpy.linalg import inv, norm
from boxmot.motion.kalman_filters.base_kalman_filter import BaseKalmanFilter
import logging

class AdaptiveNoiseEKF_CTRvA_SportMOT(BaseKalmanFilter):
    """
    基于EKF的自适应噪声调整滤波器，采用CTRA非线性运动模型，
    状态向量为 [x, y, a, h, v, yaw, yaw_rate, acc]。

    创新点：
      1. 将过程噪声与测量噪声的自适应调整凝练为一个模块，并引入检测器目标置信度，
         低置信度时放大测量噪声、提高鲁棒性；高置信度时降低噪声、使更新更准确。
      2. 动态雅可比矩阵更新策略：在预测阶段动态融合解析雅可比矩阵与数值计算雅可比，
         当系统非线性程度较高时，更多依赖数值计算的结果，从而提升模型精度。
    """
    def __init__(self, process_noise_factor=26.0, measurement_noise_factor=2.0, use_dynamic_jacobian=True):
        # 状态维度为8
        super().__init__(ndim=8)
        self.logger = logging.getLogger("AdaptiveNoiseEKF_CTRvA_SportMOT")
        self.process_noise_factor = process_noise_factor
        self.measurement_noise_factor = measurement_noise_factor
        self.use_dynamic_jacobian = use_dynamic_jacobian
        # 解析噪声权重设置
        self._std_weight_position = 1.0 / 8
        self._std_weight_velocity = 1.0 / 120
        # 数值雅可比计算时的扰动比例
        self.dynamic_jacobian_epsilon = 1e-5

    def initiate(self, measurement: np.ndarray):
        """
        根据初始测量值 [x, y, a, h] 初始化状态和协方差，
        其余状态初始为0。
        """
        mean = np.hstack([measurement, np.zeros(4)])  # 初始 v, yaw, yaw_rate, acc 均为 0
        cov_diag = self._get_initial_covariance_std(measurement)
        covariance = np.diag(cov_diag)
        self.logger.debug("Initiate: mean=%s, covariance=%s", mean, covariance)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray, dt: float = 1.0):
        """
        EKF预测步骤：
         - 根据CTRA非线性模型 f() 得到预测均值；
         - 计算雅可比矩阵 F：如果启用动态雅可比，则融合解析和数值雅可比；
         - 根据自适应噪声模块构造过程噪声 Q；
         - 更新协方差。
        """
        mean_pred = self._f(mean, dt)
        if self.use_dynamic_jacobian:
            F = self._F_dynamic(mean, dt)
        else:
            F = self._F(mean, dt)
        Q_diag = self._get_adaptive_noise_std(mean, dt)
        Q = np.diag(Q_diag)
        covariance_pred = F @ covariance @ F.T + Q
        self.logger.debug("Predict: mean_pred=%s, covariance_pred=%s", mean_pred, covariance_pred)
        return mean_pred, covariance_pred

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, dt: float = 1.0, confidence: float = 1.0):
        """
        EKF更新步骤：
         - 测量模型为 z = H * state，其中 H = [I_4, 0]；
         - 引入检测器输出的目标置信度信息，根据置信度调整测量噪声；
         - 根据标准卡尔曼更新公式计算更新增益 K 并更新状态。
         
        参数:
         - measurement: [x, y, a, h]
         - confidence: 检测器置信度（0～1之间），1表示最高置信度。
        """
        H = np.hstack([np.eye(4), np.zeros((4, 4))])
        z_pred = H @ mean
        innovation = measurement - z_pred
        R = np.diag(self._get_adaptive_measurement_noise_std(mean[:4], measurement[3], confidence))
        S = H @ covariance @ H.T + R
        K = covariance @ H.T @ inv(S)
        mean_updated = mean + K @ innovation
        covariance_updated = (np.eye(len(mean)) - K @ H) @ covariance
        self.logger.debug("Update: measurement=%s, z_pred=%s, K=%s, confidence=%.3f", measurement, z_pred, K, confidence)
        return mean_updated, covariance_updated

    def multi_predict(self, multi_mean: np.ndarray, multi_covariance: np.ndarray, dt: float = 1.0):
        new_mean = []
        new_cov = []
        for mean, cov in zip(multi_mean, multi_covariance):
            m, c = self.predict(mean, cov, dt)
            new_mean.append(m)
            new_cov.append(c)
        return np.array(new_mean), np.array(new_cov)

    def _f(self, state: np.ndarray, dt: float):
        """
        CTRA运动模型预测：
         - s = v * dt + 0.5 * acc * dt^2
         - phi = yaw + yaw_rate * dt / 2
         - 根据模型计算 x, y 的更新；其他状态按一阶近似更新。
        """
        x, y, a, h, v, yaw, yaw_rate, acc = state
        s = v * dt + 0.5 * acc * dt**2
        phi = yaw + yaw_rate * dt / 2.0
        x_new = x + s * np.cos(phi)
        y_new = y + s * np.sin(phi)
        a_new = a
        h_new = h
        v_new = v + acc * dt
        yaw_new = yaw + yaw_rate * dt
        yaw_rate_new = yaw_rate
        acc_new = acc
        return np.array([x_new, y_new, a_new, h_new, v_new, yaw_new, yaw_rate_new, acc_new])

    def _F(self, state: np.ndarray, dt: float):
        """
        解析计算状态转移函数 f(state) 关于 state 的雅可比矩阵 F = df/dstate。
        """
        x, y, a, h, v, yaw, yaw_rate, acc = state
        s = v * dt + 0.5 * acc * dt**2
        phi = yaw + yaw_rate * dt / 2.0
        F = np.eye(8)
        # 对于 x, y 分量
        F[0, 4] = dt * np.cos(phi)
        F[0, 7] = 0.5 * dt**2 * np.cos(phi)
        F[0, 5] = -s * np.sin(phi)
        F[0, 6] = -s * np.sin(phi) * (dt / 2.0)
        
        F[1, 4] = dt * np.sin(phi)
        F[1, 7] = 0.5 * dt**2 * np.sin(phi)
        F[1, 5] = s * np.cos(phi)
        F[1, 6] = s * np.cos(phi) * (dt / 2.0)
        # a, h 分量不变
        
        # 对 v 更新： v_new = v + acc * dt
        F[4, 4] = 1
        F[4, 7] = dt
        
        # 对 yaw 更新： yaw_new = yaw + yaw_rate * dt
        F[5, 5] = 1
        F[5, 6] = dt
        # yaw_rate 和 acc 保持不变
        return F

    def _F_numeric(self, state: np.ndarray, dt: float):
        """
        利用有限差分方法数值计算雅可比矩阵。
        """
        n = len(state)
        F_num = np.zeros((n, n))
        # 根据状态大小自适应计算扰动步长 epsilon
        epsilon = self.dynamic_jacobian_epsilon * np.maximum(np.abs(state), 1.0)
        for i in range(n):
            state_plus = np.copy(state)
            state_minus = np.copy(state)
            state_plus[i] += epsilon[i]
            state_minus[i] -= epsilon[i]
            f_plus = self._f(state_plus, dt)
            f_minus = self._f(state_minus, dt)
            F_num[:, i] = (f_plus - f_minus) / (2 * epsilon[i])
        return F_num

    def _F_dynamic(self, state: np.ndarray, dt: float):
        """
        动态雅可比矩阵更新策略：
          - 计算解析雅可比矩阵 F_analytical 与数值雅可比矩阵 F_numeric，
          - 计算两者的相对差异作为非线性度量，
          - 根据非线性度量融合两种雅可比：
                F_dynamic = (1 - w) * F_analytical + w * F_numeric,
            其中 w = clip( ||F_numeric - F_analytical|| / ||F_analytical||, 0, 1 )。
        """
        F_analytical = self._F(state, dt)
        F_numeric = self._F_numeric(state, dt)
        # 计算相对差异的均值作为非线性权重
        diff_norm = norm(F_numeric - F_analytical, ord='fro')
        base_norm = norm(F_analytical, ord='fro')
        if base_norm < 1e-8:
            weight = 0.0
        else:
            weight = np.clip(diff_norm / base_norm, 0.0, 1.0)
        F_dynamic = (1 - weight) * F_analytical + weight * F_numeric
        self.logger.debug("Dynamic Jacobian: weight=%.3f", weight)
        return F_dynamic

    def _get_initial_covariance_std(self, measurement: np.ndarray):
        """
        根据测量 [x, y, a, h] 设置初始标准差，扩展到8维状态，
        这里以 h 值作为尺度因子。
        """
        h_val = measurement[3]
        return np.array([
            3.5 * h_val,   # x
            3.5 * h_val,   # y
            1e-1,          # a
            3.5 * h_val,   # h
            20 * h_val,    # v
            0.1,           # yaw
            0.1,           # yaw_rate
            0.5 * h_val    # acc
        ])

    def _get_adaptive_noise_std(self, mean: np.ndarray, dt: float):
        """
        自适应过程噪声标准差：
          - 以初始协方差（前4个状态）为基准；
          - 对于 v, yaw, yaw_rate, acc，根据当前状态动态调整：
              当加速度 (acc) 或角速度 (yaw_rate) 较大时，增大对应噪声因子；
          - 最终所有噪声乘以 process_noise_factor 进行整体放大。
        """
        base = self._get_initial_covariance_std(mean[:4])
        h_val = mean[3]
        acc_factor = 1.0 + np.abs(mean[7])  # mean[7]为acc
        yaw_rate_factor = 1.0 + np.abs(mean[6])
        v_noise = 10 * self._std_weight_velocity * h_val * acc_factor
        yaw_noise = 0.05 * yaw_rate_factor
        yaw_rate_noise = 0.05 * yaw_rate_factor
        acc_noise = 1.0 * acc_factor
        Q_diag = np.concatenate([base[:4], [v_noise, yaw_noise, yaw_rate_noise, acc_noise]])
        return self.process_noise_factor * Q_diag

    def _get_adaptive_measurement_noise_std(self, mean_measurement: np.ndarray, h_val: float, confidence: float):
        """
        自适应测量噪声标准差：
          - 以初始尺度为基准，结合检测器输出的置信度信息进行调整；
          - 置信度越低（接近0）时，噪声因子越大；置信度越高（接近1）时，噪声因子越小；
          - 调整公式：
                effective_factor = measurement_noise_factor * (1 + (1 - confidence))
          - 当 confidence=1 时，effective_factor 等于 measurement_noise_factor，
            当 confidence=0 时，effective_factor 翻倍。
        """
        # 限制置信度不低于0.1，防止出现过大噪声
        confidence = max(confidence, 0.1)
        effective_factor = self.measurement_noise_factor * (1 + (1 - confidence))
        noise_std = np.array([
            self._std_weight_position * h_val,
            self._std_weight_position * h_val,
            0.15,
            self._std_weight_position * h_val
        ])
        return effective_factor * noise_std
