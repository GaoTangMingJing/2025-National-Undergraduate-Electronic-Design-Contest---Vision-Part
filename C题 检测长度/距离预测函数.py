import numpy as np
from scipy import interpolate

def predict_distance(height_px):
    """
    根据像素高度预测实际距离（厘米）
    
    参数:
        height_px (float): 检测到的目标像素高度值
    
    返回:
        float: 预测的实际距离（单位：厘米）
    
    说明:
        基于校准数据：
        距离(cm): [200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100]
        高度(px): [172.5, 181.5, 193.04, 204.14, 217.15, 232.29, 250.09, 270.06, 292.55, 320.54, 354.02]
    """
    # 校准数据
    distances = np.array([200, 195, 190, 185, 180, 175, 170, 165, 160, 155, 150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100])
    height_pix = np.array([170.5, 174.50, 179.5, 184.5, 188.50, 195.0, 200.0, 206.00, 212.5, 218.5, 226.50, 234.50, 243.01, 252.0, 261.00, 272.0, 283.50, 295.5, 308.5, 322.00, 339.0])
    
    # 创建线性插值器（使用您提供的校准数据）
    interpolator = interpolate.interp1d(
        height_pix,
        distances,
        kind='linear',          # 线性插值
        fill_value='extrapolate' # 允许外推超出数据范围
    )
    
    # 预测并返回距离
    return float(interpolator(height_px))

# 测试函数
if __name__ == "__main__":
    # 测试数据点（包含校准点和新点）
    test_heights = [
        170, 172.5, 180, 193.04, 200, 217.15, 
        230, 250.09, 270, 292.55, 300, 320.54, 350, 354.02,
        172.5, 181.5, 193.04, 204.14, 217.15, 232.29, 250.09, 270.06, 292.55, 320.54, 354.02
    ]
    
    print("像素高度(px) → 预测距离(cm)")
    print("-------------------------")
    for h in test_heights:
        distance = predict_distance(h)
        print(f"{h:>8.2f} px → {distance:>6.2f} cm")
    
    # 可视化结果（可选）
    import matplotlib.pyplot as plt
    
    # 原始校准数据
    distances = [200, 195, 190, 185, 180, 175, 170, 165, 160, 155, 150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100]
    height_pix = [170.5, 174.50, 179.5, 184.5, 188.50, 195.0, 200.0, 206.00, 212.5, 218.5, 226.50, 234.50, 243.01, 252.0, 261.00, 272.0, 283.50, 295.5, 308.5, 322.00, 339.0]
    
    # 生成拟合曲线
    h_range = np.linspace(min(height_pix)*0.9, max(height_pix)*1.1, 100)
    d_pred = [predict_distance(h) for h in h_range]
    
    plt.figure(figsize=(10, 6))
    plt.plot(height_pix, distances, 'bo', label='校准点')
    plt.plot(h_range, d_pred, 'r-', label='拟合曲线')
    plt.xlabel('像素高度 (px)')
    plt.ylabel('距离 (cm)')
    plt.title('像素高度与距离关系')
    plt.grid(True)
    plt.legend()
    
    # 标记测试点
    for h in test_heights:
        d = predict_distance(h)
        plt.plot(h, d, 'go', markersize=8)
        plt.annotate(f'{h}px→{d:.1f}cm', (h, d), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.show()