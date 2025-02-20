import numpy as np
import matplotlib.pyplot as plt

# 加载深度图
# depth_map = np.load('../test_cases/know_depth.npy')
# depth_map = np.load('../test_cases/mask.npy')
# depth_map = np.load('../output/in-the-wild_example/depth_npy/RGB_pred.npy')

depth_mask = np.load('../output/office_bright/3_17/depth_mask.npy')

# 处理无效值（如果有的话）
depth_mask = np.nan_to_num(depth_mask)  # 将nan转换为0
depth_mask = depth_mask

print(f'depth_map: {depth_mask}')

# 使用jet颜色映射显示
plt.figure(figsize=(10, 7))
plt.imshow(depth_mask, cmap='jet')
plt.colorbar(label='深度')
plt.title('深度图')
plt.show()