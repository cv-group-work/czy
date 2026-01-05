import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

# ===================== 1. 配置参数（仅需改这3处！）=====================
SAM_WEIGHT_PATH = "sam_vit_b_01ec64.pth"  # 你的SAM权重文件路径
IMAGE_PATH = "apple.jpg"                   # 你的待分割图片路径
SAVE_RESULT_PATH = "samresult.jpg"           # 分割结果保存路径

# ===================== 2. 初始化SAM模型 =====================
# 模型类型与权重对应：vit_b/vit_l/vit_h，和下载的权重一致即可
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动用GPU/CPU

# 加载模型
sam = sam_model_registry[model_type](checkpoint=SAM_WEIGHT_PATH).to(device=device)
predictor = SamPredictor(sam)

# ===================== 3. 加载图像并编码 =====================
image = cv2.imread(IMAGE_PATH)
image_ori = image.copy()  # 保存原图用于绘制结果
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)  # SAM一次性编码，支持多轮提示

# ===================== 4. 方式1：Box Prompt 框提示分割（重点）=====================
# 框坐标格式：[x1, y1, x2, y2] → x1/y1=左上角，x2/y2=右下角
# ✅ 直接改这里的坐标，框选你要分割的目标即可！
box_coords = np.array([4, 220, 964, 1135])

# 执行框分割
masks, _, _ = predictor.predict(
    box=box_coords[None, :],  # 固定格式，无需修改
    multimask_output=False    # 只输出1个最优分割结果
)
mask = masks[0]  # 获取最终分割掩码

# ===================== 5. 方式2：Points Prompt 点提示分割（按需替换）=====================
# 如需用「点提示」，注释上面的Box代码，取消下面注释即可！
# foreground_points = np.array([[400, 300]])  # 前景点：点在要分割的目标上
# background_points = np.array([[100, 100]])  # 背景点：点在不要分割的区域上
# point_labels = np.array([1, 0])             # 1=前景点，0=背景点
# all_points = np.concatenate([foreground_points, background_points], axis=0)
# # 执行点分割
# masks, _, _ = predictor.predict(
#     point_coords=all_points,
#     point_labels=point_labels,
#     multimask_output=False
# )
# mask = masks[0]

# ===================== 6. 可视化结果并保存（自动完成）=====================
# 将掩码转为彩色，叠加到原图上
color = (0, 255, 0)  # 分割掩码颜色：绿色
mask_color = np.zeros_like(image_ori)
mask_color[mask] = color
# 融合原图与掩码（透明度0.5，清晰看到分割效果）
result = cv2.addWeighted(image_ori, 0.7, mask_color, 0.3, 0)
# 绘制Box框（如果用Box提示，显示框选区域）
cv2.rectangle(result, (box_coords[0], box_coords[1]),
              (box_coords[2], box_coords[3]), (255, 0, 0), 2)

# 保存最终结果
cv2.imwrite(SAVE_RESULT_PATH, result)
print(f"✅ 分割完成！结果已保存至：{SAVE_RESULT_PATH}")
print(f"✅ 运行设备：{device} | 模型类型：{model_type}")