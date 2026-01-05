import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
def fun_sam(img_path,save_path,x1,y1,x2,y2,x3,y3,x4,y4):
    # ===================== 1. 配置参数（按需修改，标注清晰）=====================
    #SAM_WEIGHT_PATH = "sam_vit_b_01ec64.pth"  # SAM权重文件路径
    #model_type = "vit_b"  # 模型类型：vit_b/vit_l/vit_h
    SAM_WEIGHT_PATH = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    IMAGE_PATH = img_path  # 待分割图片路径
    SAVE_RESULT_PATH = save_path  # 结果保存路径

    # ===================== 2. 初始化SAM模型（固定代码，无需修改）=====================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=SAM_WEIGHT_PATH).to(device=device)
    predictor = SamPredictor(sam)
    print(f"✅ 模型加载完成 | 运行设备：{device} | 模型类型：{model_type}")

    # ===================== 5. 图像加载与预处理（固定代码）=====================
    image = cv2.imread(img_path)
    assert image is not None, f"❌ 图片加载失败，请检查路径：{img_path}"
    image_ori = image.copy()  # 保存原图（用于绘制结果）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)  # SAM图像编码（仅需执行1次）

    # ========== ✅ 选择2：使用【点提示分割】（取消注释即可运行，注释上面框提示代码） ==========
    foreground_points = np.array([[x1, y1], [x2, y2]])  # 目标上的点（必填）
    background_points = np.array([[x3, y3], [x4, y4]])  # 背景上的点（可选，可留空[]）
    mask, draw_box = point_prompt_segment(image_rgb, foreground_points, background_points,predictor)

    # ===================== 7. 结果可视化+保存（自动适配两种提示方式）=====================
    # 生成绿色分割掩码（透明度30%，不遮挡原图）
    color = (0, 255, 0)
    mask_color = np.zeros_like(image_ori)
    mask_color[mask] = color
    result = cv2.addWeighted(image_ori, 0.7, mask_color, 0.3, 0)

    # ✅ 智能适配：只有框提示才绘制红色框，点提示不绘制
    if draw_box is not None:
        cv2.rectangle(result, (draw_box[0], draw_box[1]),
                      (draw_box[2], draw_box[3]), (255, 0, 0), 2)

    # 保存最终结果
    cv2.imwrite(save_path, result)
    print(f"\n✅ 分割任务完成！")
    print(f"✅ 结果已保存至：{save_path}")

# ===================== 4. 核心功能：独立的【点提示分割】函数 =====================
def point_prompt_segment(image_rgb, foreground_points, background_points,predictor):
    """
    纯点提示分割函数（独立无依赖）
    :param image_rgb: RGB格式的输入图像
    :param foreground_points: 前景点坐标，格式[[x1,y1], [x2,y2]...]
    :param background_points: 背景点坐标，格式[[x1,y1], [x2,y2]...]
    :return: 分割掩码mask + None（无框坐标，用于统一逻辑）
    """
    # 拼接前景点/背景点 + 对应标签（1=前景，0=背景）
    point_labels = np.array([1] * len(foreground_points) + [0] * len(background_points))
    all_points = np.concatenate([foreground_points, background_points], axis=0)

    # SAM标准点分割流程
    masks, _, _ = predictor.predict(
        point_coords=all_points,
        point_labels=point_labels,
        multimask_output=False
    )
    mask = masks[0]
    return mask, None


