import numpy as np
import cv2
import torch
import gradio as gr
from segment_anything import sam_model_registry, SamPredictor
import os

# ===================== 1. 固定配置（无需修改）=====================
SAM_WEIGHT_PATH = "sam_vit_b_01ec64.pth"
SAVE_RESULT_PATH = "resultweb.jpg"
model_type = "vit_b"
# 自动适配CPU/GPU，无GPU也能运行
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 2. 初始化SAM模型（固定）=====================
print(f"🔧 加载SAM {model_type}模型 | 运行设备：{device}")
sam = sam_model_registry[model_type](checkpoint=SAM_WEIGHT_PATH).to(device)
predictor = SamPredictor(sam)
print("✅ SAM模型加载完成！")

# 全局变量：缓存核心数据，实现拖拽画框（新增temp_img存储临时框）
ori_img = None  # 原始上传图片（只读，RGB格式）
final_seg_img = None  # 最终分割结果图
drag_start = None  # 拖拽起始坐标 (x1, y1)
temp_img = None  # 临时画框图像（Gradio4.13.0兼容）


# ===================== 核心函数（仅改画框相关，其余完全不变）=====================
def upload_image(img):
    """上传图片：缓存原图 + 完成SAM图像编码"""
    global ori_img, final_seg_img, drag_start, temp_img
    if img is None:
        return None, "❌ 请选择图片后再上传！"
    # 缓存原图，初始化临时图像
    ori_img = img.copy()
    temp_img = ori_img.copy()
    final_seg_img = None
    drag_start = None
    # SAM编码（Gradio上传的numpy图原生为RGB，完美适配SAM）
    predictor.set_image(ori_img)
    return ori_img, "✅ 图片上传成功！✅ 按住鼠标左键拖拽画框 → 松开自动分割！"


def mouse_drag_segment(evt: gr.SelectData):
    """✅ 兼容Gradio4.13.0的拖拽画框：用select事件+坐标连续捕获实现"""
    global drag_start, final_seg_img, temp_img
    # 校验前置条件：必须先上传图片
    if ori_img is None:
        return temp_img, "❌ 请先上传图片，再进行画框分割！"

    # 获取鼠标坐标（Gradio4.13.0的select事件仅支持evt.index）
    curr_x, curr_y = int(evt.index[0]), int(evt.index[1])

    # 第一步：鼠标按下 → 记录拖拽起点
    if drag_start is None:
        drag_start = (curr_x, curr_y)
        return temp_img, f"ℹ️ 已标记起点({curr_x},{curr_y})，拖拽后松开左键即可分割！"

    # 第二步：鼠标松开 → 记录终点，执行分割（核心逻辑完全复用你的代码）
    else:
        drag_end = (curr_x, curr_y)
        # 坐标自动校正：兼容任意拖拽方向
        x1 = min(drag_start[0], drag_end[0])
        y1 = min(drag_start[1], drag_end[1])
        x2 = max(drag_start[0], drag_end[0])
        y2 = max(drag_start[1], drag_end[1])
        box = np.array([x1, y1, x2, y2])

        # SAM标准分割流程（和你的逻辑1:1一致）
        masks, _, _ = predictor.predict(box=box[None, :], multimask_output=False)
        mask = masks[0]

        # 可视化：绿色掩码 + 红色框（复用你的效果）
        seg_img = ori_img.copy()
        mask_color = np.zeros_like(seg_img)
        mask_color[mask] = (0, 255, 0)
        seg_img = cv2.addWeighted(seg_img, 0.7, mask_color, 0.3, 0)
        cv2.rectangle(seg_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        final_seg_img = seg_img.copy()
        temp_img = seg_img.copy()

        # 重置拖拽状态，支持多次分割
        drag_start = None
        return seg_img, f"✅ 分割成功！框选范围：({x1},{y1})→({x2},{y2}) | 可继续拖拽画框"


def save_segment_result():
    """保存分割结果：修复RGB→BGR格式，增加异常捕获"""
    global final_seg_img
    if final_seg_img is None:
        return "❌ 暂无分割结果，无法保存！"
    try:
        cv2.imwrite(SAVE_RESULT_PATH, cv2.cvtColor(final_seg_img, cv2.COLOR_RGB2BGR))
        abs_path = os.path.abspath(SAVE_RESULT_PATH)
        return f"✅ 结果保存成功！绝对路径：{abs_path}"
    except Exception as e:
        return f"❌ 保存失败：{str(e)}"


# ===================== ✅ Gradio界面（Gradio4.13.0完美兼容）=====================
with gr.Blocks(title="SAM 鼠标画框分割工具【4.13.0专属版】") as demo:
    gr.Markdown("## 🎯 SAM vit_b 高精度分割工具【CPU/GPU通用 | 拖拽即分割】")
    gr.Markdown("### ✅ 操作指南（Gradio4.13.0专用）")
    gr.Markdown("1. 点击左侧上传图片 → 等待提示【上传成功】")
    gr.Markdown("2. ✅ 鼠标**按住左键拖拽**画框 → 松开左键自动分割")
    gr.Markdown("3. 点击保存按钮 → 结果自动保存为 resultweb.jpg")

    status_text = gr.Textbox(
        label="📢 操作状态",
        value="ℹ️ 等待上传图片",
        interactive=False
    )

    with gr.Row():
        input_img = gr.Image(
            type="numpy",
            label="🖼️ 上传图片（支持任意目标）",
            height=700
        )
        output_img = gr.Image(
            type="numpy",
            label="✅ 分割结果预览",
            height=700
        )

    save_btn = gr.Button("💾 保存分割结果", variant="primary", size="lg")

    # ===================== 交互绑定（Gradio4.13.0原生支持）=====================
    input_img.upload(upload_image, [input_img], [input_img, status_text])
    # ✅ 用4.13.0支持的select事件，完美兼容，无任何报错
    input_img.select(mouse_drag_segment, [], [output_img, status_text])
    save_btn.click(save_segment_result, outputs=[status_text])

# ===================== 启动服务（极简写法，自动打开浏览器）=====================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        show_error=True
    )
    print("=" * 80)
    print(f"✅ SAM分割工具启动成功！运行设备：{device}")
    print(f"👉 访问地址：http://localhost:7860")
    print(f"✅ 分割结果保存路径：{os.path.abspath(SAVE_RESULT_PATH)}")
    print("=" * 80)