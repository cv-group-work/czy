from flask import Flask, request, jsonify
import os
import base64
import funsam
import sam
import json
app = Flask(__name__)
# ✅ 1. 图片保存目录（默认不变，自动创建）
SAVE_IMG_DIR = "image1"#保存原图
SAVE_IMG_DIR1 = "image2"#保存修改后的图
os.makedirs(SAVE_IMG_DIR, exist_ok=True)
# ✅ 2. 接口路径（默认不变）
@app.route("/receive_data", methods=["POST"])
def receive_data():
    try:
        # 解析常数
        x1 = float(request.form.get("x1"))
        y1 = float(request.form.get("y1"))
        x2 = float(request.form.get("x2"))
        y2 = float(request.form.get("y2"))
        x3 = float(request.form.get("x3"))
        y3 = float(request.form.get("y3"))
        x4 = float(request.form.get("x4"))
        y4 = float(request.form.get("y4"))
        print(x1, y1)
        print(x2, y2)
        print(x3, y3)
        print(x4, y4)
        # 解析图片
        img_file = request.files.get("img_file")
        if not img_file:
            return jsonify({"code": 400, "msg": "未上传图片"}), 400
        # 保存图片
        img_save_path = os.path.join(SAVE_IMG_DIR, img_file.filename)
        img_file.save(img_save_path)
        print("原图已保存到", img_save_path)
        img_save_path1 = os.path.join(SAVE_IMG_DIR1, img_file.filename)
        print("修改后的保存地址", img_save_path)
        funsam.fun_sam(img_save_path,img_save_path1,x1,y1,x2,y2,x3,y3,x4,y4)
        # 返回响应
        # 1. 二进制读取图片文件
        with open(img_save_path1, "rb") as f:
            img_bytes = f.read()
        # 2. 核心：打包「业务数据」+「Base64格式文件」为复合字典
        # 图片二进制 → Base64字符串（才能和JSON数据共存于字典）
        res_data = {
            "code": 200,
            "msg": "接收成功",
            "data": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "x3": x3, "y4": y4, },
            "file_base64": base64.b64encode(img_bytes).decode("utf-8")  # 追加文件数据
        }
        return jsonify(res_data), 200
    except Exception as e:
        return jsonify({"code": 500, "msg": f"失败：{str(e)}"}), 500
if __name__ == "__main__":
    # ✅ 仅需修改这里的端口（例：8080），其余默认
    app.run(host="0.0.0.0", port=8080, debug=False)
