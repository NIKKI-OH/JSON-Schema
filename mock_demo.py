import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 1. 模拟环境准备 (代替下载过程)
# ==========================================
def create_dummy_image(index):
    """创建一张模拟图片，上面画个框，假装是数据集里的图"""
    width, height = 640, 480
    # 生成灰色背景图
    image = Image.new('RGB', (width, height), color=(220, 220, 220))
    draw = ImageDraw.Draw(image)
    
    # 在图上随机画一个矩形 (模拟物体)
    x = random.randint(50, 400)
    y = random.randint(50, 300)
    w = random.randint(50, 150)
    h = random.randint(50, 150)
    
    # 画出来，方便可视化验证时对比
    draw.rectangle([x, y, x+w, y+h], outline="blue", width=3)
    
    # 存到本地，模拟图片文件
    img_filename = f"sample_image_{index}.jpg"
    image.save(img_filename)
    
    return image, img_filename, [x, y, w, h]

def normalize_bbox(bbox, w, h):
    """坐标归一化工具"""
    x, y, bw, bh = bbox
    return [
        round(x / w, 4),
        round(y / h, 4),
        round((x + bw) / w, 4),
        round((y + bh) / h, 4)
    ]

# ==========================================
# 2. 核心逻辑：ETL 流水线 (模拟版)
# ==========================================
def run_mock_pipeline():
    print("🚀 启动 ETL 流水线 (模拟数据模式)...")
    
    unified_data = []
    
    # 模拟处理 3 条数据
    for i in range(3):
        # 1. 造假数据
        image, filename, raw_bbox = create_dummy_image(i)
        w, h = image.size
        
        # 2. 数据清洗/归一化
        norm_bbox = normalize_bbox(raw_bbox, w, h)
        label = ["cat", "dog", "robot_arm"][i] # 随机给个标签
        
        print(f"  正在处理样本 {i}: {label}...")
        
        # 3. 填入统一 Schema (这是面试官要看的核心!)
        entry = {
            "id": f"mock_sample_{i:03d}",
            "data_source": "visual_genome_simulated",
            "task_type": "spatial_understanding",
            "media": {
                "image_path": filename,
                "image_size": [w, h]
            },
            "spatial_annotations": [{
                "label": label,
                "bbox_2d": norm_bbox,
                "description": f"A {label} inside the blue box."
            }],
            "conversations": [
                {
                    "from": "human", 
                    "value": f"Where is the {label}?"
                },
                {
                    "from": "gpt", 
                    "value": f"It is located at <box>{norm_bbox}</box>."
                }
            ]
        }
        unified_data.append(entry)
        
        # 保存第一张图用来做可视化验证
        if i == 0:
            verify_data = (image, entry)

    # 4. 保存 JSONL 结果
    output_file = "unified_spatial_data.jsonl"
    with open(output_file, "w", encoding='utf-8') as f:
        for item in unified_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\n✅ ETL 完成! 数据已保存为: {output_file}")
    
    # 5. 生成可视化验证图 (Proof of Work)
    visualize_verification(verify_data[0], verify_data[1])

def visualize_verification(image, entry):
    print("🎨 正在生成验证图片 (Verification Plot)...")
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    ax = plt.gca()
    
    img_w, img_h = image.size
    
    # 读取我们生成的 JSON 数据，把框画回去，证明数据格式是对的
    ann = entry["spatial_annotations"][0]
    box = ann["bbox_2d"] # [x1, y1, x2, y2] 归一化的
    
    # 反归一化
    x = box[0] * img_w
    y = box[1] * img_h
    w = (box[2] - box[0]) * img_w
    h = (box[3] - box[1]) * img_h
    
    # 画红框 (Red Box) - 对应 JSON 里的数据
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    plt.text(x, y-10, f"JSON Label: {ann['label']}", color='red', fontsize=12, fontweight='bold')
    plt.title(f"Verification: JSON Data aligned with Image\nID: {entry['id']}")
    plt.axis('off')
    
    plt.savefig("verification_plot.png")
    print(f"✅ 验证图片已生成: verification_plot.png")
    print("🎉 恭喜！你可以去左侧文件栏查看这两个生成的文件了！")

if __name__ == "__main__":
    run_mock_pipeline()