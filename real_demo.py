# ==========================================
# 修改后的 run_real_pipeline 函数
# ==========================================
def run_real_pipeline():
    print("🚀 启动 ETL 流水线 (Real-World Scenarios)...")
    
    unified_data = []
    
    for item in REAL_SAMPLES:
        # 1. 实时获取真实图片
        try:
            image = download_image(item["url"])
        except:
            print(f"⚠️ 下载失败: {item['url']}，跳过...")
            continue
            
        w, h = image.size
        
        # 2. 执行数据清洗 (归一化)
        norm_bbox = normalize_bbox(item["raw_bbox"], w, h)
        
        # 3. 填入统一 Schema
        entry = {
            "id": item["id"],
            "data_source": "coco_2017_val_subset",
            "task_type": "detection",
            "media": {
                "image_url": item["url"],
                "image_size": [w, h]
            },
            "spatial_annotations": [{
                "label": item["label"],
                "bbox_2d": norm_bbox
            }],
            "conversations": [
                {
                    "from": "human",
                    "value": f"Identify the {item['label']} in the image."
                },
                {
                    "from": "gpt",
                    "value": f"I found a {item['label']} at <box>{norm_bbox}</box>."
                }
            ]
        }
        unified_data.append(entry)
        
        # 【修改点在这里】不再是只存一张，而是每处理一张，就画一张！
        # 给每张图起个不同的名字，防止覆盖
        output_filename = f"verify_{item['label'].replace(' ', '_')}.png"
        visualize_verification(image, entry, output_filename)

    # 4. 导出 JSONL
    with open("unified_spatial_data_real.jsonl", "w", encoding='utf-8') as f:
        for d in unified_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            
    print("\n✅ 所有数据处理完成! JSONL 已生成。")

# ==========================================
# 同时也要微调一下 visualize_verification 函数，让它接收文件名
# ==========================================
def visualize_verification(image, entry, save_name):
    print(f"🎨 正在生成验证图: {save_name} ...")
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    ax = plt.gca()
    
    w_img, h_img = image.size
    ann = entry["spatial_annotations"][0]
    box = ann["bbox_2d"]
    
    x = box[0] * w_img
    y = box[1] * h_img
    w = (box[2] - box[0]) * w_img
    h = (box[3] - box[1]) * h_img
    
    rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='#00FF00', facecolor='none')
    ax.add_patch(rect)
    
    plt.text(x, y-10, f" {ann['label']} ", color='black', fontsize=12, fontweight='bold', bbox=dict(facecolor='#00FF00', edgecolor='none'))
    
    plt.axis('off')
    plt.title(f"Visual Verification: {entry['id']}")
    
    # 使用传入的文件名保存
    plt.savefig(save_name, bbox_inches='tight')
    plt.close() # 画完这就关掉，释放内存