import json
import os
import shutil
from pathlib import Path


def split_dataset_sequential(original_data_dir, output_dir, train_size=290, val_size=40):
    """
    按顺序划分数据集，自动创建标准目录结构

    参数:
        original_data_dir: 原始数据集目录路径（包含OASIS_dataset.json和对应的imagesTr/labelsTr/masksTr）
        output_dir: 输出目录路径（将自动创建）
        train_size: 训练集大小
        val_size: 验证集大小
    """
    # 设置路径
    original_data_dir = Path(original_data_dir)
    output_dir = Path(output_dir)

    # 验证输入目录
    if not original_data_dir.exists():
        raise FileNotFoundError(f"原始数据集目录不存在: {original_data_dir}")

    json_path = original_data_dir / "OASIS_dataset.json"
    if not json_path.exists():
        raise FileNotFoundError(f"JSON文件不存在: {json_path}")

    # 创建输出目录结构
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "masks").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "masks").mkdir(parents=True, exist_ok=True)
    (output_dir / "test" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "test" / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "test" / "masks").mkdir(parents=True, exist_ok=True)

    # 加载JSON数据
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 验证数据量
    total_samples = len(data['training'])
    test_size = total_samples - train_size - val_size
    assert test_size > 0, f"划分错误：总样本数{total_samples}不足以支持{train_size}(train)+{val_size}(val)的划分"

    # 按顺序划分
    train_set = data['training'][:train_size]
    val_set = data['training'][train_size:train_size + val_size]
    test_set = data['training'][train_size + val_size:]

    # 文件复制函数
    def copy_files(items, split_name):
        for item in items:
            # 构建原始文件绝对路径
            src_img = original_data_dir / item['image']
            src_label = original_data_dir / item['label']
            src_mask = original_data_dir / item['mask']

            # 构建目标路径
            img_name = src_img.name
            label_name = src_label.name
            mask_name = src_mask.name

            dst_img = output_dir / split_name / "images" / img_name
            dst_label = output_dir / split_name / "labels" / label_name
            dst_mask = output_dir / split_name / "masks" / mask_name

            # 复制文件
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_label, dst_label)
            shutil.copy2(src_mask, dst_mask)

    # 执行复制
    copy_files(train_set, "train")
    copy_files(val_set, "val")
    copy_files(test_set, "test")

    # 生成新的JSON文件
    new_json = {
        "dataset_info": {
            "original_dataset": str(original_data_dir),
            "split_created": str(output_dir),
            "total_samples": total_samples,
            "splits": {
                "train": train_size,
                "val": val_size,
                "test": test_size
            }
        },
        "paths": {
            "train": {
                "images": "./train/images",
                "labels": "./train/labels",
                "masks": "./train/masks"
            },
            "val": {
                "images": "./val/images",
                "labels": "./val/labels",
                "masks": "./val/masks"
            },
            "test": {
                "images": "./test/images",
                "labels": "./test/labels",
                "masks": "./test/masks"
            }
        }
    }

    with open(output_dir / "dataset_info.json", 'w') as f:
        json.dump(new_json, f, indent=4)

    print("数据集划分完成！")
    print(f"输出目录结构: {output_dir}")
    print(f"训练集: {train_size}个样本")
    print(f"验证集: {val_size}个样本")
    print(f"测试集: {test_size}个样本")


if __name__ == "__main__":
    # 使用示例 - 替换为您的实际路径
    original_path = r"C:\Users\lenovo\Desktop\OASISYUANBAN\OASIS"  # 包含OASIS_dataset.json的目录
    output_path = r"C:\Users\lenovo\Desktop\OASIS"  # 将创建的新目录

    split_dataset_sequential(
        original_data_dir=original_path,
        output_dir=output_path,
        train_size=290,
        val_size=40
    )