#!/bin/bash

# ========== 请根据实际情况修改此变量 ==========
DATA_DIR="/Users/yayunpan/Temp/OASIS"

# ========== 创建目标目录结构 ==========
mkdir -p "$DATA_DIR/train/images" "$DATA_DIR/train/labels" "$DATA_DIR/train/masks"
mkdir -p "$DATA_DIR/valid/images" "$DATA_DIR/valid/labels" "$DATA_DIR/valid/masks"
mkdir -p "$DATA_DIR/test/images"  "$DATA_DIR/test/labels"  "$DATA_DIR/test/masks"

# ========== 移动训练集图像/标签/掩码 ==========
# 若您没有 labelsTr 或 masksTr 文件夹，请注释/删除对应行
mv "$DATA_DIR/imagesTr/"* "$DATA_DIR/train/images/" 2>/dev/null
mv "$DATA_DIR/labelsTr/"* "$DATA_DIR/train/labels/" 2>/dev/null
mv "$DATA_DIR/masksTr/"*  "$DATA_DIR/train/masks/"  2>/dev/null

# ========== 从训练集中抽取部分数据到验证集（可选） ==========
# 例如抽取 20 个文件用于验证集。若不需要验证集，可注释此段。
cd "$DATA_DIR/train/images"
ls | head -n 20 | xargs -I {} mv {} "$DATA_DIR/valid/images/" 2>/dev/null

cd "$DATA_DIR/train/labels"
ls | head -n 20 | xargs -I {} mv {} "$DATA_DIR/valid/labels/" 2>/dev/null

cd "$DATA_DIR/train/masks"
ls | head -n 20 | xargs -I {} mv {} "$DATA_DIR/valid/masks/" 2>/dev/null

# ========== 移动测试集图像/标签/掩码 ==========
# 若您没有 labelsTs 或 masksTs 文件夹，请注释/删除对应行
mv "$DATA_DIR/imagesTs/"* "$DATA_DIR/test/images/" 2>/dev/null
# mv "$DATA_DIR/labelsTs/"* "$DATA_DIR/test/labels/" 2>/dev/null
mv "$DATA_DIR/masksTs/"*  "$DATA_DIR/test/masks/"  2>/dev/null

echo "数据目录重组完毕！"