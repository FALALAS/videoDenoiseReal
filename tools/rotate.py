from PIL import Image
import os

# 设置源文件夹和目标文件夹
src_folder = '../lightNoise/noisy'
dst_folder = '../lightNoise/noisy'

# 创建目标文件夹如果它不存在
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

# 遍历源文件夹中的所有文件
for filename in os.listdir(src_folder):
    if filename.endswith('.png'):  # 可以根据需要修改文件类型
        img_path = os.path.join(src_folder, filename)
        img = Image.open(img_path)
        # 旋转图片180度
        img_rotated = img.rotate(180)
        # 保存旋转后的图片到目标文件夹
        img_rotated.save(os.path.join(dst_folder, filename))

print("所有图片已经旋转并保存至", dst_folder)
