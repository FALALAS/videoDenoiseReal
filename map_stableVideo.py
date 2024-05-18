import cv2
import numpy as np
import time
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def padding(frame, padding_width):
    padding_frame = cv2.copyMakeBorder(frame, padding_width, padding_width, padding_width, padding_width,
                                       cv2.BORDER_REFLECT)
    return padding_frame

start_time = time.time()

# 文件夹路径
clean_folder = './heavyNoise/bm3d'
noised_folder = './heavyNoise/noisy'
output_folder = './heavyNoise/mapsv'
os.makedirs(output_folder, exist_ok=True)

# 第一帧是干净的
clean_path = './heavyNoise/gt.png'
prev_frame = cv2.imread(clean_path)
prev_denoised_frame = prev_frame
denoised_frame = prev_frame
output_path = os.path.join(output_folder, '00000000.png')
cv2.imwrite(output_path, denoised_frame)

# 参数
num_images = 708
win_size = 3
win_area = win_size * win_size
varn = 312.5
padding_width = win_size // 2

h = prev_frame.shape[0]
w = prev_frame.shape[1]

padding_h = h + 2 * padding_width
padding_w = w + 2 * padding_width

# 遍历图片文件
for frame_number in range(1, num_images):
    # 构造文件名
    filename = f'{frame_number:08d}.png'
    noised_path = os.path.join(noised_folder, filename)
    current_frame = cv2.imread(noised_path)
    aligned_frame = prev_denoised_frame
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    aligned_frame_gray = cv2.cvtColor(prev_denoised_frame, cv2.COLOR_BGR2GRAY)

    aligned_frame_gray = aligned_frame_gray.astype(float)
    current_frame_gray = current_frame_gray.astype(float)

    kernel = np.ones((win_size, win_size), np.float32) / win_area
    current_frame_gray2 = current_frame_gray ** 2
    x_mean = cv2.filter2D(current_frame_gray, -1, kernel, borderType=cv2.BORDER_REFLECT)
    x2_mean = cv2.filter2D(current_frame_gray2, -1, kernel, borderType=cv2.BORDER_REFLECT)
    diff2_mean = x2_mean - 2 * aligned_frame_gray * x_mean + aligned_frame_gray ** 2
    varx = diff2_mean - varn
    cnt_wrong = np.sum(np.sum(varx < 0))
    # cnt_wrong = -1
    varx[varx < 0] = 0
    lam = (varn / (varx + 0.1))[:, :, np.newaxis]
    denoised_frame = (current_frame / (1 + lam) + aligned_frame * lam / (1 + lam)).clip(0, 255).astype(np.uint8)

    # denoised_frame = denoised_frame[padding_width: -padding_width, padding_width: -padding_width, :]
    # current_frame = current_frame[padding_width: -padding_width, padding_width: -padding_width, :]
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, denoised_frame)
    prev_denoised_frame = denoised_frame
    prev_frame = current_frame

    current_time = time.time()  # 获取当前时间
    elapsed_time = current_time - start_time  # 计算经过的时间
    print(f"已处理到第 {frame_number} 帧，用时 {elapsed_time:.2f} 秒异常窗口 {cnt_wrong} 个")