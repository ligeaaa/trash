import time
import threading
import pytesseract
import cv2
import numpy as np
import mss
import os
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from loguru import logger

# ----------- 配置 -----------
pytesseract.pytesseract.tesseract_cmd = r"tesseract"  # 可以修改为你的绝对路径
frame_rate = 0.2
keyword = "521"
save_dir = os.path.join(os.getcwd(), "captured")
log_file = os.path.join(save_dir, "monitor.log")
logger.add(log_file, rotation="1 MB")
# --------------------------------

# 检查是否存在相似图片
def check_exist(new_img, save_dir, keyword, time_range=100):
    keyword_dir = os.path.join(save_dir, keyword)
    if not os.path.exists(keyword_dir):
        return False
    now = datetime.now()
    listdir = os.listdir(keyword_dir)
    listdir.sort(reverse=True)
    for filename in listdir:
        if not filename.endswith(".jpg"):
            continue
        try:
            time_part_full = filename.replace(f"{keyword}_", "").replace(".jpg", "")
            time_part = '_'.join(time_part_full.split('_')[:2])  # 只保留时间戳部分
            file_time = datetime.strptime(time_part, "%Y%m%d_%H%M%S")
            diff_seconds = (now - file_time).total_seconds()
            if diff_seconds > time_range:
                break
            existing_img = cv2.imread(os.path.join(keyword_dir, filename))
            if existing_img is None:
                continue

            existing_img_gray = cv2.cvtColor(existing_img, cv2.COLOR_BGR2GRAY)
            new_img_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

            # 检查1：尺寸差异
            if existing_img_gray.shape != new_img_gray.shape:
                continue

            # 检查2：结构相似度
            score_ssim = ssim(existing_img_gray, new_img_gray)
            logger.debug(f"SSIM: {score_ssim:.4f} for {filename}")
            if score_ssim > 0.9:
                logger.info(f"图片 {filename} 与当前截图结构相似度高，判定为重复。")
                return True

            # 检查3：像素绝对误差
            diff = cv2.absdiff(existing_img_gray, new_img_gray)
            mean_diff = np.mean(diff)
            logger.debug(f"Mean pixel diff: {mean_diff:.4f} for {filename}")
            if mean_diff < 5:
                logger.info(f"图片 {filename} 与当前截图像素差异小，判定为重复。")
                return True

        except Exception as e:
            logger.error(f"处理文件 {filename} 出错: {e}")
            continue
    return False

# 主程序逻辑
def start_monitoring(frame_rate, keyword, save_dir):
    frame_rate = float(frame_rate)
    os.makedirs(save_dir, exist_ok=True)
    keyword_dir = os.path.join(save_dir, keyword)
    os.makedirs(keyword_dir, exist_ok=True)

    def capture_and_detect():
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # 全屏

            while True:
                start_time = time.time()
                img = np.array(sct.grab(monitor))
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang='eng')

                n_boxes = len(data['text'])
                found = False
                num = 0

                for i in range(n_boxes):
                    text = data['text'][i].strip()
                    if keyword.lower() in text.lower():
                        found = True
                        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])

                        padding = 50
                        x1 = max(x - padding, 0)
                        y1 = max(y - padding, 0)
                        x2 = min(x + w + padding, img.shape[1])
                        y2 = min(y + h + padding, img.shape[0])

                        cropped_img = img[y1:y2, x1:x2]

                        if not check_exist(cropped_img, save_dir, keyword):
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{keyword}_{timestamp}_{num}.jpg"
                            num += 1
                            save_path = os.path.join(keyword_dir, filename)
                            cv2.imwrite(save_path, cv2.cvtColor(cropped_img, cv2.COLOR_BGRA2BGR))
                            logger.success(f"[+] 发现关键词 '{keyword}'，截图已保存到 {save_path}")
                        else:
                            logger.info(f"[=] 已存在相似截图，跳过保存。")

                if not found:
                    logger.debug(f"[-] 本帧未找到关键词 '{keyword}'")

                elapsed = time.time() - start_time
                sleep_time = max(1.0 / frame_rate - elapsed, 0)
                time.sleep(sleep_time)

    capture_thread = threading.Thread(target=capture_and_detect)
    capture_thread.daemon = True
    capture_thread.start()

if __name__ == "__main__":
    start_monitoring(frame_rate, keyword, save_dir)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n程序退出.")