import cv2
import numpy as np
import os
import glob

def order_points(pts):
  
    pts = pts.reshape(4, 2).astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]   # 左上
    ordered[2] = pts[np.argmax(s)]   # 右下

    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]  # 右上
    ordered[3] = pts[np.argmax(diff)]  # 左下

    return ordered


def find_document_corners(image, debug=False):
    """
    在影像中找出面積最大、且可以被逼近成 4 個頂點的輪廓，回傳排序好的 4 個角點座標 
    """
    orig_h, orig_w = image.shape[:2]

    # 縮小圖片加速邊緣偵測，之後再把座標換算回原始尺寸
    scale = 800.0 / max(orig_h, orig_w)
    resized = cv2.resize(image, (int(orig_w * scale), int(orig_h * scale)))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny 邊緣偵測 + 膨脹，讓邊緣線段更容易連成封閉輪廓
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    if debug:
        dbg = resized.copy()
        cv2.drawContours(dbg, contours, -1, (0, 255, 0), 2)
        # os.makedirs("debug_contours", exist_ok=True)

    best_quad = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 面積太小的輪廓不太可能是我們要的主體，忽略
        area = cv2.contourArea(approx)
        if area < 0.15 * resized.shape[0] * resized.shape[1]:
            continue

        if len(approx) == 4:
            best_quad = approx
            break

    if best_quad is None:
        return None, edged if debug else None

    # 把座標從縮小後的尺寸換算回原始圖片尺寸
    corners = best_quad.reshape(4, 2).astype("float32") / scale
    return order_points(corners), edged if debug else None

def find_corners_by_hough(image, debug=False):
    """
    當 Contour + approxPolyDP 無法找到四邊形時，
    使用 HoughLinesP 找主要直線，再透過直線交點取得四個角點。
    """

    orig_h, orig_w = image.shape[:2]

    # 縮小圖片
    scale = 800.0 / max(orig_h, orig_w)
    resized = cv2.resize(
        image,
        (int(orig_w * scale), int(orig_h * scale))
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny
    edged = cv2.Canny(blurred, 50, 150)

    # Hough Line Transform
    lines = cv2.HoughLinesP(
        edged,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(0.25 * min(resized.shape[:2])),
        maxLineGap=30
    )

    if lines is None:
        return None, edged if debug else None

    # 儲存水平線與垂直線
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(-1)

        dx = x2 - x1
        dy = y2 - y1

        length = np.sqrt(dx ** 2 + dy ** 2)

        if length == 0:
            continue

        angle = np.degrees(np.arctan2(dy, dx))

        # 把角度限制在 [-90, 90]
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180

        # 接近水平
        if abs(angle) < 20:
            horizontal_lines.append(
                (x1, y1, x2, y2, length)
            )

        # 接近垂直
        elif abs(angle) > 70:
            vertical_lines.append(
                (x1, y1, x2, y2, length)
            )

    # 至少需要兩條水平線 + 兩條垂直線
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return None, edged if debug else None

    # 找最上面 / 最下面的水平線
    horizontal_lines = sorted(
        horizontal_lines,
        key=lambda line: line[4],
        reverse=True
    )

    horizontal_candidates = horizontal_lines[:10]

    top_line = min(
        horizontal_candidates,
        key=lambda line: (line[1] + line[3]) / 2
    )

    bottom_line = max(
        horizontal_candidates,
        key=lambda line: (line[1] + line[3]) / 2
    )

    
    # 找最左 / 最右的垂直線
    vertical_lines = sorted(
        vertical_lines,
        key=lambda line: line[4],
        reverse=True
    )

    vertical_candidates = vertical_lines[:10]

    left_line = min(
        vertical_candidates,
        key=lambda line: (line[0] + line[2]) / 2
    )

    right_line = max(
        vertical_candidates,
        key=lambda line: (line[0] + line[2]) / 2
    )

    # 將兩條線段視為無限長直線，計算交點
    def line_intersection(line1, line2):

        x1, y1, x2, y2, _ = line1
        x3, y3, x4, y4, _ = line2

        denominator = (
            (x1 - x2) * (y3 - y4)
            - (y1 - y2) * (x3 - x4)
        )

        if abs(denominator) < 1e-8:
            return None

        px = (
            (x1 * y2 - y1 * x2) * (x3 - x4)
            - (x1 - x2) * (x3 * y4 - y3 * x4)
        ) / denominator

        py = (
            (x1 * y2 - y1 * x2) * (y3 - y4)
            - (y1 - y2) * (x3 * y4 - y3 * x4)
        ) / denominator

        return np.array([px, py], dtype=np.float32)

    # 四個交點
    tl = line_intersection(top_line, left_line)
    tr = line_intersection(top_line, right_line)
    br = line_intersection(bottom_line, right_line)
    bl = line_intersection(bottom_line, left_line)

    if any(p is None for p in [tl, tr, br, bl]):
        return None, edged if debug else None

    corners = np.array(
        [tl, tr, br, bl],
        dtype=np.float32
    )

    # =========================================================
    # 確認四個點大致在圖片範圍附近
    # =========================================================

    h, w = resized.shape[:2]

    if np.any(corners[:, 0] < -0.2 * w):
        return None, edged if debug else None

    if np.any(corners[:, 0] > 1.2 * w):
        return None, edged if debug else None

    if np.any(corners[:, 1] < -0.2 * h):
        return None, edged if debug else None

    if np.any(corners[:, 1] > 1.2 * h):
        return None, edged if debug else None

    # 換回原始圖片座標
    corners = corners / scale

    corners = order_points(corners)

    return corners, edged if debug else None

def manual_select_corners(image, window_name="手動選取四個角點 (依序: 左上 -> 右上 -> 右下 -> 左下，按 q 取消)"):
    """
    當自動偵測失敗時的備用方案：跳出視窗，讓使用者用滑鼠依序點 4 個角點。
    左鍵點擊新增一個點，每點一個會畫一個紅點方便確認位置。
    點滿 4 個點後自動關閉視窗並回傳排序好的座標；按 q 可以直接放棄這張圖。
    回傳 4 個角點 (numpy array) 或 None（使用者取消）。
    """
    points = []
    display = image.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.line(display, points[-2], points[-1], (0, 255, 0), 2)
            cv2.imshow(window_name, display)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if len(points) == 4:
            break
        if key == ord("q"):
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)
    return order_points(np.array(points, dtype="float32"))


def warp_document(image, corners):
    """
    根據四個角點座標，計算輸出矩形的合理寬高，
    再做透視轉換，回傳校正後的影像。
    """
    (tl, tr, br, bl) = corners

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    max_width = max(max_width, 10)
    max_height = max(max_height, 10)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


def process_folder(input_dir, output_dir, debug=False, manual_fallback=False):
    os.makedirs(output_dir, exist_ok=True)
    if debug:
        os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for e in exts:
        image_paths.extend(glob.glob(os.path.join(input_dir, e)))
        image_paths.extend(glob.glob(os.path.join(input_dir, e.upper())))

    if not image_paths:
        print(f"在 {input_dir} 找不到任何圖片檔案")
        return

    success_count = 0
    for path in sorted(image_paths):
        filename = os.path.basename(path)
        image = cv2.imread(path)
        if image is None:
            print(f"[跳過] 無法讀取：{filename}")
            continue

        corners, edged = find_document_corners(image, debug=debug)

        if corners is None:
            print(f"[自動偵測失敗] {filename}")
            if debug and edged is not None:
                cv2.imwrite(os.path.join(output_dir, "debug", f"edges_{filename}"), edged)

            if manual_fallback:
                print("   -> 跳出視窗，請手動依序點選 4 個角點 (左上, 右上, 右下, 左下)，按 q 跳過此圖")
                corners = manual_select_corners(image)
                if corners is None:
                    print(f"   [跳過] 使用者取消：{filename}")
                    continue
            else:
                print(f"   [跳過] {filename}（如需手動補點，執行時將 manual_fallback 設為 True）")
                continue

        warped = warp_document(image, corners)
        out_path = os.path.join(output_dir, f"corrected_{filename}")
        cv2.imwrite(out_path, warped)
        success_count += 1
        print(f"[成功] {filename} -> {out_path}")

        if debug:
            vis = image.copy()
            for (x, y) in corners:
                cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255), -1)
            cv2.polylines(vis, [corners.astype(int)], True, (0, 255, 0), 3)
            cv2.imwrite(os.path.join(output_dir, "debug", f"corners_{filename}"), vis)

    print(f"\n完成：{success_count}/{len(image_paths)} 張圖片成功校正")


if __name__ == "__main__":
    
    INPUT_DIR = "Quiz3_Perspective_Transform/images"         
    OUTPUT_DIR = "Quiz3_Perspective_Transform/output/corrected"      
    DEBUG = True                  # True: 額外輸出偵測過程的除錯圖
    MANUAL_FALLBACK = True        # True: 自動偵測失敗時跳出視窗讓你手動點4個角點

    process_folder(INPUT_DIR, OUTPUT_DIR, debug=DEBUG, manual_fallback=MANUAL_FALLBACK)