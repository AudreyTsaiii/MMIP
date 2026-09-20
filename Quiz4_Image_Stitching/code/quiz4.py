import cv2
import numpy as np
import os
 
def preprocess(image):
    """
    前處理：轉灰階前先用 CLAHE（限制對比自適應直方圖均衡化）強化局部對比，
    可以在亮度不均、逆光或整體偏暗的照片上提升特徵點偵測的穩定性。
    也做一次輕微去雜訊 (bilateral filter)，避免雜訊被誤判成特徵點。
    """
    denoised = cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
 
 
def detect_and_match(img1, img2, ratio=0.75):
    """
    用 SIFT 偵測兩張圖的關鍵點與描述子，再用 KNN + Lowe's ratio test 篩選匹配。
    回傳: kp1, kp2, good_matches
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
 
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
 
    if des1 is None or des2 is None:
        return kp1, kp2, []
 
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(des1, des2, k=2)
 
    good_matches = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        # Lowe's ratio test：最佳匹配要明顯比次佳匹配好，才算可靠
        if m.distance < ratio * n.distance:
            good_matches.append(m)
 
    return kp1, kp2, good_matches
 
 
def stitch(img1, img2, kp1, kp2, good_matches, min_matches=10):
    """
    用篩選過的匹配點估計 Homography，將 img2 warp 到 img1 的座標系並拼接。
    回傳拼接後影像，若匹配點不足則回傳 None。
    """
    if len(good_matches) < min_matches:
        return None, None
 
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
 
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None, None
 
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
 
    # 計算 img2 四個角點轉換後會落在哪裡，藉此決定輸出畫布大小，避免畫面被裁掉
    corners_img2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_img2, H)
    all_corners = np.concatenate(
        (warped_corners, np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)),
        axis=0
    )
 
    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)
 
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    output_size = (x_max - x_min, y_max - y_min)
 
    panorama = cv2.warpPerspective(img2, translation @ H, output_size)
    panorama[-y_min:-y_min + h1, -x_min:-x_min + w1] = np.where(
        panorama[-y_min:-y_min + h1, -x_min:-x_min + w1] == 0,
        img1,
        panorama[-y_min:-y_min + h1, -x_min:-x_min + w1]
    )
    # 重疊區域取兩張圖像素的較大值，簡單處理重疊融合，避免全黑接縫
    overlap_region = img1
    existing = panorama[-y_min:-y_min + h1, -x_min:-x_min + w1]
    blended = np.maximum(existing, overlap_region)
    panorama[-y_min:-y_min + h1, -x_min:-x_min + w1] = blended
 
    return panorama, mask
 
 
def main(img1_path, img2_path, output_path, use_preprocess=False, debug=False, min_matches=10):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print("讀取圖片失敗，請確認路徑正確")
        return
 
    proc1, proc2 = img1, img2
    if use_preprocess:
        proc1 = preprocess(img1)
        proc2 = preprocess(img2)
 
    kp1, kp2, good_matches = detect_and_match(proc1, proc2)
    print(f"偵測到關鍵點：img1={len(kp1)}, img2={len(kp2)}")
    print(f"通過 ratio test 的匹配點數：{len(good_matches)}")
 
    if debug:
        kp_img1 = cv2.drawKeypoints(proc1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        kp_img2 = cv2.drawKeypoints(proc2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imwrite("Quiz4_Image_Stitching/output/debug/debug_keypoints_img1.jpg", kp_img1)
        # cv2.imwrite("Quiz4_Image_Stitching/output/debug/debug_keypoints_img2.jpg", kp_img2)
 
        match_vis = cv2.drawMatches(
            proc1, kp1, proc2, kp2, good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite("Quiz4_Image_Stitching/output/debug/debug_matches2.jpg", match_vis)
        print("已輸出 debug_keypoints_img1.jpg / debug_keypoints_img2.jpg / debug_matches.jpg")
 
    panorama, mask = stitch(img1, img2, kp1, kp2, good_matches, min_matches=min_matches)
 
    if panorama is None:
        print(f"[失敗] 匹配點數不足 (需要至少 {min_matches} 個)，無法估計 Homography，拼接失敗")
        return
 
    cv2.imwrite(output_path, panorama)
    inlier_count = int(mask.sum()) if mask is not None else 0
    print(f"[成功] RANSAC inlier 數：{inlier_count}/{len(good_matches)}")
    print(f"拼接結果已輸出：{output_path}")
 
 
if __name__ == "__main__":
   
    IMG1_PATH = "Quiz4_Image_Stitching/images/IMG_0166.jpg"
    IMG2_PATH = "Quiz4_Image_Stitching/images/IMG_0168.jpg"
    OUTPUT_PATH = "Quiz4_Image_Stitching/output/merge2.jpg"
    USE_PREPROCESS = True     # True: 拼接前先做 CLAHE 前處理
    DEBUG = True              # True: 輸出關鍵點圖與匹配連線圖
    MIN_MATCHES = 10          # 估計 Homography 所需的最少匹配點數
 
    main(IMG1_PATH, IMG2_PATH, OUTPUT_PATH,
         use_preprocess=USE_PREPROCESS, debug=DEBUG, min_matches=MIN_MATCHES)
 