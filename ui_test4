import sys
import cv2
import threading
import queue
import numpy as np
import os
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QMessageBox, QGridLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

# === 카메라 보정 파라미터 및 RTSP 주소 설정 ===
calib_files = [
    "C:/code_env/raspi_doc/calibration/cam1/calibration_result_cam1_4k.npz",
    "C:/code_env/raspi_doc/calibration/cam2/calibration_result_cam2_4k.npz",
    "C:/code_env/raspi_doc/calibration/cam3/calibration_result_cam3_4k.npz",
    "C:/code_env/raspi_doc/calibration/cam4/calibration_result_cam4_4k.npz",
]

calib_params = []
for f in calib_files:
    data = np.load(f)
    K = data['K']
    dist = data['dist']
    calib_params.append((K, dist))

rtsp_urls = [
    "rtsp://admin:0099887766a@uriworks-suntech.iptime.org:5540/Streaming/Channels/1101",
    "rtsp://admin:0099887766a@uriworks-suntech.iptime.org:5540/Streaming/Channels/1201",
    "rtsp://admin:0099887766a@uriworks-suntech.iptime.org:5540/Streaming/Channels/1301",
    "rtsp://admin:0099887766a@uriworks-suntech.iptime.org:5540/Streaming/Channels/1401"
]

class VideoPanel(QWidget):  # === 카메라 영상 처리 및 UI 제어 ===
    def __init__(self, panel_id, rtsp_url, K, dist):
        super().__init__()
        self.panel_id = panel_id  # 1~4
        self.rtsp_url = rtsp_url
        self.K = K
        self.dist = dist

        self.result_frame = None
        self.capture_thread = None
        self.stop_threads = True

        # === 탑뷰 변환 관련 변수 ===
        self.H = None
        self.is_topview = False
        self.scale = 5  # 1cm 당 5픽셀
        self.width_cm = 156      # 가로 156cm (4cm x 39)
        self.height_cm = 301.7   # 세로 301.7cm
        self.width_px = round(self.width_cm * self.scale)
        self.height_px = round(self.height_cm * self.scale)
        self.img_width = 3840
        self.img_height = 2160
        self.dst_points = self.generate_dst_points()

        layout = QVBoxLayout()
        self.video_label = QLabel(f"Camera {panel_id}")
        layout.addWidget(self.video_label)

        btn_layout = QHBoxLayout()
        self.btn_connect = QPushButton("카메라 연결")
        self.btn_save = QPushButton("프레임 저장")
        self.btn_edge = QPushButton("엣지")
        self.btn_topview = QPushButton("탑뷰 변환 시작")
        self.btn_exit = QPushButton("카메라 종료")

        self.btn_connect.clicked.connect(self.start_camera)
        self.btn_save.clicked.connect(self.save_frame)
        self.btn_edge.clicked.connect(self.edge_detection)
        self.btn_topview.clicked.connect(self.toggle_topview)
        self.btn_exit.clicked.connect(self.close_camera)

        btn_layout.addWidget(self.btn_connect)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_edge)
        btn_layout.addWidget(self.btn_topview)
        btn_layout.addWidget(self.btn_exit)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)

    # --- 탑뷰 변환용 목적지 좌표 생성 ---
    def generate_dst_points(self):
        # 좌표 순서: Top-left, Top-right, Bottom-right, Bottom-left (시계 방향)
        dst_pts = np.array([
            [0, 0],
            [self.width_px - 1, 0],
            [self.width_px - 1, self.height_px - 1],
            [0, self.height_px - 1]
        ], dtype=np.float32)
        return dst_pts

    # --- txt 파일에서 4개 좌표 불러오기 함수 ---
    def load_points_from_txt(self, filepath):
        points = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    coords_str = line.split('],')
                    for c_str in coords_str:
                        c_str = c_str.replace('[', '').replace(']', '').strip()
                        if not c_str:
                            continue
                        x_str, y_str = c_str.split(',')
                        points.append([float(x_str), float(y_str)])
        except Exception as e:
            print(f"[cam{self.panel_id}] 좌표 파일 읽기 오류: {e}")
            return None
        return np.array(points, dtype=np.float32)

    # --- 호모그래피 계산 및 4K 중앙 배치 포함한 투시 변환 ---
    def perspective_transform(self, img, src_pts):
        H, status = cv2.findHomography(src_pts, self.dst_points, cv2.RANSAC)
        if H is None:
            raise RuntimeError(f"[cam{self.panel_id}] 호모그래피 계산 실패")

        # 중앙 배치 이동 변환 행렬
        offset_x = (self.img_width - self.width_px) // 2
        offset_y = (self.img_height - self.height_px) // 2

        T = np.array([
            [1, 0, offset_x],
            [0, 1, offset_y],
            [0, 0, 1]
        ], dtype=np.float64)

        H_total = T @ H

        warped = cv2.warpPerspective(img, H_total, (self.img_width, self.img_height))
        return warped, H_total

    def rtsp_capture(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        while not self.stop_threads:
            ret, frame = cap.read()
            if not ret:
                print(f"[cam{self.panel_id}] RTSP 프레임 읽기 실패")
                continue

            undistorted = cv2.undistort(frame, self.K, self.dist, None)

            if self.is_topview and self.H is not None:
                try:
                    frame_warped = cv2.warpPerspective(undistorted, self.H, (self.img_width, self.img_height))
                    self.result_frame = frame_warped
                except Exception as e:
                    print(f"[cam{self.panel_id}] 탑뷰 변환 중 오류: {e}")
                    self.result_frame = undistorted
            else:
                self.result_frame = undistorted

        cap.release()
        print(f"[cam{self.panel_id}] rtsp_capture 쓰레드 종료")

    def start_camera(self):
        if self.capture_thread is None or not self.capture_thread.is_alive():
            self.stop_threads = False
            self.capture_thread = threading.Thread(target=self.rtsp_capture, daemon=True)
            self.capture_thread.start()
            print(f"[cam{self.panel_id}] 카메라 연결 및 쓰레드 시작")
        else:
            print(f"[cam{self.panel_id}] 이미 카메라 연결 중")

    def close_camera(self):
        if self.capture_thread and self.capture_thread.is_alive():
            self.stop_threads = True
            print(f"[cam{self.panel_id}] 카메라 연결 종료 요청됨")
        else:
            print(f"[cam{self.panel_id}] 카메라가 이미 종료됨")

    def save_frame(self):
        if self.result_frame is not None:
            base_dir = "save_result"
            cam_dir = os.path.join(base_dir, f"cam{self.panel_id}")

            if self.is_topview:
                save_dir = os.path.join(cam_dir, "top_view")
            else:
                save_dir = os.path.join(cam_dir, "original")

            os.makedirs(save_dir, exist_ok=True)

            now = datetime.now().strftime("%Y%m%d_%H%M%S")

            frame_to_save = self.result_frame.copy()

            filename = os.path.join(save_dir, f"{now}.jpg")
            cv2.imwrite(filename, frame_to_save)
            print(f"[{self.panel_id}] 프레임 저장 완료: {filename}")
        else:
            print(f"[{self.panel_id}] 저장할 프레임이 없습니다.")

    def toggle_topview(self):
        if not self.is_topview:
            # 카메라 번호에 맞게 points 파일명 지정
            txt_path = f"C:/code_env/raspi_doc/topview_test/points{self.panel_id}.txt"
            if not os.path.exists(txt_path):
                QMessageBox.warning(self, "경고", f"[cam{self.panel_id}] 탑뷰 변환용 좌표 파일이 없습니다:\n{txt_path}")
                return

            src_points = self.load_points_from_txt(txt_path)
            if src_points is None or len(src_points) != 4:
                QMessageBox.warning(self, "경고", f"[cam{self.panel_id}] 좌표가 4개가 아닙니다.")
                return

            try:
                self.H = None
                warped_dummy, H = self.perspective_transform(
                    np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8), src_points)
                self.H = H
                self.is_topview = True
                self.btn_topview.setText("탑뷰 변환 종료")
                print(f"[cam{self.panel_id}] 탑뷰 변환 시작")
            except Exception as e:
                QMessageBox.warning(self, "오류", f"[cam{self.panel_id}] 호모그래피 계산 중 오류:\n{e}")
        else:
            self.H = None
            self.is_topview = False
            self.btn_topview.setText("탑뷰 변환 시작")
            print(f"[cam{self.panel_id}] 탑뷰 변환 종료")

    def edge_detection(self):
        if self.result_frame is not None:
            print(f"[{self.panel_id}] 엣지 검출 실행 중...")

            frame_to_process = self.result_frame.copy()

            gray = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)

            patch = gray.astype(np.float32) / 255.0
            mean, std = patch.mean() * 255, patch.std() * 255

            if mean < 80:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                patch = clahe.apply((patch * 255).astype(np.uint8)).astype(np.float32) / 255.0

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_clahe = clahe.apply(gray)

            blur = cv2.GaussianBlur(gray_clahe, (0, 0), 3)
            unsharp = cv2.addWeighted(gray_clahe, 1.5, blur, -0.5, 0)

            mean_val = np.mean(unsharp)
            std_val = np.std(unsharp)
            t_low = max(0, mean_val - std_val if mean_val > std_val else mean_val)
            t_high = min(255, mean_val + std_val)
            if t_low >= t_high:
                t_low = max(0, t_high - 10)

            edges = cv2.Canny(unsharp, t_low, t_high)
            edges_invert = cv2.bitwise_not(edges)
            edges_bgr = cv2.cvtColor(edges_invert, cv2.COLOR_GRAY2BGR)

            base_dir = "save_result"
            cam_dir = os.path.join(base_dir, f"cam{self.panel_id}")

            if self.is_topview:
                edge_dir = os.path.join(cam_dir, "edge")
            else:
                edge_dir = os.path.join(cam_dir, "ori_edge")

            os.makedirs(edge_dir, exist_ok=True)

            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(edge_dir, f"{now}_edge.jpg")
            cv2.imwrite(filename, edges_bgr)

            self.result_frame = edges_bgr

            print(f"[{self.panel_id}] 엣지 검출 완료 및 저장: {filename}")
        else:
            print(f"[{self.panel_id}] 프레임이 없어 엣지 검출 불가")

    def update_frame(self):
        if self.result_frame is not None:
            frame_to_show = self.result_frame.copy()

            rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image).scaled(720, 480, Qt.KeepAspectRatio)
            self.video_label.setPixmap(pixmap)

class MainApp(QWidget):  # === 4개의 VideoPanel과 정합 및 엣지 UI ===
    def __init__(self):
        super().__init__()
        self.setWindowTitle("4-Camera ArUco UI")

        main_layout = QHBoxLayout()
        grid = QGridLayout()

        self.panels = []
        idx = 0
        for i in range(2):
            for j in range(2):
                K, dist = calib_params[idx]
                panel = VideoPanel(panel_id=idx + 1, rtsp_url=rtsp_urls[idx], K=K, dist=dist)
                self.panels.append(panel)
                grid.addWidget(panel, i, j)
                idx += 1

        self.alignment_label = QLabel()
        self.alignment_label.setFixedSize(1080, 720)
        self.alignment_label.setAlignment(Qt.AlignCenter)

        align_btn_layout = QVBoxLayout()
        align_btn_layout.addWidget(self.alignment_label)

        self.align_button = QPushButton("정합 실행")
        self.align_button.clicked.connect(self.align_btn_clicked)
        align_btn_layout.addWidget(self.align_button)

        self.edge_align_button = QPushButton("정합 엣지")
        self.edge_align_button.clicked.connect(self.edge_align_clicked)
        align_btn_layout.addWidget(self.edge_align_button)

        main_layout.addLayout(grid)
        main_layout.addLayout(align_btn_layout)

        self.setLayout(main_layout)

        self.align_result = None
        self.align_save_path = None

    def align_btn_clicked(self):
        print("[MainApp] 정합(오버레이) 버튼 클릭됨")

        frames = []
        for panel in self.panels:
            if panel.is_topview and panel.result_frame is not None:
                frame = panel.result_frame.copy()
                frames.append(frame)
            else:
                frames.append(None)

        # 기준 이미지 설정 (cam1)
        if frames[0] is None:
            print("[MainApp] cam1이 탑뷰 모드 아니거나 프레임 없음. 정합 불가.")
            return

        result = frames[0]

        for idx in range(1, len(frames)):
            frame = frames[idx]
            if frame is not None:
                result = np.maximum(result, frame)
                print(f"[MainApp] cam{idx + 1} 오버레이 정합 적용됨")
            else:
                print(f"[MainApp] cam{idx + 1} 프레임 없음, 건너뜀")

        self.align_result = result

        base_dir = "save_result"
        align_dir = os.path.join(base_dir, "alignment")
        os.makedirs(align_dir, exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.align_save_path = os.path.join(align_dir, f"{now}_overlay_align.jpg")

        cv2.imwrite(self.align_save_path, result)
        print(f"[MainApp] 오버레이 정합 이미지 저장 완료: {self.align_save_path}")

        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(1080, 720, Qt.KeepAspectRatio)
        self.alignment_label.setPixmap(pixmap)

    def edge_align_clicked(self):
        if self.align_result is None:
            print("[MainApp] 먼저 정합 실행 후 시도하세요.")
            return

        print("[MainApp] 정합 엣지 버튼 클릭됨")

        gray = cv2.cvtColor(self.align_result, cv2.COLOR_BGR2GRAY)

        patch = gray.astype(np.float32) / 255.0
        mean, std = patch.mean() * 255, patch.std() * 255

        if mean < 80:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            patch = clahe.apply((patch * 255).astype(np.uint8)).astype(np.float32) / 255.0

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)

        blur = cv2.GaussianBlur(gray_clahe, (0, 0), 3)
        unsharp = cv2.addWeighted(gray_clahe, 1.5, blur, -0.5, 0)

        mean_val = np.mean(unsharp)
        std_val = np.std(unsharp)
        t_low = max(0, mean_val - std_val if mean_val > std_val else mean_val)
        t_high = min(255, mean_val + std_val)
        if t_low >= t_high:
            t_low = max(0, t_high - 10)

        edges = cv2.Canny(unsharp, t_low, t_high)
        edges_inv = cv2.bitwise_not(edges)
        edges_bgr = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

        base_dir = "save_result"
        align_dir = os.path.join(base_dir, "alignment")
        edge_dir = os.path.join(align_dir, "edge")
        os.makedirs(edge_dir, exist_ok=True)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(edge_dir, f"{now}_align_edge.jpg")
        cv2.imwrite(filename, edges_bgr)

        rgb = cv2.cvtColor(edges_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(1080, 720, Qt.KeepAspectRatio)
        self.alignment_label.setPixmap(pixmap)

        print(f"[MainApp] 정합 엣지 이미지 저장 완료: {filename}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
