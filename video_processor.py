import cv2, os
import pandas as pd
from scipy.signal import find_peaks


import cv2, os
import pandas as pd
from scipy.signal import find_peaks

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.motion_data = []

        if not self.cap.isOpened():
            print("Error: Could not open video file.")
            exit()

    def process_video(self):
        # print("Starting Video Processing:")
        fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        frame_number = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            fgmask = fgbg.apply(frame)
            th = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY)[1]
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            total_motion = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    total_motion += area
            self.motion_data.append([frame_number, total_motion])
            frame_number += 1 
        self.cap.release()

    def video_filt(self):
        # print("Starting Video Filtering:")
        df = pd.DataFrame(self.motion_data, columns=['Frame', 'TotalMotion'])
        # print(df)
        peaks, _ = find_peaks(df["TotalMotion"], height=30000)

        consecutive_count = 0
        start_index = None
        frame_ranges = []

        for index, row in df.iterrows():
            if index > peaks[0]:
                if row['TotalMotion'] < 20000:
                    if consecutive_count == 0:
                        start_index = index
                    consecutive_count += 1
                else:
                    if consecutive_count >= 270:
                        end_frame = index - 21
                        if end_frame - start_index > 900:
                            end_frame = start_index + 900
                        frame_ranges.append({'start_frame': start_index + 60, 'end_frame': end_frame})
                    consecutive_count = 0
                    start_index = None

        if consecutive_count >= 270:
            end_frame = df.index[-21]
            if end_frame - start_index > 900:
                end_frame = start_index + 900
            frame_ranges.append({'start_frame': start_index + 60, 'end_frame': end_frame})

        self.frame_ranges_df = pd.DataFrame(frame_ranges)
        df = pd.DataFrame(frame_ranges)
        print("Video Frame Ranges")
        print(df)

    def crop_video(self, start_frame, end_frame, trim_cnt):
        cap = cv2.VideoCapture(self.video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        input_dir = os.path.dirname(self.video_path)
        input_filename = os.path.basename(self.video_path)
        output_video_path = os.path.join(input_dir, f'TRIM_{trim_cnt}_{input_filename}')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1

        cap.release()
        out.release()

        print(f'Video has been cropped and saved as {output_video_path}')