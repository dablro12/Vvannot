import sys, os, datetime
sys.path.append('../')
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from model.utils.sort import Sort
import csv
import streamlit as st
from model.utils.resize import img_cropper
from ultralytics import SAM

GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

class ObjectTracker:
    def __init__(self, video_path: str, save_path: str, model_path: str, cocolabel: str, confidence_threshold: float, first_position: tuple):
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.tracker_type = 'sort'
        self.position_li = []
        self.position_li.append(first_position)
        self.cap, self.out = self.init_video(video_path, save_path)
        self.model, self.tracker, self.coco128_class = self.init_tracker(model_path, cocolabel)
        self.csv_file, self.csv_writer = self.init_save_csv(save_path)
        self.save_path = save_path  # Initialize self.save_path
        self.seg_model = self.init_segmentation(model_name = 'weights/sam2_t.pt')
    def init_video(self, video_path, save_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None, None
        
        WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Video resolution: {WIDTH}x{HEIGHT}")
        print(f"Video FPS: {FPS}")
        
        cap.set(cv2.CAP_PROP_FPS, FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(save_path, fourcc, FPS, (WIDTH, HEIGHT))
        print('#### [C] init video')
        return cap, out
    
    def init_segmentation(self, model_name:str):
        model = SAM(model_name)
        return model
    
    def init_tracker(self, model_path, cocolabel):
        model = YOLO(model_path)
        if self.tracker_type == 'sort':
            tracker = Sort()
        elif self.tracker_type == 'deepsort':
            tracker = DeepSort(max_age=30, n_init=1)
        
        if not os.path.exists(cocolabel):
            raise FileNotFoundError(f"Error: The file {cocolabel} does not exist.")
        
        with open(cocolabel, 'r') as file:
            coco128_class = file.read().split('\n')
        
        print('#### [C] init tracker')
        return model, tracker, coco128_class
    
    def init_save_csv(self, save_path):
        csv_file = open(save_path.replace('mp4', 'csv'), mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Frame', 'Track ID', 'xmin', 'ymin', 'xmax', 'ymax', 'Confidence'])
        print('#### [C] init csv')
        return csv_file, csv_writer
    
    def video_analysis(self, start, end):
        total = (end - start).total_seconds()
        fps = 1 / total
        return fps

    def human_detection(self, frame):
        """
        Human Object Detection - Yolov8n
        """
        results = []
        # Object Detection - Yolov8n
        detection = self.model(source=[frame], save=False)[0]
        
        # 사람만 필터링 - 사람 class id : 0
        for data in detection.boxes.data.tolist():
            confidence = float(data[4])
            class_id = int(data[5])
            if confidence < self.CONFIDENCE_THRESHOLD:
                continue
            
            if class_id == 0:
                xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])
                print(f"results : {results}")
        return results
    def human_segmentation(self, frame, bbox):
        results = self.seg_model(frame, bboxes=[bbox])
        # frame에 마스크 표시
        for result in results:
            mask = result.masks.xy
            mask = [np.array(m, dtype=np.int32) for m in mask]  # Ensure mask is in the correct format
            cv2.fillPoly(frame, mask, (0, 0, 255))  # Using red color for the mask
        
    def human_sort_tracker(self, results, frame):
        tracks = self.tracker.update(np.array([r[0] for r in results]))  # Extract only the bounding boxes
        for track in tracks:
            track_id = int(track[4])
            print(f"track_id : {track_id}")
            xmin, ymin, xmax, ymax = int(track[0]), int(track[1]), int(track[2]), int(track[3])
            print(f"xmin, ymin, xmax, ymax : {xmin}, {ymin}, {xmax}, {ymax}")
            if not self.target_bool:
                self.target_track_id = track_id
                self.target_bool = True

            if track_id == self.target_track_id:
                # sam2 모델로 사람 영역 추출
                self.human_segmentation(frame, bbox=(xmin, ymin, xmax, ymax))
                # 사람 표시
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
                cv2.putText(frame, f'ID: {track_id}', (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
                
                for result in results:
                    if np.array_equal(track[:4], result[0]):
                        confidence = result[1]
                        self.csv_writer.writerow([self.frame_cnt, track_id, xmin, ymin, xmax, ymax, confidence])
    
    def human_deepsort_tracker(self, results, frame, original_frame, crop_position):
        tracks = self.tracker.update_tracks(results, frame=frame)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            if not self.target_bool:
                self.target_track_id = track.track_id  # 수동 선택
                self.target_bool = True
                
            if track.track_id == self.target_track_id:
                ltrb = track.to_ltrb()
                xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
                
                # sam2 모델로 사람 영역 추출
                self.human_segmentation(frame, bbox = (xmin, ymin, xmax, ymax))
                
                # 사람 표시
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
                cv2.putText(frame, f'ID: {track.track_id}', (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
                
                for result in results:
                    if np.array_equal(track[:4], result[0]):
                        confidence = result[1]
                        self.csv_writer.writerow([self.frame_cnt, track.track_id, xmin, ymin, xmax, ymax, confidence])
    
    def main(self):
        print('#### [C] main start')
        self.track_id = 0
        self.target_bool = False
        self.frame_cnt = 0
        self.avg_fps = []
        while True:
            self.frame_cnt += 1
            start = datetime.datetime.now()
            ret, self.original_frame = self.cap.read()
            if not ret:
                break
            
            results = self.human_detection(self.original_frame)
            if results:  # results가 비어 있지 않은지 확인
                self.position_li.append(results[-1][0])  # 좌표 업데이트
            if self.tracker_type == 'sort':
                self.human_sort_tracker(results, self.original_frame)
            else:
                self.human_tracker(results, self.original_frame, self.original_frame, self.position_li[-1])
            
            end = datetime.datetime.now()
            fps = self.video_analysis(start, end)
            self.avg_fps.append(fps)
            cv2.putText(self.original_frame, f'FPS: {fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            print(f'#### [C] frame : {self.frame_cnt}')
            # 비디오 저장
            self.out.write(self.original_frame)
            
        # 평균 프레임 출력
        self.cap.release()
        self.out.release()
        self.csv_file.close()
        cv2.destroyAllWindows()
        print('#### [C] main end')
        print(f'#### [C] avg fps : {np.mean(np.array(self.avg_fps))}')
        
        return self.save_path

if __name__ == '__main__':
    tracker = ObjectTracker(
        video_path='/home/eiden/eiden/Vvannot/data/tennis_play.mp4',
        save_path='/home/eiden/eiden/Vvannot/data/tennis_play_res-sort.mp4',
        model_path='/home/eiden/eiden/Vvannot/model/weights/yolov8n.pt',
        cocolabel='/home/eiden/eiden/Vvannot/model/weights/coco128.txt',
        confidence_threshold=0.5,
        first_position=(322, 100, 552, 478)
    )
    print(f"save_path : {tracker.main()}")
