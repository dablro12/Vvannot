import cv2

def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        return None

def crop_video(video_path):
    """
    진행중
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()