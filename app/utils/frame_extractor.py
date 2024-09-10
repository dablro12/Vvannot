import cv2

def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, frame = cap.read()
    
    cap.release()
    if ret:
        return frame, (width, height)
    else:
        return None

def extract_position(df):
    left = df['left'].values[0] * 2
    top = df['top'].values[0] * 2
    width = df['width'].values[0] * 2
    height = df['height'].values[0] * 2
    return (left, top, width, height)
