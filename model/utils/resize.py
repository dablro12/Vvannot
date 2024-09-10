def img_cropper(frame, position):
    x, y, w, h = position
    return frame[y:y+h, x:x+w]