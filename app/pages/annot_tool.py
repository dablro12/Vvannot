import sys, os 
sys.path.append('../')

import time 
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from app.utils.frame_extractor import extract_first_frame, extract_position 
import json
from PIL import Image, ImageOps

import pandas as pd 
import os 
from model.tracker import ObjectTracker

def load_annotations(video_name):
    try:
        with open(f"annotations/{video_name}.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def update_dataframe(annot_json):
    all_df = pd.DataFrame()
    for key, df in annot_json.items():
        all_df = pd.concat([all_df, df], ignore_index=True)
        
    # 중복 방지를 위해 label과 label_color을 제외하고 같은 행은 하위 행에서 삭제
    all_df = all_df.drop_duplicates(subset=all_df.columns.difference(['label', 'label_color']), keep='first')
    
    return all_df

def crop_img_viewer(canvas_result, bg_image):
    if canvas_result.json_data is None:
        st.error("No annotations found.")
        return

    # Crop and display the rectangle annotations
    for obj in canvas_result.json_data["objects"]:
        if obj["type"] == "rect":
            # Adjust coordinates for the original image size
            left = obj["left"] * 2
            top = obj["top"] * 2
            width = obj["width"] * 2
            height = obj["height"] * 2
            right = left + width
            bottom = top + height

            # Ensure the image is in the correct mode
            if bg_image.mode != "RGB":
                bg_image = bg_image.convert("RGB")

            cropped_image = bg_image.crop((left, top, right, bottom))
            st.image(cropped_image, caption=f"Cropped {selected_label['name']}")

selected_label = None 
canvas_result = None
annot_json = {}
save_path = None

def annot_tool():
    global annot_json
    global selected_label
    global canvas_result
    global save_path
    st.title("Annotation Tool")
    
    if 'selected_video' in st.session_state:
        # 절대경로로 바꾸기 
        video_path = os.path.abspath(f"uploaded_videos/{st.session_state['selected_video']}")
        frame, size = extract_first_frame(video_path)
        
        if frame is not None: # 첫 프레임 추출 성공했을떄 
            st.image(frame, channels="BGR", caption="Original Video 영상")
            
            annotations = load_annotations(st.session_state['selected_video'])
            # Sidebar for canvas parameters
            drawing_mode = st.sidebar.selectbox(
                # "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
                "Drawing tool:", ("rect", "point", "line")
            )
            
            stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
            if drawing_mode == 'point':
                point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)

            realtime_update = st.sidebar.checkbox("Update in realtime", True)

            # Label settings
            if 'labels' not in st.session_state:
                st.session_state['labels'] = []

            with st.sidebar.form(key='label_form'):
                label_name = st.text_input("Label name", "")
                label_color = st.color_picker("Label color", "#FF0000")
                if st.form_submit_button("Add Label"):
                    st.session_state['labels'].append({'name': label_name, 'color': label_color})

            st.sidebar.write("Labels:")
            for label in st.session_state['labels']:
                st.sidebar.write(f"{label['name']} - {label['color']}")
                if st.sidebar.button(f"Delete {label['name']}"):
                    st.session_state['labels'].remove(label)
            if not st.session_state['labels']:
                st.sidebar.error("Please add at least one label before annotating.")
                return 
            
            selected_label = st.sidebar.selectbox("Select label for annotation", st.session_state['labels'], format_func=lambda x: x['name'])
            # Update stroke and background color based on selected label
            if selected_label:
                stroke_color = selected_label['color']
                bg_color = selected_label['color']
            else:
                stroke_color = "#00FF00" # 기본 색상 설정
                bg_color = "#eee" # 기본 색상 설정

            #PIL Image로 바꾸기
            bg_image = Image.fromarray(frame)
            
            canvas_result = st_canvas(
                fill_color = "rgba(0, 255, 0, 0.3)",  # 초록형광색 형태
                stroke_width = stroke_width,
                stroke_color = stroke_color,
                background_color = bg_color,
                background_image = bg_image,
                update_streamlit=realtime_update,
                height=frame.shape[0]//2,
                width=frame.shape[1]//2,
                drawing_mode=drawing_mode,
                point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
                key="anno_tool",
            )
            
            if canvas_result and canvas_result.json_data is not None and selected_label:
                # 중목차 이미지 title
                objects_df = pd.json_normalize(canvas_result.json_data["objects"]) # object는 dataframe 형태임
                
                if not objects_df.empty: # 빈데이터가 아니면
                    # object_df에서 label 이름을 제일 첫번쨰로 해서 annot_json에 할당
                    objects_df.insert(0, 'label', selected_label['name'])
                    objects_df.insert(1, 'label_color', selected_label['color'])
                    # Crop and display the rectangle annotations
                    # crop_img_viewer(canvas_result, bg_image)
            # Process 버튼 생성
            process_button = st.button("Process")
            
            if process_button and not objects_df.empty:
                # 진행중인 프로세스 상태 보여주기 - ObjectTracker가 처리할때 까지 기다리기 
                with st.spinner("Processing..."):
                    
                    # save_path 얻어질때까지 하기 
                    while save_path is None:
                        first_position = extract_position(objects_df)
                        save_path = ObjectTracker(
                            video_path=video_path,
                            save_path=os.path.abspath(f"app/annotations/{st.session_state['selected_video']}"),
                            model_path=os.path.abspath('model/weights/yolov8n.pt'),
                            cocolabel=os.path.abspath('model/weights/coco128.txt'),
                            confidence_threshold=0.5,
                            first_position=first_position
                        ).main()

            if save_path is not None:
                print(save_path)
                # Ensure the video file is valid and exists
                if os.path.exists(save_path):
                    st.success("Video processed successfully!")
                    # Display processed video
                    # video_file = open(save_path, 'rb').read()
                    # st.video(save_path, format = 'video/{save_path.split(".")[-1]}')  # You can pass the file path directly
                    # Create a download button for the processed video
                    with open(save_path, "rb") as video_file:
                        video_bytes = video_file.read()
                        st.download_button(
                            label="Download Processed Video",
                            data=video_bytes,
                            file_name=os.path.basename(save_path),
                            mime=f"video/{save_path.split('.')[-1]}"
                        )
                        
                    # Provide option to download the JSON file
                    json_path = save_path.replace(save_path.split('.')[-1], 'json')
                    if os.path.exists(json_path):
                        with open(json_path, 'rb') as json_file:
                            json_bytes = json_file.read()
                            st.download_button(
                                label='Download Annotation Mask JSON',
                                data=json_bytes,
                                file_name=os.path.basename(json_path),
                            )
                else:
                    st.error("The processed video could not be found.")

                
        else:
            st.error("Failed to extract the first frame.")
    else:
        st.error("No video selected.")
