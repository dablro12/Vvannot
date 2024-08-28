import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
from utils.frame_extractor import extract_first_frame
import numpy as np
import json
from PIL import Image
import pandas as pd 
import os 

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

selected_label = None 
canvas_result = None
annot_json = {}
def annot_tool():
    global annot_json
    global selected_label
    global canvas_result
    st.title("Annotation Tool")
    
    if 'selected_video' in st.session_state:
        video_path = f"uploaded_videos/{st.session_state['selected_video']}"
        frame = extract_first_frame(video_path)
        
        if frame is not None: # 첫 프레임 추출 성공했을떄 
            st.image(frame, channels="BGR", caption="Original Video 영상")
            
            annotations = load_annotations(st.session_state['selected_video'])

            # Sidebar for canvas parameters
            drawing_mode = st.sidebar.selectbox(
                # "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
                "Drawing tool:", ("rect", "point", "line", )
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
                height=frame.shape[0],
                drawing_mode=drawing_mode,
                point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
                key="anno_tool",
            )
            
            if canvas_result.json_data is not None and selected_label:
                # 중목차 이미지 title
                st.image(canvas_result.image_data, caption="Mask 결과")
                # 처음에는 컬럼에 빈데이터로 생성
                # 추가로 label 을 제일 앞으로 추가
                objects_df = pd.json_normalize(canvas_result.json_data["objects"]) # object는 dataframe 형태임 
                
                if not objects_df.empty: # 빈데이터가 아니면
                    # object_df에서 label 이름을 제일 첫번쨰로 해서 annot_json에 할당
                    objects_df.insert(0, 'label', selected_label['name'])
                    objects_df.insert(1, 'label_color', selected_label['color'])
                
                    # row 검사해서 row안에 있는 값이 다른 행과 같은지 보고 같으면 중복이라 판단하고 중복되는 행은 삭제
                    if selected_label['name'] in annot_json:
                        annot_json[selected_label['name']] = pd.concat([annot_json[selected_label['name']], objects_df]).drop_duplicates(subset=objects_df.columns.difference(['label', 'label_color']), keep='first').reset_index(drop=True)
                    else:
                        annot_json[selected_label['name']] = objects_df # 중복되는 행 삭제후 dict에 저장 
                # annot_json의 dataframe들을 streamlit에 출력: df의 칼럼은 모두 동일하므로 하나의 데이터프레임으로 모아서 streamlit에 출력 
                    all_df = update_dataframe(annot_json)   
                # 데이터프레임을 streamlit에 출력
                    st.write(all_df)  # st.dataframe(all_df) 대신 st.write(all_df) 사용
                    canvas_result = None
            
            # json형식으로 저장 
            if st.button("Save Annotation"):
                if not os.path.exists("annotations"):
                    os.makedirs("annotations")
                # json 안에 데이터프레임이 있기에 이를 json 형식으로 바꾼 후 저장하기
                # 데이터프레임을 json 형식으로 바꾸기
                json_data = {label: json.loads(df.to_json(orient='records')) for label, df in annot_json.items()}
                # json 형식으로 바꾼 후 저장하기
                with open(f"annotations/{st.session_state['selected_video']}.json", "w") as f:
                    json.dump(json_data, f)
                st.success("Annotation saved!")
        else:
            st.error("Failed to extract the first frame.")
    else:
        st.error("No video selected.")
