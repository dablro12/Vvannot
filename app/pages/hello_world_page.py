import streamlit as st
import os

def hello_world_page():
    st.title("Hello World")
    st.write("Hello, World!")
    
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file is not None:
        video_dir = "uploaded_videos"
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        
        video_path = os.path.join(video_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {uploaded_file.name} successfully!")
        
        if 'uploaded_videos' not in st.session_state:
            st.session_state['uploaded_videos'] = []
        
        if uploaded_file.name not in st.session_state['uploaded_videos']:
            st.session_state['uploaded_videos'].append(uploaded_file.name)
        
        st.write("### Uploaded Videos")
        for video in st.session_state['uploaded_videos']:
            if st.button(video):
                st.session_state['selected_video'] = video
                st.session_state['page'] = 'annot_tool'
                st.rerun()