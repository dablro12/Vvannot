# Vvannot - Video Object Annotation Tool
Vvannot : Video Object Tracking annotation Tool with Open-Source

# Main Page
![main](./img/mainpage.png)

# Result
![result](./img/demo_example.gif)

# Install Requirement
0. cd vvannot
1. conda create -n {env_name} python=3.8
2. conda activate {env_name} .ENV
3. pip install -r requirements.txt

# Streamlit Run
streamlit run app/main.py
* Upload DIr : pages
* Save Dir : annotations/{video_name}.mp4
* Weight Dir : weights/{weight}.pth
* Annotation Dir : app/annotations/{video_name}.json

# Usage 
1. Register Your Account
2. Upload Video
3. Annotate Object *Bounding-box 