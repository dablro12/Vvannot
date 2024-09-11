import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from pages.login_page import login_page
from pages.register_page import register_page
from pages.hello_world_page import hello_world_page
from pages.annot_tool import annot_tool

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if st.session_state['logged_in']:
        st.sidebar.button("Logout", key="sidebar_logout_button", on_click=lambda: st.session_state.update({'logged_in': False, 'page': 'login'}))
        
        if st.session_state.get('page') == 'hello_world':
            hello_world_page()
        elif st.session_state.get('page') == 'annot_tool':
            annot_tool()
    else:
        st.sidebar.button("Register", key="sidebar_register_button", on_click=lambda: st.session_state.update({'page': 'register'}))
        st.sidebar.button("Login", key="sidebar_login_button", on_click=lambda: st.session_state.update({'page': 'login'}))
        
        if st.session_state.get('page') == 'register':
            register_page()
        else:
            login_page()

if __name__ == "__main__":
    main()