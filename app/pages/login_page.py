import streamlit as st
from DB.database import init_db

def login_page():
    conn, c = init_db()
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login", key="login_button"):
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        if c.fetchone():
            st.session_state['logged_in'] = True
            st.session_state['page'] = 'hello_world'
            st.rerun()
        else:
            st.error("Invalid username or password")
    conn.close()