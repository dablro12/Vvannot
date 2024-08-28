import streamlit as st
from DB.database import init_db

def register_page():
    conn, c = init_db()
    st.title("Register Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register", key="register_button"):
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        st.success("User registered successfully!")
    conn.close()
