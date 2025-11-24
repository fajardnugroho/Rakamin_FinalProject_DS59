import streamlit as st
st.set_page_config(page_title="Recruitment Dashboard", layout="wide")
import pandas as pd

st.sidebar.title("Navigasi")

menu = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "Dashboard",
        "Tes Page"
    ]
)

if menu == "Dashboard":
    from views import dasboard
    dasboard.run()

# elif menu == "Tes Page":
#     from views import _test
#     _test.run()
