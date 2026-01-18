import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from catboost import CatBoostClassifier

# ========================================
# НАСТРОЙКА СТРАНИЦЫ
# ========================================

st.set_page_config(
    page_title="🛡️ Детектор фишинга",
    layout="wide",
    initial_sidebar_state="expanded"
)
