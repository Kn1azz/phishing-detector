import streamlit as st 
st.set_page_config(
    page_title="🛡️ Детектор фишинга",
    layout="wide",
    initial_sidebar_state="expanded"
)
import pandas as pd
from catboost import CatBoostClassifier

# Загружаем модель
model = CatBoostClassifier()
model.load_model('phishing_detector_catboost.cbm')

# Список всех фич (точно как в обучении)
FEATURES = [  # Скопируй из твоего X_enhanced.columns
    'having_IP_Address', 'URL_Length', 'having_At_Symbol', 'Prefix_Suffix',
    'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length',
    'port', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH',
    'Submitting_to_email', 'Abnormal_URL', 'on_mouseover', 'RightClick',
    'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index',
    'Links_pointing_to_page', 'Statistical_report', 'total_red_flags',
    'ssl_anchor_interaction', 'no_ssl_short_reg', 'subdomain_prefix'
]

st.title("Phishing Detector 🛡️")
st.markdown("Введите признаки сайта для проверки на фишинг. Значения: -1 (подозрительно), 0 (нейтрально), 1 (нормально).")

# Форма ввода (красиво, с слайдерами/селектами)
inputs = {}
col1, col2 = st.columns(2)
for i, feat in enumerate(FEATURES):
    if i % 2 == 0:
        with col1:
            inputs[feat] = st.selectbox(feat, [-1, 0, 1], index=1)  # По умолчанию 1
    else:
        with col2:
            inputs[feat] = st.selectbox(feat, [-1, 0, 1], index=1)

if st.button("Проверить сайт"):
    df = pd.DataFrame([inputs]).reindex(columns=FEATURES, fill_value=0)
    proba = model.predict_proba(df)[0][1]
    pred = 1 if proba >= 0.5 else 0

    st.success(f"Вероятность фишинга: {proba:.2%}")
    if pred == 1:
        st.error("ОПАСНО! Это может быть фишинг.")
    else:
        st.success("Похоже на безопасный сайт.")

# Добавь футер
st.markdown("---")
st.markdown("Создано Мухаммадом. GitHub: [phishing-detector](https://github.com/твой-username/phishing-detector)")
