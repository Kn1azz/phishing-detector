import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import os

# Настройки страницы
st.set_page_config(page_title="Phishing Detector 🛡️", page_icon="🛡️", layout="wide")

# Заголовок приложения
st.title("🛡️ Phishing Detector — Защита от Фишинга")
st.markdown("Проверьте сайт на фишинг за секунды. Модель CatBoost — Recall 96.6%")

# Точные 26 фич после очистки + 4 новые комбинации
FEATURES = [
    'having_IP_Address', 'URL_Length', 'having_At_Symbol', 'Prefix_Suffix',
    'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length',
    'port', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH',
    'Submitting_to_email', 'Abnormal_URL', 'on_mouseover',
    'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index',
    'Links_pointing_to_page', 'Statistical_report', 'age_of_domain',
    'total_red_flags', 'ssl_anchor_interaction', 'no_ssl_short_reg', 'subdomain_prefix'
]

# Русские названия фич (для отображения)
FEATURES_RU = {
    'having_IP_Address': 'Наличие IP-адреса в URL',
    'URL_Length': 'Длина URL',
    'having_At_Symbol': 'Символ @ в URL',
    'Prefix_Suffix': 'Префикс/суффикс в домене (дефис)',
    'having_Sub_Domain': 'Наличие субдомена',
    'SSLfinal_State': 'Состояние SSL-сертификата',
    'Domain_registeration_length': 'Длина регистрации домена',
    'port': 'Нестандартный порт',
    'Request_URL': 'URL запроса',
    'URL_of_Anchor': 'URL якоря',
    'Links_in_tags': 'Ссылки в тегах',
    'SFH': 'Server Form Handler',
    'Submitting_to_email': 'Отправка формы на email',
    'Abnormal_URL': 'Аномальный URL',
    'on_mouseover': 'OnMouseOver в JavaScript',
    'DNSRecord': 'DNS-запись',
    'web_traffic': 'Веб-трафик (Alexa rank)',
    'Page_Rank': 'Google Page Rank',
    'Google_Index': 'Индекс Google',
    'Links_pointing_to_page': 'Ссылки, указывающие на страницу',
    'Statistical_report': 'Статистический отчёт (PhishTank)',
    'age_of_domain': 'Возраст домена',
    'total_red_flags': 'Общее количество подозрительных признаков',
    'ssl_anchor_interaction': 'Взаимодействие SSL и URL якоря',
    'no_ssl_short_reg': 'Нет SSL + короткая регистрация домена',
    'subdomain_prefix': 'Субдомен + префикс/дефис'
}

# Загрузка модели
model_path = 'phishing_detector_catboost.cbm'
if os.path.exists(model_path):
    model = CatBoostClassifier()
    model.load_model(model_path)
else:
    st.error("Файл модели 'phishing_detector_catboost.cbm' не найден. Загрузите его в корень репозитория.")
    st.stop()

# Боковое меню
page = st.sidebar.selectbox("Выберите раздел", ["Проверить сайт", "Что означают признаки", "О модели"])

if page == "Проверить сайт":
    st.header("📝 Введите характеристики сайта")
    st.write("Заполните признаки ниже (-1 = подозрительно, 0 = нейтрально, 1 = нормально)")

    inputs = {}
    col1, col2 = st.columns(2)
    for i, feat_en in enumerate(FEATURES):
        feat_ru = FEATURES_RU.get(feat_en, feat_en)
        default_idx = 1 if 'SSL' in feat_en or 'Google_Index' in feat_en else 0
        with col1 if i % 2 == 0 else col2:
            inputs[feat_en] = st.selectbox(feat_ru, [-1, 0, 1], index=default_idx)

    if st.button("Проверить сайт", type="primary"):
        with st.spinner("Анализируем..."):
            df_input = pd.DataFrame([inputs])
            df_input = df_input[FEATURES]  # строго тот же порядок
            df_input = df_input.astype(float)  # гарантируем числа

            proba = model.predict_proba(df_input)[0][1]  # вероятность класса 1 (фишинг)

            st.subheader("Результат")
            if proba >= 0.5:
                st.error(f"⚠️ ФИШИНГОВЫЙ САЙТ! Вероятность: **{proba*100:.1f}%**")
                st.markdown("**Рекомендация:** Не вводите данные, это опасно!")
            else:
                st.success(f"✅ БЕЗОПАСНЫЙ САЙТ! Вероятность фишинга: **{proba*100:.1f}%**")
                st.markdown("**Рекомендация:** Похоже на легитимный сайт.")

            # Прогресс-бар
            st.progress(proba)
            st.caption(f"Фишинг: {proba*100:.1f}% | Безопасность: {(1-proba)*100:.1f}%")

elif page == "Что означают признаки":
    st.header("📖 Что означают признаки")
    st.markdown("""
    Значения:  
    - **-1** — Подозрительно (красный флаг, повышает риск фишинга)  
    - **0** — Нейтрально  
    - **1** — Нормально (безопасно, снижает риск)
    """)

    for feat_en, desc_ru in FEATURES_RU.items():
        with st.expander(desc_ru):
            st.write(f"**Признак:** {desc_ru}")
            st.write("**-1:** Подозрительно (например, нет SSL, IP вместо домена)") 
            st.write("**0:** Нейтрально (среднее значение)")
            st.write("**1:** Нормально (валидный сертификат, популярный сайт)")

elif page == "О модели":
    st.header("🤖 О модели CatBoost")
    st.markdown("""
    Модель: **CatBoost** — лучший выбор для категориальных данных (-1/0/1).  
    Обучена на улучшенном датасете UCI Phishing Websites с 26 признаками.
    """)

    st.subheader("Ключевые параметры")
    st.markdown("""
    - iterations: 1500 (с early_stopping)  
    - learning_rate: 0.035  
    - depth: 6  
    - class_weights: [1.0, 1.25] (больше веса фишингу)  
    - eval_metric: Recall
    """)

    st.subheader("Результаты на тесте")
    st.markdown("""
    - **Recall (фишинг)**: **0.966** (пропущено всего 32 из 951)  
    - **Precision**: **0.982**  
    - **Accuracy**: **0.98**  
    - **ROC-AUC**: **0.9967**  
    - **False Negatives (пропущенные фишинги)**: **32** (на пороге 0.5)
    """)

    st.subheader("Топ-5 самых важных признаков")
    st.markdown("""
    1. SSLfinal_State — 26.14%  
    2. URL_of_Anchor — 12.77%  
    3. web_traffic — 7.48%  
    4. Links_in_tags — 6.29%  
    5. ssl_anchor_interaction (новая фича) — 5.92%
    """)

    st.success("Модель готова к реальному использованию — быстро и точно! 🌟")

# Футер
st.sidebar.markdown("---")
st.sidebar.write("Разработано Muhammad в Душанбе, 2026")
st.sidebar.write("GitHub: github.com/kn1azz/phishing-detector")
