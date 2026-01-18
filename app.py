import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# ─── Загрузка модели ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model('phishing_detector_catboost.cbm')
    return model

model = load_model()

# ─── Список базовых признаков ─────────────────────────────────────────────────
base_features = [
    'having_IP_Address', 'URL_Length', 'having_At_Symbol', 'Prefix_Suffix',
    'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length', 'port',
    'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH',
    'Submitting_to_email', 'Abnormal_URL', 'on_mouseover', 'age_of_domain',
    'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index',
    'Links_pointing_to_page', 'Statistical_report'
]

# ─── Подробные описания признаков ─────────────────────────────────────────────
feature_descriptions = {
    'having_IP_Address': 'Есть ли IP-адрес вместо домена в URL?',
    'URL_Length': 'Длина URL (очень длинные — подозрительны)',
    'having_At_Symbol': 'Есть ли символ @ в URL?',
    'Prefix_Suffix': 'Есть ли дефис (-) в доменном имени?',
    'having_Sub_Domain': 'Количество поддоменов (много — подозрительно)',
    'SSLfinal_State': 'Качество SSL-сертификата (-1 — плохой/отсутствует)',
    'Domain_registeration_length': 'Срок регистрации домена (короткий — подозрительно)',
    'port': 'Используется ли нестандартный порт?',
    'Request_URL': '% внешних ресурсов в запросах',
    'URL_of_Anchor': '% подозрительных ссылок в якорях (<a> теги)',
    'Links_in_tags': '% подозрительных ссылок в мета-тегах',
    'SFH': 'Server Form Handler (куда отправляется форма)',
    'Submitting_to_email': 'Отправка формы на email?',
    'Abnormal_URL': 'Аномалии в структуре URL',
    'on_mouseover': 'Изменяется ли status bar при наведении мыши?',
    'age_of_domain': 'Возраст домена (молодой — подозрительно)',
    'DNSRecord': 'Есть ли запись в DNS?',
    'web_traffic': 'Популярность сайта по трафику',
    'Page_Rank': 'PageRank от Google',
    'Google_Index': 'Проиндексирован ли сайт Google?',
    'Links_pointing_to_page': 'Количество внешних ссылок на страницу',
    'Statistical_report': 'Есть ли в отчётах о фишинге?'
}

# ─── Функция создания дополнительных признаков ────────────────────────────────
def add_engineered_features(df):
    df = df.copy()
    df['total_red_flags'] = (df == '-1').sum(axis=1).astype(str)
    df['ssl_anchor_interaction'] = (df['SSLfinal_State'].astype(int) *
                                    df['URL_of_Anchor'].astype(int)).astype(str)
    df['no_ssl_short_reg'] = ((df['SSLfinal_State'] == '-1') &
                              (df['Domain_registeration_length'] == '-1')).astype(int).astype(str)
    df['subdomain_prefix'] = (df['having_Sub_Domain'].astype(int) *
                              df['Prefix_Suffix'].astype(int)).astype(str)
    return df

# ─── Конфигурация приложения ──────────────────────────────────────────────────
st.set_page_config(page_title="Phishing Detector", layout="wide", page_icon="🛡️")

# ─── Боковая панель ───────────────────────────────────────────────────────────
st.sidebar.title("🛡️ Phishing Detector")
st.sidebar.markdown("**Обнаружение фишинговых сайтов**")
pages = ["Главная", "Проверка сайта", "Все признаки", "О модели"]
page = st.sidebar.radio("Навигация", pages)

# ─── Главная страница с атмосферными фото ─────────────────────────────────────
if page == "Главная":
    st.title("🛡️ Phishing Detector")
    
    # Качественное атмосферное фото (тёмная тема, кибербезопасность)
    st.image(
        "https://xakep.ru/wp-content/uploads/2018/03/158954/phishing.jpg",
        caption="Защита в цифровом мире",
        use_column_width=True
    )
    
    st.markdown("""
    ### Приложение для проверки сайтов на фишинг
    
    Это инструмент на базе машинного обучения, который помогает определить,  
    является ли сайт фишинговым или легитимным.
    
    **Разработчик:** Muhamadasror  
    =====Душанбе, Таджикистан=====  
    **Год:** 2026
    
    Модель обучена на более чем 11 000 **(после удаления дублткатов меньше)** примерах и показывает очень высокую точность.
    
    **Перейдите в раздел «Проверка сайта»**, чтобы начать анализ!
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            "https://images.unsplash.com/photo-1563986768494-4dee2763ff3f?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            caption="Безопасное соединение",
            use_column_width=True
        )
    
    with col2:
        st.image(
            "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            caption="Кибербезопасность будущего",
            use_column_width=True
        )

# ─── тут страница проверка сайта ───────────────────────────────────────────────────────────
elif page == "Проверка сайта":
    st.title("🔍 Проверка сайта на фишинг")
    
    st.info("""
    Введите значения признаков сайта, который хотите проверить.  
    Все признаки закодированы следующим образом:  
    **-1** — подозрительно / плохо  
    **0** — нейтрально  
    **1** — нормально / хорошо
    """)
    
    col1, col2 = st.columns(2)
    input_data = {}
    
    for i, feature in enumerate(base_features):
        with col1 if i % 2 == 0 else col2:
            val = st.selectbox(
                label=feature,
                options=["-1", "0", "1"],
                index=1,
                help=feature_descriptions.get(feature, "Описание отсутствует"),
                key=f"input_{feature}"
            )
            input_data[feature] = val
    
    if st.button("🚀 Проверить сайт", type="primary", use_container_width=True):
        with st.spinner("Анализирую..."):
            input_df = pd.DataFrame([input_data]).astype(str)
            enhanced_df = add_engineered_features(input_df)
            
            try:
                pred = model.predict(enhanced_df)[0]
                proba = model.predict_proba(enhanced_df)[0][1] * 100
                
                if pred == 1:
                    st.error(f"""
                    ### ⚠️ ОПАСНО! Фишинговый сайт!
                    Вероятность фишинга: **{proba:.1f}%**
                    """)
                    st.markdown("**Не вводите** личные данные, пароли или банковские карты!")
                else:
                    st.success(f"""
                    ### ✅ Сайт выглядит безопасным
                    Вероятность фишинга: **{proba:.1f}%**
                    """)
                    st.markdown("Но всегда будьте осторожны — проверяйте адрес и сертификат вручную.")
            except Exception as e:
                st.error("Произошла ошибка при предсказании модели")
                st.caption(str(e))

# ─── Все признаки ─────────────────────────────────────────────────────────────
elif page == "Все признаки":
    st.title("📋 Подробное описание всех признаков")
    
    st.markdown("""
    Здесь подробно объясняется **каждый** признак модели.  
    Все значения имеют одинаковую кодировку:
    
    - **-1** → подозрительно / плохо (часто указывает на фишинг)  
    - **0** → нейтрально / среднее значение  
    - **1** → нормально / хорошо (характерно для легитимных сайтов)
    """)
    
    for feature in base_features:
        with st.expander(f"**{feature}** — {feature_descriptions[feature]}"):
            st.markdown("### Что проверяет этот признак?")
            st.write(feature_descriptions[feature])
            
            st.markdown("### Что значит каждое значение?")
            
            if feature == 'SSLfinal_State':
                st.markdown("""
                - **-1** → Нет SSL, просрочен, самоподписанный или от ненадёжного центра — **очень плохо!**  
                - **0** → Частично валидный сертификат  
                - **1** → Полностью валидный HTTPS от надёжного центра — **очень хорошо**
                """)
                st.caption("Это один из самых важных признаков модели!")
                
            elif feature == 'URL_of_Anchor':
                st.markdown("""
                - **-1** → Много ссылок <a>, где текст не совпадает с реальным URL — **очень подозрительно**  
                - **0** → Среднее количество  
                - **1** → Все ссылки честные и ведут туда, куда указано
                """)
                st.caption("Самый важный признак по важности в модели!")
                
            else:
                st.markdown("""
                - **-1** → Подозрительное значение (повышает риск фишинга)  
                - **0** → Нейтральное  
                - **1** → Нормальное, безопасное значение
                """)
            
            st.caption("Чем больше признаков с -1 — тем выше вероятность фишинга.")

# ─── О модели ========================================================
elif page == "О модели":
    st.title("ℹ️ О модели")
    
    st.markdown("""
    ### Основные характеристики модели
    
    - **Алгоритм:** CatBoost (градиентный бустинг)  
    - **Датасет:** ~11 055 примеров  
    - **Признаков:** 22 базовых + 4 новых (инженерных)  
    - **ROC-AUC:** **0.995**  
    - **Accuracy:** ~**97%**  
    - **Recall по фишингу:** **95–96%** (главная цель — не пропустить опасные сайты!)
    
    **Топ-важные признаки (по убыванию):**
    1. URL_of_Anchor  
    2. SSLfinal_State  
    3. Prefix_Suffix  
    4. SFH  
    5. Links_in_tags  
    6. web_traffic
    
    Модель специально настроена на **высокий recall** по классу фишинга.
    """)
    
    st.markdown("---")
    st.caption("Удачи и безопасного серфинга в интернете! 🌐🔒")
