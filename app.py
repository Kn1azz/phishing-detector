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

# ─── Подробные описания всех признаков на русском ─────────────────────────────
feature_details = {
    'having_IP_Address': 'Наличие IP-адреса вместо доменного имени в URL. Подозрительно, если используется IP.',
    'URL_Length': 'Длина URL-адреса. Очень длинные адреса часто используются в фишинге.',
    'having_At_Symbol': 'Наличие символа @ в адресе. Почти всегда признак фишинга.',
    'Prefix_Suffix': 'Наличие дефиса в доменной части (например, pay-pa1.com). Часто признак подделки.',
    'having_Sub_Domain': 'Количество поддоменов. Слишком много поддоменов — подозрительно.',
    'SSLfinal_State': '**Один из самых важных признаков**. Качество и наличие SSL-сертификата.',
    'Domain_registeration_length': 'Как давно зарегистрирован домен. Новые домены часто используются мошенниками.',
    'port': 'Использование нестандартного порта (не 80/443). Очень подозрительно.',
    'Request_URL': 'Процент внешних ресурсов (картинки, скрипты и т.д.), загружаемых с других доменов.',
    'URL_of_Anchor': 'Процент ссылок <a>, у которых текст и реальный URL не совпадают или подозрительны.',
    'Links_in_tags': 'Процент подозрительных ссылок внутри мета-тегов (link, meta, script и т.д.).',
    'SFH': 'Server Form Handler — куда отправляются данные из форм. Если на чужой домен — плохо.',
    'Submitting_to_email': 'Отправка данных формы напрямую на email. Очень подозрительно.',
    'Abnormal_URL': 'Наличие аномалий в структуре URL (например, двойные слеши, странные символы).',
    'on_mouseover': 'Изменяется ли строка состояния браузера при наведении на ссылки (часто скрывают настоящий URL).',
    'age_of_domain': 'Возраст домена в днях. Молодые домены чаще используются для фишинга.',
    'DNSRecord': 'Наличие записи в DNS. Отсутствие — серьёзный красный флаг.',
    'web_traffic': 'Оценка популярности сайта по трафику (Alexa-like). Низкий трафик подозрителен.',
    'Page_Rank': 'PageRank от Google (примерная оценка). Низкий ранг — подозрительно.',
    'Google_Index': 'Проиндексирован ли сайт в Google. Отсутствие индексации — плохо.',
    'Links_pointing_to_page': 'Количество внешних ссылок, ведущих на эту страницу. Мало ссылок — подозрительно.',
    'Statistical_report': 'Есть ли этот сайт в известных отчётах/базах фишинговых сайтов.'
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

# ─── Конфигурация и боковая панель ────────────────────────────────────────────
st.set_page_config(page_title="Phishing Detector", layout="wide")

st.sidebar.title("🛡️ Phishing Detector")
pages = ["Главная", "Проверка сайта", "Все признаки", "О модели"]
page = st.sidebar.radio("Навигация", pages)

# ─── Главная ──────────────────────────────────────────────────────────────────
if page == "Главная":
    st.title("🛡️ Phishing Detector")
    st.markdown("""
    ### Инструмент для обнаружения фишинговых сайтов
    
    Это приложение на базе модели **CatBoost** помогает быстро определить,  
    является ли сайт безопасным или представляет угрозу.
    
    **Разработчик:** Muhamadasror  
    **Место:** Душанбе, Таджикистан  
    **Год:** 2025–2026
    
    Модель показывает высокую точность и особенно хорошо распознаёт фишинг (высокий recall).
    
    → Переходите в раздел «Проверка сайта», чтобы начать анализ!
    """)

# ─── Проверка сайта ───────────────────────────────────────────────────────────
elif page == "Проверка сайта":
    st.title("🔍 Проверка сайта на фишинг")
    
    st.info("""
    Выберите значения для каждого признака:  
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
                help=feature_details.get(feature, "Описание отсутствует"),
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
                    st.markdown("**Не вводите** личные данные, пароли, карты!")
                else:
                    st.success(f"""
                    ### ✅ Сайт выглядит безопасным
                    Вероятность фишинга: **{proba:.1f}%**
                    """)
                    st.markdown("Но всё равно будьте осторожны — проверяйте адрес и сертификат.")
            except Exception as e:
                st.error("Произошла ошибка при предсказании")
                st.caption(str(e))

# ─── НОВАЯ СТРАНИЦА: Все признаки ─────────────────────────────────────────────
elif page == "Все признаки":
    st.title("📋 Все признаки модели")
    
    st.markdown("""
    Здесь подробное объяснение **каждого** признака, который использует модель.  
    Все значения закодированы одинаково:  
    **-1** — подозрительно / плохо  
    **0** — нейтрально  
    **1** — нормально / хорошо
    """)
    
    for feature, description in feature_details.items():
        with st.expander(f"**{feature}**"):
            st.write(description)

# ─── О модели ─────────────────────────────────────────────────────────────────
elif page == "О модели":
    st.title("ℹ️ О модели")
    
    st.markdown("""
    ### Основные характеристики
    
    - Алгоритм: **CatBoost** (градиентный бустинг)  
    - Датасет: ~11 055 примеров  
    - Признаков: 22 базовых + 4 новых (инженерных)  
    - ROC-AUC: **0.995**  
    - Accuracy: ~**97%**  
    - Recall по фишингу: **95–96%** (особый акцент на то, чтобы не пропускать опасные сайты)
    
    **Самые важные признаки** (по убыванию важности):
    1. URL_of_Anchor  
    2. SSLfinal_State  
    3. Prefix_Suffix  
    4. SFH  
    5. Links_in_tags  
    6. web_traffic  
    ... и другие
    
    Модель специально настроена на высокий recall по классу фишинга.
    """)
    
    st.markdown("---")
    st.caption("Удачи и безопасного серфинга! 🌐🔒")
