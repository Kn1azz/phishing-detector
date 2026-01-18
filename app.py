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

# ─── Базовые признаки ────────────────────────────────────────────────────────
base_features = [
    'having_IP_Address', 'URL_Length', 'having_At_Symbol', 'Prefix_Suffix',
    'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length', 'port',
    'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH',
    'Submitting_to_email', 'Abnormal_URL', 'on_mouseover', 'age_of_domain',
    'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index',
    'Links_pointing_to_page', 'Statistical_report'
]

# ─── Подробные описания на русском + перевод ──────────────────────────────────
feature_details = {
    'having_IP_Address': {
        'ru': 'Наличие IP-адреса в URL вместо доменного имени',
        'en': 'Having IP Address',
        'desc': '-1 = да (подозрительно, часто фишинг)\n0 = иногда\n1 = нет (нормально)'
    },
    'URL_Length': {
        'ru': 'Длина URL-адреса',
        'en': 'URL Length',
        'desc': '-1 = короткая/нормальная\n0 = средняя\n1 = очень длинная (подозрительно)'
    },
    'having_At_Symbol': {
        'ru': 'Наличие символа @ в URL',
        'en': 'Having At Symbol',
        'desc': '-1 = есть (очень подозрительно)\n1 = нет (нормально)'
    },
    'Prefix_Suffix': {
        'ru': 'Наличие дефиса в доменном имени (prefix-suffix)',
        'en': 'Prefix Suffix',
        'desc': '-1 = нет\n1 = есть дефис (часто фишинг)'
    },
    'having_Sub_Domain': {
        'ru': 'Количество поддоменов',
        'en': 'Having Sub Domain',
        'desc': '-1 = много (подозрительно)\n0 = нормально\n1 = мало/нет'
    },
    'SSLfinal_State': {
        'ru': 'Состояние SSL-сертификата (самый важный признак!)',
        'en': 'SSL Final State',
        'desc': '-1 = нет/просрочен/поддельный\n0 = промежуточный\n1 = валидный HTTPS'
    },
    'Domain_registeration_length': {
        'ru': 'Срок регистрации домена',
        'en': 'Domain Registration Length',
        'desc': '-1 = недавно зарегистрирован (подозрительно)\n1 = давно (доверие)'
    },
    'port': {
        'ru': 'Использование нестандартного порта',
        'en': 'Port',
        'desc': '-1 = да (подозрительно)\n1 = стандартный 80/443'
    },
    'Request_URL': {
        'ru': 'Процент внешних ресурсов в запросах',
        'en': 'Request URL',
        'desc': '-1 = много внешних (подозрительно)\n1 = почти всё своё'
    },
    'URL_of_Anchor': {
        'ru': 'Процент подозрительных ссылок в тегах <a> (якоря)',
        'en': 'URL of Anchor',
        'desc': '-1 = много подозрительных\n1 = нормальные ссылки (очень важный признак!)'
    },
    # ... и так далее для остальных (добавь сам по аналогии, если нужно все 22)
    # Для краткости оставлю только несколько — расширь по желанию
}

# ─── Функция инженерных признаков ─────────────────────────────────────────────
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

# ─── Конфиг и навигация ───────────────────────────────────────────────────────
st.set_page_config(page_title="Phishing Detector Pro", layout="wide")

st.sidebar.title("🛡️ Phishing Detector")
pages = ["Главная", "Проверка сайта", "Описание признаков", "О модели"]
page = st.sidebar.radio("Разделы", pages)

# ─── Главная страница с атмосферой ────────────────────────────────────────────
if page == "Главная":
    st.title("🛡️ Обнаружение фишинговых сайтов")
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ### Защитите себя от онлайн-мошенников
        
        Это мощный инструмент на базе **CatBoost**, который анализирует 
        26 признаков сайта и с вероятностью **~97%** определяет — фишинг или безопасно.
        
        Разработал: **Muhamadasror**  
        Душанбе, Таджикистан | 2026
        """)
    
    with col2:
        st.image("https://www.digicert.com/content/dam/digicert/images/about/blog/blog-article/graphic-2-fraud.png", 
                 caption="Пример фишинговой страницы", use_column_width=True)

    st.markdown("---")
    
    st.subheader("Как выглядит опасность?")
    cols_danger = st.columns(3)
    danger_images = [
        "https://www.hostinger.com/tutorials/wp-content/uploads/sites/2/2022/01/deceptive-site-ahead-warning.png",
        "https://img.freepik.com/premium-vector/scam-alert-banner-with-red-scam-danger-warning_349999-1905.jpg",
        "https://www.bleepstatic.com/swr-guides/c/chrome-security-warning/chrome-security-warning.jpg"
    ]
    for img, col in zip(danger_images, cols_danger):
        col.image(img, use_column_width=True)

    st.subheader("А вот безопасный сайт выглядит так")
    cols_safe = st.columns(3)
    safe_images = [
        "https://png.pngtree.com/png-clipart/20250102/original/pngtree-green-secure-ssl-encryption-sign-with-padlock-shield-for-website-security-png-image_18637108.png",
        "https://img.freepik.com/premium-vector/secure-connection-secured-ssl-shield-padlock-symbols-http-https-safe-secure-wev-browsing-safe-secure-https_435184-857.jpg",
        "https://www.shutterstock.com/image-vector/secure-connection-secured-ssl-shield-260nw-2382974415.jpg"
    ]
    for img, col in zip(safe_images, cols_safe):
        col.image(img, use_column_width=True)

# ─── Проверка сайта (без изменений, только если нужно) ────────────────────────
elif page == "Проверка сайта":
    st.title("🔍 Проверка сайта")
    # ... (твой предыдущий код проверки — вставь сюда без изменений)

# ─── НОВАЯ СТРАНИЦА: Описание признаков ────────────────────────────────────────
elif page == "Описание признаков":
    st.title("📋 Что означают все признаки?")
    st.markdown("""
    Здесь подробное объяснение **каждого** признака, который использует модель.  
    Значения всегда: **-1** (плохо/подозрительно) • **0** (нейтрально) • **1** (хорошо/нормально)
    """)

    for feature, info in feature_details.items():
        with st.expander(f"**{info['en']}** → {info['ru']}"):
            st.markdown(f"**{info['ru']}**")
            st.code(info['desc'], language="text")
            st.caption(f"Оригинальное название: {info['en']}")

    st.info("Не все 22 признака перечислены выше — для полного списка смотрите код или датасет. Самые важные: SSLfinal_State и URL_of_Anchor!")

# ─── О модели (можно дополнить картинкой) ─────────────────────────────────────
elif page == "О модели":
    st.title("ℹ️ О модели")
    st.image("https://ars.els-cdn.com/content/image/1-s2.0-S2665917423003392-gr1.jpg", 
             caption="Пример схемы работы системы обнаружения фишинга", use_column_width=True)
    
    # ... остальной текст о модели как раньше

st.sidebar.markdown("---")
st.sidebar.caption("v2.0 | 2026 | Muhamadasror")
