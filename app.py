import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# Загрузка модели
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model('phishing_detector_catboost.cbm')
    return model

model = load_model()

# Список базовых признаков (22 фичи после очистки)
base_features = [
    'having_IP_Address', 'URL_Length', 'having_At_Symbol', 'Prefix_Suffix',
    'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length', 'port',
    'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH',
    'Submitting_to_email', 'Abnormal_URL', 'on_mouseover', 'age_of_domain',
    'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index',
    'Links_pointing_to_page', 'Statistical_report'
]

# Описания признаков (для объяснения на странице)
feature_descriptions = {
    'having_IP_Address': 'Есть ли IP-адрес в URL? (-1: да, подозрительно; 1: нет, нормально)',
    'URL_Length': 'Длина URL (-1: нормальная; 0: средняя; 1: очень длинная, подозрительно)',
    'having_At_Symbol': 'Есть ли символ @ в URL? (-1: да, подозрительно; 1: нет)',
    'Prefix_Suffix': 'Есть ли дефис в домене? (-1: нет; 1: да, подозрительно)',
    'having_Sub_Domain': 'Количество поддоменов (-1: много; 0: средне; 1: мало)',
    'SSLfinal_State': 'Статус SSL-сертификата (-1: нет или подозрительный; 0: промежуточный; 1: валидный)',
    'Domain_registeration_length': 'Срок регистрации домена (-1: короткий, подозрительно; 1: длинный)',
    'port': 'Используется ли нестандартный порт? (-1: да; 1: нет)',
    'Request_URL': 'Процент внешних ресурсов в URL (-1: много; 0: средне; 1: мало)',
    'URL_of_Anchor': 'Процент подозрительных ссылок в якорях (-1: много; 0: средне; 1: мало)',
    'Links_in_tags': 'Процент подозрительных ссылок в тегах (-1: много; 0: средне; 1: мало)',
    'SFH': 'Server Form Handler (-1: пустой или внешний; 0: about:blank; 1: тот же домен)',
    'Submitting_to_email': 'Формы отправляют на email? (-1: да; 1: нет)',
    'Abnormal_URL': 'Аномалии в URL (-1: да; 1: нет)',
    'on_mouseover': 'Изменение status bar при наведении? (-1: да; 1: нет)',
    'age_of_domain': 'Возраст домена (-1: молодой; 1: старый)',
    'DNSRecord': 'Есть ли DNS-запись? (-1: нет; 1: да)',
    'web_traffic': 'Трафик сайта (-1: низкий; 0: средний; 1: высокий)',
    'Page_Rank': 'PageRank сайта (-1: низкий; 1: высокий)',
    'Google_Index': 'Индексация в Google (-1: нет; 1: да)',
    'Links_pointing_to_page': 'Количество внешних ссылок на страницу (-1: мало; 0: средне; 1: много)',
    'Statistical_report': 'Статистические отчёты о фишинге (-1: да; 1: нет)'
}

# Функция для добавления новых признаков
def add_engineered_features(df):
    df['total_red_flags'] = (df == -1).sum(axis=1)
    df['ssl_anchor_interaction'] = df['SSLfinal_State'] * df['URL_of_Anchor']
    df['no_ssl_short_reg'] = ((df['SSLfinal_State'] == -1) & (df['Domain_registeration_length'] == -1)).astype(int)
    df['subdomain_prefix'] = df['having_Sub_Domain'] * df['Prefix_Suffix']
    return df

# Боковая панель для навигации
st.sidebar.title("Навигация")
pages = ["Главная", "Проверка сайта", "О модели"]
page = st.sidebar.selectbox("Выберите страницу", pages)

# Страница 1: Главная
if page == "Главная":
    st.title("Phishing Detector")
    st.markdown("""
    ### Добро пожаловать в Phishing Detector!
    
    Это веб-приложение для обнаружения фишинговых сайтов. Оно использует машинное обучение (модель CatBoost) для анализа признаков сайта и предсказания, является ли он фишинговым.
    
    **Разработчик:** Muhamadasror (из Душанбе, Таджикистан).  
    Это мой проект по ML для выявления онлайн-мошенничества. Я использовал датасет с 11k+ примерами и доработал модель для высокой точности.
    
    **Как использовать:**
    - Перейдите в "Проверка сайта" для ввода признаков и проверки.
    - В "О модели" узнайте детали о работе модели и её метриках.
    
    Приложение построено на Streamlit и CatBoost. Если у вас есть вопросы — пишите в комментариях!
    """)
    st.image("https://img.icons8.com/color/96/000000/shield.png", width=100)  # Иконка щита для красоты

# Страница 2: Проверка сайта
elif page == "Проверка сайта":
    st.title("Проверка сайта на фишинг")
    st.markdown("""
    ### Инструкция
    Чтобы проверить сайт, введите значения для каждого признака ниже.  
    **Что значат значения?**  
    - **-1**: Подозрительное или негативное значение (часто указывает на фишинг).  
    - **0**: Нейтральное или промежуточное.  
    - **1**: Положительное или нормальное (указывает на легитимный сайт).  
    
    Анализируйте сайт вручную (используйте инструменты вроде URL parsers или инспектор браузера) и выбирайте подходящие значения.  
    После ввода нажмите "Проверить сайт".
    """)

    # Ввод признаков с помощью selectbox (для красоты и удобства)
    input_data = {}
    cols = st.columns(2)  # Два столбца для компактности
    for i, feature in enumerate(base_features):
        with cols[i % 2]:
            input_data[feature] = st.selectbox(
                f"{feature} ({feature_descriptions[feature].split('?')[0]})",
                options=[-1, 0, 1],
                help=feature_descriptions[feature]
            )

    if st.button("Проверить сайт", key="predict_button"):
        # Создание DataFrame из ввода
        input_df = pd.DataFrame([input_data])

        # Добавление новых признаков
        enhanced_df = add_engineered_features(input_df)

        # Предсказание
        pred = model.predict(enhanced_df)[0]
        proba = model.predict_proba(enhanced_df)[0][1] * 100  # Вероятность фишинга в %

        # Вывод результата с цветом
        if pred == 1:  # Фишинг
            st.error(f"⚠️ Сайт опасен! Вероятность фишинга: {proba:.2f}%")
            st.markdown("**Рекомендация:** Не вводите личные данные на этом сайте. Проверьте URL и сертификаты.")
        else:  # Легитимный
            st.success(f"✅ Сайт выглядит безопасным. Вероятность фишинга: {proba:.2f}%")
            st.markdown("**Рекомендация:** Всё равно будьте осторожны с подозрительными сайтами.")

# Страница 3: О модели
elif page == "О модели":
    st.title("О модели")
    st.markdown("""
    ### Как работает модель?
    Модель основана на **CatBoost** (градиентный бустинг для категориальных данных).  
    - **Датасет:** 11,055 примеров с 30+ признаками (после очистки — 22 базовых + 4 новых).  
    - **Целевая переменная:** Фишинг (1) vs. Легитимный (0).  
    - **Новые признаки:**  
      - total_red_flags: Количество подозрительных значений (-1).  
      - ssl_anchor_interaction: Взаимодействие SSL и Anchor ссылок.  
      - no_ssl_short_reg: Нет SSL + короткая регистрация домена.  
      - subdomain_prefix: Много субдоменов + дефис в домене.  
    - **Обучение:** 1500 итераций, early stopping, фокус на recall для минимизации false negatives.  
    - **Топ-признаки:** SSLfinal_State, URL_of_Anchor, Prefix_Suffix и другие (см. ниже).
    
    ### Метрики модели (на тестовой выборке)
    - **ROC-AUC:** 0.995 (отлично!).  
    - **Accuracy:** 97%.  
    - **Recall (для фишинга):** 96% (стандартный порог) / 95% (кастомный порог для высокого recall).  
    - **Precision (для фишинга):** 96%.  
    - **Confusion Matrix (стандарт):**  
      [[1193, 38],  
       [37, 943]]  
    - **Confusion Matrix (кастомный порог):**  
      [[1204, 27],  
       [49, 931]]  
    
    Модель фокусируется на минимизации пропусков фишинга (высокий recall), даже если это чуть снижает precision.
    
    **Топ-15 важных признаков:**
    """)
    # Таблица топ-фич из вывода
    top_fi = pd.DataFrame({
        'Признак': ['URL_of_Anchor', 'SSLfinal_State', 'Prefix_Suffix', 'SFH', 'Links_in_tags', 'web_traffic', 'Links_pointing_to_page', 'total_red_flags', 'having_IP_Address', 'having_Sub_Domain', 'DNSRecord', 'Request_URL', 'subdomain_prefix', 'ssl_anchor_interaction', 'Google_Index'],
        'Важность': [19.89, 13.56, 7.59, 7.14, 6.46, 6.45, 4.97, 4.55, 3.91, 3.90, 3.34, 2.40, 2.32, 2.28, 2.12]
    })
    st.table(top_fi)

    st.markdown("""
    Модель сохранена как .cbm и загружается быстро. Если нужно доработать — добавьте автоматическое извлечение фич из URL!
    """)
