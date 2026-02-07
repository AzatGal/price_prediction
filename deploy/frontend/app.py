import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Конфигурация
API_URL = "http://localhost:8000"

# Инициализация состояния сессии
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user' not in st.session_state:
    st.session_state.user = None
if 'config' not in st.session_state:
    st.session_state.config = None

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .price-display {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-interval {
        background: rgba(255,255,255,0.2);
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def get_headers():
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}


def login_page():
    st.markdown('<div class="main-header">🏠 Real Estate Predictor</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["Вход", "Регистрация"])

        with tab1:
            with st.form("login_form"):
                username = st.text_input("Имя пользователя")
                password = st.text_input("Пароль", type="password")
                submit = st.form_submit_button("Войти", use_container_width=True)

                if submit:
                    try:
                        response = requests.post(f"{API_URL}/auth/login",
                                                 json={"username": username, "password": password})
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.token = data['access_token']
                            st.session_state.user = username
                            st.success("Успешный вход!")
                            st.rerun()
                        else:
                            st.error("Неверные учетные данные")
                    except Exception as e:
                        st.error(f"Ошибка подключения: {e}")

        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Имя пользователя")
                new_password = st.text_input("Пароль", type="password")
                submit_reg = st.form_submit_button("Зарегистрироваться", use_container_width=True)

                if submit_reg:
                    try:
                        response = requests.post(f"{API_URL}/auth/register",
                                                 json={"username": new_username, "password": new_password})
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.token = data['access_token']
                            st.session_state.user = new_username
                            st.success("Регистрация успешна!")
                            st.rerun()
                        else:
                            st.error("Ошибка регистрации")
                    except Exception as e:
                        st.error(f"Ошибка подключения: {e}")


def load_config():
    if st.session_state.config is None:
        try:
            response = requests.get(f"{API_URL}/config")
            st.session_state.config = response.json()
        except:
            st.error("Не удалось загрузить конфигурацию")


def render_feature_input(name, config):
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"**{config['label']}**")

    with col2:
        if config['type'] == 'categorical':
            return st.selectbox(
                f"select_{name}",
                options=config['options'],
                index=config['options'].index(config.get('default', config['options'][0])),
                label_visibility="collapsed"
            )
        elif config['type'] == 'int':
            return st.number_input(
                f"num_{name}",
                min_value=config['min'],
                max_value=config['max'],
                value=config.get('default', config['min']),
                step=1,
                label_visibility="collapsed"
            )
        else:  # float
            return st.number_input(
                f"num_{name}",
                min_value=float(config['min']),
                max_value=float(config['max']),
                value=float(config.get('default', config['min'])),
                step=0.1,
                label_visibility="collapsed"
            )


def prediction_page():
    load_config()

    st.markdown(f'<div class="main-header">Привет, {st.session_state.user}! 👋</div>', unsafe_allow_html=True)

    # Боковая панель
    with st.sidebar:
        st.header("Навигация")
        page = st.radio("", ["Новое предсказание", "История запросов"])

        st.divider()
        if st.button("Выйти", use_container_width=True):
            st.session_state.token = None
            st.session_state.user = None
            st.rerun()

    if page == "Новое предсказание":
        st.subheader("🔮 Новое предсказание стоимости")

        with st.container():
            st.markdown("### Характеристики объекта")

            features = {}
            cols = st.columns(2)
            config_items = list(st.session_state.config.items())

            for i, (name, cfg) in enumerate(config_items):
                with cols[i % 2]:
                    features[name] = render_feature_input(name, cfg)

            st.divider()

            col_pred, col_clear = st.columns(2)
            with col_pred:
                predict_btn = st.button("🚀 Рассчитать стоимость", type="primary", use_container_width=True)

            with col_clear:
                if st.button("🔄 Очистить", use_container_width=True):
                    st.rerun()

            if predict_btn:
                with st.spinner("Анализируем данные..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/predict",
                            json={"features": features},
                            headers=get_headers()
                        )

                        if response.status_code == 200:
                            result = response.json()

                            # Карточка с результатом
                            st.markdown("""
                            <div class="prediction-card">
                                <h3 style="text-align: center; margin-bottom: 1rem;">📊 Результат оценки</h3>
                            """, unsafe_allow_html=True)

                            price = result['predicted_price']
                            price_formatted = f"{price:,.0f} ₽".replace(",", " ")

                            st.markdown(f"""
                                <div class="price-display">{price_formatted}</div>
                                <div style="text-align: center; opacity: 0.9;">
                                    ~ {result['price_per_m2']:,.0f} ₽/м²
                                </div>
                            """, unsafe_allow_html=True)

                            # Доверительный интервал
                            low = f"{result['confidence_low']:,.0f} ₽".replace(",", " ")
                            high = f"{result['confidence_high']:,.0f} ₽".replace(",", " ")

                            st.markdown(f"""
                                <div class="confidence-interval">
                                    <div style="text-align: center; font-size: 0.9rem; margin-bottom: 0.5rem;">
                                        Доверительный интервал ({result['uncertainty_percent']}% погрешность)
                                    </div>
                                    <div style="display: flex; justify-content: space-between; font-weight: bold;">
                                        <span>{low}</span>
                                        <span>{high}</span>
                                    </div>
                                    <div style="background: rgba(255,255,255,0.3); height: 6px; border-radius: 3px; margin-top: 0.5rem;">
                                        <div style="background: white; height: 100%; width: 60%; margin-left: 20%; border-radius: 3px;"></div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Визуализация факторов
                            st.markdown("### 📈 Анализ факторов")

                            factors = {
                                'Площадь': features['area'] * 100000,
                                'Этаж': features['floor'] * 50000,
                                'Год постройки': (features['year_built'] - 1900) * 20000,
                                'Расстояние до метро': (20 - features['metro_distance']) * 30000
                            }

                            fig = go.Figure(go.Bar(
                                x=list(factors.keys()),
                                y=list(factors.values()),
                                marker_color=['#1E88E5', '#43A047', '#FB8C00', '#E53935'],
                                text=[f"{v:,.0f}" for v in factors.values()],
                                textposition='auto',
                            ))
                            fig.update_layout(
                                title="Вклад факторов в стоимость",
                                xaxis_title="Фактор",
                                yaxis_title="Влияние на цену (₽)",
                                template="plotly_white",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        else:
                            st.error(f"Ошибка: {response.text}")
                    except Exception as e:
                        st.error(f"Ошибка соединения: {e}")

    else:  # История запросов
        st.subheader("📚 История запросов")

        # Фильтры
        with st.expander("🔍 Фильтры", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                date_from = st.date_input("С даты", datetime.now() - timedelta(days=30))
                date_to = st.date_input("По дату", datetime.now())

            with col2:
                district_filter = st.multiselect(
                    "Район",
                    options=["Центральный", "Северный", "Южный", "Западный", "Восточный"]
                )
                house_type_filter = st.multiselect(
                    "Тип дома",
                    options=["Кирпичный", "Панельный", "Монолитный", "Блочный", "Деревянный"]
                )

            with col3:
                min_price = st.number_input("Мин. цена", min_value=0, value=0, step=100000)
                max_price = st.number_input("Макс. цена", min_value=0, value=100000000, step=100000)

            filter_btn = st.button("Применить фильтры", use_container_width=True)

        # Загрузка данных
        try:
            params = {}
            if filter_btn:
                params = {
                    'date_from': date_from.isoformat(),
                    'date_to': date_to.isoformat(),
                    'min_price': min_price if min_price > 0 else None,
                    'max_price': max_price if max_price < 100000000 else None
                }
                if district_filter:
                    params['district'] = district_filter[0]
                if house_type_filter:
                    params['house_type'] = house_type_filter[0]

                # Убираем None значения
                params = {k: v for k, v in params.items() if v is not None}

            response = requests.get(
                f"{API_URL}/predictions",
                params=params,
                headers=get_headers()
            )

            if response.status_code == 200:
                predictions = response.json()

                if not predictions:
                    st.info("История пуста. Сделайте первое предсказание!")
                else:
                    # Статистика
                    total = len(predictions)
                    avg_price = sum(p['predicted_price'] for p in predictions) / total

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Всего объектов", total)
                    col2.metric("Средняя цена", f"{avg_price:,.0f} ₽".replace(",", " "))
                    col3.metric("Диапазон",
                                f"{min(p['predicted_price'] for p in predictions):,.0f} - {max(p['predicted_price'] for p in predictions):,.0f} ₽")

                    # График тренда
                    df = pd.DataFrame(predictions)
                    df['created_at'] = pd.to_datetime(df['created_at'])

                    fig = px.line(df, x='created_at', y='predicted_price',
                                  title="Динамика цен",
                                  labels={'predicted_price': 'Цена (₽)', 'created_at': 'Дата'})
                    fig.update_traces(mode='markers+lines')
                    st.plotly_chart(fig, use_container_width=True)

                    # Таблица с редактированием
                    st.markdown("### Список объектов")

                    for pred in predictions:
                        with st.container():
                            col_info, col_price, col_actions = st.columns([3, 2, 1])

                            features = pred['features']

                            with col_info:
                                st.markdown(f"""
                                **{features.get('rooms', '?')} комн., {features.get('area', '?')} м²**  
                                {features.get('district', '?')} район, {features.get('house_type', '?')} дом  
                                {features.get('floor', '?')}/{features.get('total_floors', '?')} этаж, {features.get('year_built', '?')} г.
                                """)

                            with col_price:
                                price = pred['predicted_price']
                                st.markdown(f"""
                                <div style="text-align: right;">
                                    <div style="font-size: 1.3rem; font-weight: bold; color: #1E88E5;">
                                        {price:,.0f} ₽
                                    </div>
                                    <div style="font-size: 0.8rem; color: gray;">
                                        {pred['created_at'][:10]}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                            with col_actions:
                                if st.button("🗑️", key=f"del_{pred['id']}", help="Удалить"):
                                    try:
                                        del_resp = requests.delete(
                                            f"{API_URL}/predictions/{pred['id']}",
                                            headers=get_headers()
                                        )
                                        if del_resp.status_code == 200:
                                            st.success("Удалено!")
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Ошибка: {e}")

                                if st.button("🔄", key=f"recalc_{pred['id']}", help="Пересчитать"):
                                    try:
                                        recalc_resp = requests.post(
                                            f"{API_URL}/predictions/{pred['id']}/recalculate",
                                            headers=get_headers()
                                        )
                                        if recalc_resp.status_code == 200:
                                            st.success("Обновлено!")
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Ошибка: {e}")

                            st.divider()
            else:
                st.error("Не удалось загрузить историю")
        except Exception as e:
            st.error(f"Ошибка: {e}")


def main():
    if st.session_state.token is None:
        login_page()
    else:
        prediction_page()


if __name__ == "__main__":
    main()