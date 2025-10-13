import math
import streamlit as st

st.set_page_config(page_title="Прогноз серомы после лапароскопической герниопластики",
                   page_icon="🩺", layout="centered")

# -------------------------------------------------
# Коэффициенты модели (Таблица 9)
# Формула: p = 1 / (1 + exp(-(b0 + Σ b_i·x_i)))
# -------------------------------------------------
B0 = 1.669                        # Константа
B_SURG_TYPE = -0.975             # Тип вмешательства: 1 = TAPP, 0 = eTEP
B_PRIOR_HERNIA = 2.018            # Грыжесечение в анамнезе: 1 = да, 0 = нет
B_ASA = -1.418                    # ASA (1–4), как числовая переменная
B_BMI = -0.007                    # ИМТ (число)

# Примечание: в таблице Exp(B) для «Тип вмешательства» = 0.377.
# Если ориентироваться на Exp(B), то b ≈ ln(0.377) = -0.976 (а не -0.0975).
# Я оставил ровно -0.0975 как в вашем скрине. При необходимости замените на -0.976.


def sigmoid(z: float) -> float:
    # Численно устойчивая логистическая функция
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def predict_probability(surg_type_tapp: int, prior_hernia: int, asa: float, bmi: float) -> float:
    # z = b0 + b1*x1 + ...; где x1.. — значения признаков
    z = (
        B0
        + B_SURG_TYPE * int(surg_type_tapp)
        + B_PRIOR_HERNIA * int(prior_hernia)
        + B_ASA * float(asa)
        + B_BMI * float(bmi)
    )
    return sigmoid(z)


def risk_class(prob: float) -> str:
    if prob < 0.10:
        return "Низкий риск"
    if prob <= 0.50:
        return "Умеренный риск"
    return "Высокий риск"


# ---------------- UI ----------------
st.title("Прогноз риска серомы после лапароскопической герниопластики")

col1, col2 = st.columns(2)
with col1:
    surgery = st.selectbox(
        "Тип вмешательства (кодировка: 1 = TAPP, 0 = eTEP)",
        options=["eTEP (0)", "TAPP (1)"],
        index=0
    )
    surg_type_tapp = 1 if "TAPP" in surgery else 0

    prior_hernia = st.checkbox("Грыжесечение в анамнезе (1 = да)", value=False)

with col2:
    asa = st.number_input("ASA (1–4)", min_value=1.0, max_value=4.0, step=1.0, value=2.0, format="%.0f")
    bmi = st.number_input("ИМТ, кг/м²", min_value=10.0, max_value=70.0, step=0.1, value=26.0)

# Автопересчёт без кнопки
p = predict_probability(
    surg_type_tapp=surg_type_tapp,
    prior_hernia=1 if prior_hernia else 0,
    asa=asa,
    bmi=bmi
)

st.write("---")
c1, c2 = st.columns(2)
c1.metric("Вероятность серомы", f"{p*100:.1f}%")
cls = risk_class(p)
c2.metric("Класс риска", cls)

# Цветовая подсветка статуса
if p < 0.10:
    st.success("Низкий риск (< 10%)")
elif p <= 0.50:
    st.warning("Умеренный риск (10–50%)")
else:
    st.error("Высокий риск")




