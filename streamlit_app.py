import math
import streamlit as st

st.set_page_config(page_title="Прогноз осложнений после лапароскопической герниопластики",
                   page_icon="🩺", layout="centered")

# ------------ утилиты ------------
def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

def risk_class(prob: float) -> str:
    if prob < 0.10:
        return "Низкий риск"
    if prob <= 0.50:
        return "Умеренный риск"
    return "Высокий риск"

st.title("Прогноз осложнений после лапароскопической герниопластики")
tabs = st.tabs(["Серома", "Боль"])

# ================== Серома ==================
with tabs[0]:
    st.subheader("Риск серомы")

    # Коэффициенты (табл. 9), без ИМТ.
    # ВАЖНО: коэффициент при eTEP; в UI TAPP=1, eTEP=0 → внутри eTEP = 1 − TAPP.
    B0_S = 1.669
    B_SURG_TYPE_S = -0.975       # умножается на I(eTEP)
    B_PRIOR_HERNIA_S = 2.018     # 1=да, 0=нет
    B_ASA_S = -1.418             # 1..4

    def predict_seroma(e_tep: int, prior_hernia: int, asa: int) -> float:
        z = (
            B0_S
            + B_SURG_TYPE_S * int(e_tep)         # eTEP: 1, TAPP: 0
            + B_PRIOR_HERNIA_S * int(prior_hernia)
            + B_ASA_S * int(asa)
        )
        return sigmoid(z)

    col1, col2 = st.columns(2)
    with col1:
        surg_label_s = st.selectbox("Тип вмешательства", options=["TAPP", "eTEP"], key="s_surg")
        tapp_indicator = 1 if surg_label_s == "TAPP" else 0
        etep_indicator = 1 - tapp_indicator  # перевод в индикатор eTEP для формулы
        prior_hernia_s = st.checkbox("Грыжесечение в анамнезе", value=False, key="s_ph")

    with col2:
        asa_label_s = st.selectbox("ASA (класс)", options=["I", "II", "III", "IV"], index=1, key="s_asa")
        asa_s = ["I", "II", "III", "IV"].index(asa_label_s) + 1

    p_seroma = predict_seroma(
        e_tep=etep_indicator,
        prior_hernia=1 if prior_hernia_s else 0,
        asa=asa_s
    )

    st.write("---")
    c1, c2 = st.columns(2)
    c1.metric("Вероятность серомы", f"{p_seroma*100:.1f}%")
    c2.metric("Класс риска", risk_class(p_seroma))

    if p_seroma < 0.10:
        st.success("Низкий риск")
    elif p_seroma <= 0.50:
        st.warning("Умеренный риск")
    else:
        st.error("Высокий риск")

    st.info(
        "Дисклеймер: инструмент предназначен исключительно для исследовательских и образовательных целей. "
        "Не является медицинским изделием. Внешняя клиническая валидация и полная оценка "
        "дискриминационной способности/калибровки на исходной выборке не проводились."
    )

# ================== Боль ==================
with tabs[1]:
    st.subheader("Риск болевого синдрома")

    # Коэффициенты (табл. 14). Длительность операции исключена.
    # ВАЖНО: коэффициент при eTEP; в UI TAPP=1, eTEP=0 → внутри eTEP = 1 − TAPP.
    B0_P = -62.457
    B_BMI_P = 1.541
    B_ASA_P = 4.034
    B_INTERVENTION_E_TEP_P = 6.063   # умножается на I(eTEP)
    B_PRIOR_OPERATION_P = -3.389
    B_PRIOR_HERNIA_P = 2.669
    B_HTN_P = 3.196

    def predict_pain(bmi: float, asa: int, tapp_indicator: int,
                     prior_operation: int, prior_hernia: int, htn: int) -> float:
        e_tep = 1 - int(tapp_indicator)  # преобразуем TAPP=1/eTEP=0 → eTEP индикатор
        z = (
            B0_P
            + B_BMI_P * float(bmi)
            + B_ASA_P * int(asa)
            + B_INTERVENTION_E_TEP_P * e_tep
            + B_PRIOR_OPERATION_P * int(prior_operation)
            + B_PRIOR_HERNIA_P * int(prior_hernia)
            + B_HTN_P * int(htn)
        )
        return sigmoid(z)

    c1, c2 = st.columns(2)
    with c1:
        surg_label_p = st.selectbox("Тип вмешательства", options=["TAPP", "eTEP"], key="p_surg")
        tapp_indicator_p = 1 if surg_label_p == "TAPP" else 0

        prior_operation = st.checkbox("Оперативные вмешательства в анамнезе", key="p_prevop")
        prior_hernia_p = st.checkbox("Грыжесечение в анамнезе", key="p_prevhernia")
        htn_p = st.checkbox("Гипертоническая болезнь", key="p_htn")

    with c2:
        asa_label_p = st.selectbox("ASA (класс)", options=["I", "II", "III", "IV"], index=1, key="p_asa")
        asa_p = ["I", "II", "III", "IV"].index(asa_label_p) + 1
        bmi_p = st.number_input("ИМТ, кг/м²", min_value=10.0, max_value=70.0, step=0.1, value=26.0, key="p_bmi")
        # Длительность операции удалена по вашему решению

    p_pain = predict_pain(
        bmi=bmi_p,
        asa=asa_p,
        tapp_indicator=tapp_indicator_p,
        prior_operation=1 if prior_operation else 0,
        prior_hernia=1 if prior_hernia_p else 0,
        htn=1 if htn_p else 0
    )

    st.write("---")
    cc1, cc2 = st.columns(2)
    cc1.metric("Вероятность боли", f"{p_pain*100:.1f}%")
    cc2.metric("Класс риска", risk_class(p_pain))

    if p_pain < 0.10:
        st.success("Низкий риск")
    elif p_pain <= 0.50:
        st.warning("Умеренный риск")
    else:
        st.error("Высокий риск")

    st.info(
        "Дисклеймер: инструмент предназначен исключительно для исследовательских и образовательных целей. "
        "Не является медицинским изделием. Внешняя клиническая валидация и полная оценка "
        "дискриминационной способности/калибровки на исходной выборке не проводились."
    )
