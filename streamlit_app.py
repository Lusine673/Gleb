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

    # Таблица 9 — коэффициенты
    # Тип вмешательства: 0 = TAPP, 1 = eTEP
    B0_S = 1.669
    B_SURG_TYPE_S = -0.975       # 1=eTEP, 0=TAPP
    B_PRIOR_HERNIA_S = 2.018     # 1=да, 0=нет
    B_ASA_S = -1.418             # ASA как 1..4
    B_BMI_S = -0.007             # ИМТ (число)

    def predict_seroma(intervention_etep: int, prior_hernia: int, asa: int, bmi: float) -> float:
        z = (
            B0_S
            + B_SURG_TYPE_S * int(intervention_etep)
            + B_PRIOR_HERNIA_S * int(prior_hernia)
            + B_ASA_S * int(asa)
            + B_BMI_S * float(bmi)
        )
        return sigmoid(z)

    col1, col2 = st.columns(2)
    with col1:
        surg_label_s = st.selectbox("Тип вмешательства", options=["TAPP", "eTEP"], key="s_surg")
        intervention_etep_s = 1 if surg_label_s == "eTEP" else 0
        prior_hernia_s = st.checkbox("Грыжесечение в анамнезе", value=False, key="s_ph")

    with col2:
        asa_label_s = st.selectbox("ASA (класс)", options=["I", "II", "III", "IV"], index=1, key="s_asa")
        asa_s = ["I", "II", "III", "IV"].index(asa_label_s) + 1
        bmi_s = st.number_input("ИМТ, кг/м²", min_value=10.0, max_value=70.0, step=0.1, value=26.0, key="s_bmi")

    p_seroma = predict_seroma(
        intervention_etep=intervention_etep_s,
        prior_hernia=1 if prior_hernia_s else 0,
        asa=asa_s,
        bmi=bmi_s
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

# ================== Боль ==================
with tabs[1]:
    st.subheader("Риск болевого синдрома")

    # Таблица 14 — значимые и с тенденцией
    # Тип вмешательства: 0 = TAPP, 1 = eTEP
    B0_P = -62.457
    B_BMI_P = 1.541
    B_ASA_P = 4.034
    B_INTERVENTION_P = 6.063          # 1=eTEP, 0=TAPP
    B_PRIOR_OPERATION_P = -3.389      # 1=да, 0=нет
    B_PRIOR_HERNIA_P = 2.669          # 1=да, 0=нет
    B_HTN_P = 3.196                    # Гипертоническая болезнь (0/1)
    B_DURATION_PER_MIN_P = 0.005      # длительность: коэффициент за 1 минуту

    def predict_pain(bmi: float, asa: int, intervention_etep: int,
                     prior_operation: int, prior_hernia: int, htn: int, duration_min: float) -> float:
        z = (
            B0_P
            + B_BMI_P * float(bmi)
            + B_ASA_P * int(asa)
            + B_INTERVENTION_P * int(intervention_etep)
            + B_PRIOR_OPERATION_P * int(prior_operation)
            + B_PRIOR_HERNIA_P * int(prior_hernia)
            + B_HTN_P * int(htn)
            + B_DURATION_PER_MIN_P * float(duration_min)
        )
        return sigmoid(z)

    c1, c2 = st.columns(2)
    with c1:
        surg_label_p = st.selectbox("Тип вмешательства", options=["TAPP", "eTEP"], key="p_surg")
        intervention_etep_p = 1 if surg_label_p == "eTEP" else 0

        prior_operation = st.checkbox("Оперативные вмешательства в анамнезе", key="p_prevop")
        prior_hernia_p = st.checkbox("Грыжесечение в анамнезе", key="p_prevhernia")
        htn_p = st.checkbox("Гипертоническая болезнь", key="p_htn")

    with c2:
        asa_label_p = st.selectbox("ASA (класс)", options=["I", "II", "III", "IV"], index=1, key="p_asa")
        asa_p = ["I", "II", "III", "IV"].index(asa_label_p) + 1
        bmi_p = st.number_input("ИМТ, кг/м²", min_value=10.0, max_value=70.0, step=0.1, value=26.0, key="p_bmi")
        duration_min = st.number_input("Длительность операции, мин", min_value=10.0, max_value=600.0,
                                       step=5.0, value=90.0, key="p_dur")

    p_pain = predict_pain(
        bmi=bmi_p,
        asa=asa_p,
        intervention_etep=intervention_etep_p,
        prior_operation=1 if prior_operation else 0,
        prior_hernia=1 if prior_hernia_p else 0,
        htn=1 if htn_p else 0,
        duration_min=duration_min
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

st.caption("Шкала риска: <10% — низкий, 10–50% — умеренный, >50% — высокий. "
           "Инструмент предназначен для исследовательских целей.")
