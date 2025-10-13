import streamlit as st
import numpy as np

st.set_page_config(page_title="Прогноз осложнений", layout="centered")

# 🏷️ Название
st.title("📊 ПРОГНОЗИРОВАНИЕ СЕРОМЫ И БОЛЕВОГО СИНДРОМА ПОСЛЕ ГЕРНИОПЛАСТИКИ ЛАПАРОСКОПИЧЕСКИМИ МЕТОДАМИ")

# 🔽 ВВОД ДАННЫХ ОДИН ДЛЯ ОБЕИХ МОДЕЛЕЙ
st.header("📝 Ввод данных пациента")

st.info("Укажите клинические данные для оценки вероятности осложнений после герниопластики.")

# Ввод
BMI = st.number_input("Индекс массы тела (ИМТ)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

surgery_type = st.selectbox("Тип вмешательства", ["eTEP", "TAPP"])
hernia_history = st.checkbox("Грыжесечение в анамнезе")
asa_class = st.selectbox("ASA класс", ["I–II", "III (высокий риск)"])

# Кодировка переменных
x_surgery = 1 if surgery_type == "TAPP" else 0
x_hernia = 1 if hernia_history else 0
x_asa = 1 if asa_class == "III (высокий риск)" else 0
x_bmi = BMI

# ======================================================================================
# 📌 МОДЕЛЬ 1 — СЕРОМА
st.header("💧 Риск серомы в послеоперационном периоде")

B0_seroma = 1.669
B_surgery_seroma = -0.975
B_hernia_seroma = 2.018
B_asa_seroma = -1.418
B_bmi_seroma = -0.007

logit_seroma = (
    B0_seroma +
    B_surgery_seroma * x_surgery +
    B_hernia_seroma * x_hernia +
    B_asa_seroma * x_asa +
    B_bmi_seroma * x_bmi
)

prob_seroma = 1 / (1 + np.exp(-logit_seroma))
pct_seroma = min(max(prob_seroma * 100, 0), 100)

st.success(f"🔹 Вероятность серомы: **{pct_seroma:.2f}%**")
st.progress(min(prob_seroma, 1.0))

if prob_seroma < 0.1:
    st.markdown("🟢 Низкий риск серомы")
elif prob_seroma < 0.5:
    st.markdown("🟡 Умеренный риск серомы")
else:
    st.markdown("🔴 Высокий риск серомы")

# ======================================================================================
# 📌 МОДЕЛЬ 2 — БОЛЕВОЙ СИНДРОМ
st.header("💥 Риск болевого синдрома в послеоперационном периоде")

B0_pain = 1.669
B_surgery_pain = -0.0975
B_hernia_pain = 2.018
B_asa_pain = -1.418
B_bmi_pain = -0.007

logit_pain = (
    B0_pain +
    B_surgery_pain * x_surgery +
    B_hernia_pain * x_hernia +
    B_asa_pain * x_asa +
    B_bmi_pain * x_bmi
)

prob_pain = 1 / (1 + np.exp(-logit_pain))
pct_pain = min(max(prob_pain * 100, 0), 100)

st.success(f"🔹 Вероятность болевого синдрома: **{pct_pain:.2f}%**")
st.progress(min(prob_pain, 1.0))

if prob_pain < 0.1:
    st.markdown("🟢 Низкий болевой риск")
elif prob_pain < 0.5:
    st.markdown("🟡 Умеренный болевой риск")
else:
    st.markdown("🔴 Высокий болевой риск")
