import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="ML Decision Tree App", layout="centered")

@st.cache_resource
def load_models():
    reg_model = joblib.load('decision_tree_regression_model.pkl')
    clf_model = joblib.load('decision_tree_hr_model.pkl')
    return reg_model, clf_model

reg_model, clf_model = load_models()

st.markdown("""
<style>
    /* ปรับสีแถบแท็บด้านบนให้เป็นสีชมพู */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #db2777 !important;
        border-bottom-color: #db2777 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #db2777 !important;
    }
    
    /* กล่องหัวเว็บสีชมพูไล่เฉด Rose - Pink */
    .header-box {
        background: linear-gradient(135deg, #db2777 0%, #f472b6 100%);
        padding: 30px 20px;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 6px 15px rgba(219, 39, 119, 0.25);
    }
    .header-box h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        color: white;
    }
    .header-box p {
        margin: 8px 0 0 0;
        font-size: 15px;
        opacity: 0.95;
    }
    
    /* กล่องคำแนะนำ HR */
    .advice-box-risk {
        background-color: #fff1f2;
        border-left: 6px solid #e11d48;
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
    }
    .advice-box-safe {
        background-color: #fdf2f8;
        border-left: 6px solid #ec4899;
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
    }
    .highlight-badge-risk {
        background-color: #ffe4e6;
        color: #be123c;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .highlight-badge-safe {
        background-color: #fce7f3;
        color: #be185d;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h1>ระบบประเมินและวิเคราะห์อัจฉริยะ</h1>
    <p>Insurance Cost & HR Attrition Prediction (Decision Tree AI)</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💰 ทำนายเบี้ยประกัน", "👔 ประเมินพนักงานลาออก"])

with tab1:
    st.write("")
    col1, col2 = st.columns([1.1, 1], gap="large")
    
    with col1:
        age = st.number_input("อายุผู้เอาประกัน (ปี)", min_value=18, max_value=100, value=35)
        bmi = st.number_input("ดัชนีมวลกาย (BMI)", min_value=10.0, max_value=60.0, value=28.5, step=0.1)
        children = st.number_input("จำนวนบุตร (คน)", min_value=0, max_value=10, value=1)
        smoker_str = st.selectbox("ประวัติการสูบบุหรี่", ["ไม่สูบ (No)", "สูบ (Yes)"])
        smoker = 1 if "สูบ (Yes)" in smoker_str else 0

    with col2:
        data1 = pd.DataFrame({'age': [age], 'bmi': [bmi], 'children': [children], 'smoker': [smoker]})
        pred_price = reg_model.predict(data1)[0]
        
        max_price = 55000
        # ปรับโทนสีกราฟวงกลมให้เข้ากับธีมชมพู (ชมพูอ่อน -> ชมพูเข้ม -> แดงกุหลาบ)
        ring_color = "#f472b6" if pred_price < 15000 else ("#db2777" if pred_price < 30000 else "#be123c")
        
        fig = go.Figure(go.Pie(
            values=[min(pred_price, max_price), max(0, max_price - pred_price)],
            hole=0.78,
            marker_colors=[ring_color, "#fce7f3"],
            textinfo='none',
            hoverinfo='none',
            sort=False
        ))
        
        fig.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=240,
            annotations=[dict(
                text=f"<b style='font-size:26px; color:#831843;'>${pred_price:,.0f}</b><br><span style='font-size:13px; color:#9d174d;'>ดอลลาร์ / ปี</span>",
                x=0.5, y=0.5, font_size=20, showarrow=False
            )]
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("<p style='text-align: center; color: #9d174d; font-size: 14px; margin-top: -15px;'>ค่าเบี้ยประกันสุขภาพที่คาดว่าจะได้รับจากโมเดล AI</p>", unsafe_allow_html=True)

with tab2:
    st.write("")
    emp_age = st.number_input("อายุพนักงานปัจจุบัน (ปี)", min_value=18, max_value=70, value=30)
    
    c1, c2 = st.columns(2)
    with c1:
        income = st.number_input("เงินเดือน (Monthly Income)", min_value=1000, max_value=30000, value=4000, step=500)
    with c2:
        distance = st.number_input("ระยะทางจากบ้าน (ไมล์)", min_value=1, max_value=50, value=20)
        
    ot_str = st.radio("การทำล่วงเวลา (OverTime)", ["ไม่มี (No)", "มี (Yes)"], horizontal=True)
    ot = 1 if "มี (Yes)" in ot_str else 0

    data2 = pd.DataFrame({'Age': [emp_age], 'MonthlyIncome': [income], 'DistanceFromHome': [distance], 'OverTime': [ot]})
    pred_attrition = clf_model.predict(data2)[0]

    if pred_attrition == 1:
        st.markdown(f"""
        <div class="advice-box-risk">
            <h4 style="margin:0 0 8px 0; color:#9f1239;">ผลการประเมินความเสี่ยงและคำแนะนำสำหรับ HR</h4>
            <p style="margin:0; color:#374151; font-size:15px; line-height:1.6;">
                จากข้อมูลปัจจัยการทำงาน โมเดลคาดการณ์ว่าพนักงานรายนี้มีแนวโน้ม 
                <span class="highlight-badge-risk">🚨 เสี่ยงที่จะลาออก (Resign)</span> 
                แนะนำให้หัวหน้างานพูดคุยสอบถามความพึงพอใจ พิจารณาปรับลดภาระงานล่วงเวลา (OT) หรือทบทวนสวัสดิการเพิ่มเติม
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="advice-box-safe">
            <h4 style="margin:0 0 8px 0; color:#9d174d;">ผลการประเมินความเสี่ยงและคำแนะนำสำหรับ HR</h4>
            <p style="margin:0; color:#374151; font-size:15px; line-height:1.6;">
                จากข้อมูลปัจจัยการทำงาน โมเดลคาดการณ์ว่าพนักงานรายนี้มีแนวโน้ม 
                <span class="highlight-badge-safe">🌸 ทำงานอยู่ต่อ (Stay)</span> 
                อยู่ในเกณฑ์ปกติที่มีความมั่นคง แนะนำให้รักษาระดับสมดุลการทำงาน (Work-Life Balance) และสนับสนุนการเติบโตในสายงานต่อไป
            </p>
        </div>
        """, unsafe_allow_html=True)
