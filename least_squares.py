import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="پروژه ۲: حل دستگاه خطی", layout="wide")

st.title("🔢 حل دستگاه خطی Ax=b با Least Squares")
st.markdown("**مقایسه حداقل مربعات دستی و رگرسیون خطی**")

st.sidebar.header("📂 بارگذاری داده")
data_source = st.sidebar.radio("منبع داده:", ["آپلود فایل CSV", "دیتاست پیش‌فرض"])

df = None

if data_source == "آپلود فایل CSV":
    uploaded_file = st.sidebar.file_uploader("فایل CSV را انتخاب کنید:", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
else:
    
    try:
        df = pd.read_csv("dataset (1).csv")
        st.sidebar.success("✅ دیتاست پیش‌فرض بارگذاری شد")
    except:
        st.sidebar.error("❌ دیتاست پیش‌فرض یافت نشد")

if df is not None:
    st.subheader("📊 نمایش داده‌ها")
    st.dataframe(df.head(10), use_container_width=True)
    st.info(f"تعداد ردیف‌ها: **{len(df)}** | تعداد ستون‌ها: **{len(df.columns)}**")
    
    st.subheader("🔧 انتخاب ستون‌ها")
    
    all_cols = df.columns.tolist()
    default_b = all_cols[-1]
    default_A = all_cols[:-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        a_cols = st.multiselect(
            "ستون‌های ماتریس A (ویژگی‌ها):",
            all_cols,
            default=default_A
        )
    
    with col2:
        b_col = st.selectbox(
            "ستون بردار b (هدف):",
            all_cols,
            index=all_cols.index(default_b)
        )
    
    if len(a_cols) == 0:
        st.warning("⚠️ حداقل یک ستون برای A انتخاب کنید")
    elif b_col in a_cols:
        st.error("❌ ستون b نباید در A باشد")
    else:
        
        A = df[a_cols].values
        b = df[b_col].values
        
        st.success(f"✅ ماتریس A: **{A.shape}** | بردار b: **{b.shape}**")
        
        if st.button("🚀 حل دستگاه و مقایسه", type="primary"):
            
            st.markdown("---")
            st.subheader("📐 روش ۱: Least Squares دستی")
            
            try:
                
                ATA = A.T @ A
                ATb = A.T @ b
                x_ls = np.linalg.solve(ATA, ATb)
                
                b_pred_ls = A @ x_ls
                mse_ls = mean_squared_error(b, b_pred_ls)
                
                st.success("✅ حل شد!")
                st.write("**ضرایب محاسبه‌شده (x):**")
                coef_df_ls = pd.DataFrame({
                    'ویژگی': a_cols,
                    'ضریب': x_ls
                })
                st.dataframe(coef_df_ls, use_container_width=True)
                st.metric("میانگین مربعات خطا (MSE)", f"{mse_ls:.6f}")
                
            except np.linalg.LinAlgError:
                st.error("❌ ماتریس A^T A معکوس‌پذیر نیست! (دستگاه ناسازگار یا ستون‌های وابسته)")
                x_ls = None
                b_pred_ls = None
                mse_ls = None
            
            st.markdown("---")
            st.subheader("📈 روش ۲: رگرسیون خطی (sklearn)")
            
            model = LinearRegression()
            model.fit(A, b)
            b_pred_reg = model.predict(A)
            mse_reg = mean_squared_error(b, b_pred_reg)
            
            st.success("✅ مدل آموزش داده شد!")
            st.write("**ضرایب محاسبه‌شده:**")
            coef_df_reg = pd.DataFrame({
                'ویژگی': a_cols,
                'ضریب': model.coef_
            })
            st.dataframe(coef_df_reg, use_container_width=True)
            st.write(f"**عرض از مبدأ (intercept):** {model.intercept_:.6f}")
            st.metric("میانگین مربعات خطا (MSE)", f"{mse_reg:.6f}")
            
            st.markdown("---")
            st.subheader("🔍 مقایسه نتایج")
            
            if x_ls is not None:
                
                comparison = pd.DataFrame({
                    'روش': ['Least Squares دستی', 'رگرسیون خطی sklearn'],
                    'MSE': [mse_ls, mse_reg]
                })
                st.dataframe(comparison, use_container_width=True)
                
                fig, ax = plt.subplots(1, 2, figsize=(14, 5))
                
                ax[0].scatter(b, b_pred_ls, alpha=0.6, color='blue')
                ax[0].plot([b.min(), b.max()], [b.min(), b.max()], 'r--', lw=2)
                ax[0].set_xlabel('مقدار واقعی b', fontsize=12)
                ax[0].set_ylabel('پیش‌بینی (Least Squares)', fontsize=12)
                ax[0].set_title(f'Least Squares\nMSE = {mse_ls:.4f}', fontsize=14)
                ax[0].grid(True, alpha=0.3)
                
                ax[1].scatter(b, b_pred_reg, alpha=0.6, color='green')
                ax[1].plot([b.min(), b.max()], [b.min(), b.max()], 'r--', lw=2)
                ax[1].set_xlabel('مقدار واقعی b', fontsize=12)
                ax[1].set_ylabel('پیش‌بینی (Regression)', fontsize=12)
                ax[1].set_title(f'Linear Regression\nMSE = {mse_reg:.4f}', fontsize=14)
                ax[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                if len(a_cols) <= 10:  
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    x_pos = np.arange(len(a_cols))
                    width = 0.35
                    
                    ax2.bar(x_pos - width/2, x_ls, width, label='Least Squares', alpha=0.8)
                    ax2.bar(x_pos + width/2, model.coef_, width, label='Regression', alpha=0.8)
                    
                    ax2.set_xlabel('Features', fontsize=12)
                    ax2.set_ylabel('Coefficient Value', fontsize=12)
                    ax2.set_title('مقایسه ضرایب', fontsize=14)
                    ax2.set_xticks(x_pos)
                    ax2.set_xticklabels(a_cols, rotation=45)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    st.pyplot(fig2)
            
            st.markdown("---")
            st.subheader("📝 نتیجه‌گیری")
            
            if x_ls is not None and abs(mse_ls - mse_reg) < 1e-6:
                st.success("✅ **نتایج یکسان**: هر دو روش به یک جواب رسیدند (با دقت عددی بالا)")
            elif x_ls is not None:
                st.info(f"📊 **تفاوت MSE**: {abs(mse_ls - mse_reg):.8f}")
            
            st.write("""
            **توضیح:**
            - **Least Squares دستی**: حل مستقیم با فرمول $x = (A^T A)^{-1} A^T b$
            - **رگرسیون خطی sklearn**: استفاده از الگوریتم‌های بهینه‌شده
            - در حالت ایده‌آل، هر دو روش باید نتایج یکسانی بدهند
            """)

else:
    st.warning("⚠️ لطفاً دیتاست را بارگذاری کنید")

st.markdown("---")
st.caption("پروژه ۲ - حل دستگاه خطی با Least Squares | دانشگاه")
