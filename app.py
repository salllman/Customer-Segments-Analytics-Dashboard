import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from sqlalchemy import create_engine, text
from io import StringIO

# ------------------------------
# MySQL Connection Setup
# ------------------------------
from urllib.parse import quote_plus
from sqlalchemy import create_engine
import streamlit as st

def get_db_engine():
    mysql_secrets = st.secrets["mysql"]
    encoded_password = quote_plus(mysql_secrets["password"])
    connection_string = (
        f"mysql+mysqlconnector://{mysql_secrets['user']}:{encoded_password}"
        f"@{mysql_secrets['host']}:{mysql_secrets['port']}/{mysql_secrets['database']}"
    )
    engine = create_engine(connection_string, echo=False)
    return engine

engine = get_db_engine()

# Ensure the datasets table exists
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id INT AUTO_INCREMENT PRIMARY KEY,
            dataset_name VARCHAR(255),
            upload_time DATETIME,
            data LONGTEXT
        )
    """))
    conn.commit()

# ------------------------------
# PAGE CONFIG & SESSION STATE
# ------------------------------
st.set_page_config(page_title="Retail Analytics Dashboard", layout="wide")

if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None
if 'rfm' not in st.session_state:
    st.session_state.rfm = None
if 'forecast' not in st.session_state:
    st.session_state.forecast = None
if 'new_product_proj' not in st.session_state:
    st.session_state.new_product_proj = None

# ------------------------------
# UTILITY FUNCTIONS
# ------------------------------
def auto_map_columns(columns):
    mapping = {}
    date_candidates = [col for col in columns if col.lower() in ['purchasedate', 'date', 'orderdate']]
    mapping['Date'] = date_candidates[0] if date_candidates else None

    cust_candidates = [col for col in columns if 'customer' in col.lower()]
    mapping['CustomerID'] = cust_candidates[0] if cust_candidates else None

    amt_candidates = [col for col in columns if any(keyword in col.lower() for keyword in ['amount', 'spent', 'spend', 'price'])]
    mapping['AmountSpent'] = amt_candidates[0] if amt_candidates else None

    return mapping if None not in mapping.values() else None

@st.cache_data
def process_data(uploaded_file, file_type):
    try:
        if file_type == 'csv':
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        mapping = auto_map_columns(df.columns.tolist())
        if mapping is None:
            st.error("Mapping error: Could not automatically detect required columns (Date, CustomerID, AmountSpent).")
            return None
        df = df.rename(columns={
            mapping['Date']: 'Date',
            mapping['CustomerID']: 'CustomerID',
            mapping['AmountSpent']: 'AmountSpent'
        })
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        return df
    except Exception as e:
        st.error(f"Data processing error: {e}")
        return None

def perform_simple_rfm_segmentation(df):
    current_date = datetime.now()
    rfm = df.groupby('CustomerID').agg({
        'Date': lambda x: (current_date - x.max()).days,
        'CustomerID': 'count',
        'AmountSpent': 'sum'
    }).rename(columns={
        'Date': 'Recency',
        'CustomerID': 'Frequency',
        'AmountSpent': 'Monetary'
    }).reset_index()
    
    r_quantiles = rfm['Recency'].quantile([0.25, 0.5, 0.75]).to_dict()
    f_quantiles = rfm['Frequency'].quantile([0.25, 0.5, 0.75]).to_dict()
    m_quantiles = rfm['Monetary'].quantile([0.25, 0.5, 0.75]).to_dict()
    
    def segment_customer(row):
        if row['Recency'] <= r_quantiles[0.25] and row['Frequency'] >= f_quantiles[0.75] and row['Monetary'] >= m_quantiles[0.75]:
            return "Loyal"
        elif row['Recency'] >= r_quantiles[0.75] and row['Frequency'] <= f_quantiles[0.25] and row['Monetary'] <= m_quantiles[0.25]:
            return "At Risk"
        elif row['Monetary'] < m_quantiles[0.50] and row['Frequency'] >= f_quantiles[0.50]:
            return "Discount Shoppers"
        elif row['Frequency'] >= f_quantiles[0.50]:
            return "Repeat"
        else:
            return "Others"
    
    rfm['Segment'] = rfm.apply(segment_customer, axis=1)
    return rfm

def calculate_churn_repeat(df):
    try:
        df['Month'] = df['Date'].dt.to_period('M').astype(str)
        monthly_customers = df.groupby('Month')['CustomerID'].apply(set).reset_index(name='Customers')
        monthly_customers['PrevCustomers'] = monthly_customers['Customers'].shift(1)
        monthly_customers['RepeatRate'] = [
            len(curr & prev) / len(prev) * 100 if prev else 0
            for curr, prev in zip(monthly_customers['Customers'], monthly_customers['PrevCustomers'])
        ]
        monthly_customers['ChurnRate'] = [
            len(prev - curr) / len(prev) * 100 if prev else 0
            for curr, prev in zip(monthly_customers['Customers'], monthly_customers['PrevCustomers'])
        ]
        monthly_sales = df.groupby('Month').agg({
            'AmountSpent': 'sum',
            'CustomerID': 'nunique'
        }).rename(columns={'AmountSpent': 'TotalSales', 'CustomerID': 'UniqueCustomers'})
        return pd.merge(monthly_customers, monthly_sales, on='Month')
    except Exception as e:
        st.error(f"Churn Analysis Error: {e}")
        return pd.DataFrame()

def generate_sales_insights(monthly_data):
    insights = []
    try:
        max_sales = monthly_data.loc[monthly_data['TotalSales'].idxmax()]
        min_sales = monthly_data.loc[monthly_data['TotalSales'].idxmin()]
        avg_sales = monthly_data['TotalSales'].mean()
        insights.append(f"📈 Peak Sales: {max_sales['Month']} (${max_sales['TotalSales']:,.2f})")
        insights.append(f"📉 Lowest Sales: {min_sales['Month']} (${min_sales['TotalSales']:,.2f})")
        insights.append(f"💰 Average Monthly Sales: ${avg_sales:,.2f}")
        avg_churn = monthly_data['ChurnRate'].mean()
        avg_repeat = monthly_data['RepeatRate'].mean()
        last_churn = monthly_data['ChurnRate'].iloc[-1]
        last_repeat = monthly_data['RepeatRate'].iloc[-1]
        insights.append(f"⚡ Average Churn Rate: {avg_churn:.1f}%")
        insights.append(f"🔄 Average Repeat Rate: {avg_repeat:.1f}%")
        if last_churn > avg_churn * 1.15:
            insights.append(f"🚨 Recent Churn Spike: {last_churn:.1f}%")
        if last_repeat < avg_repeat * 0.85:
            insights.append(f"⚠️ Recent Repeat Rate Drop: {last_repeat:.1f}%")
    except Exception:
        insights.append("🔍 Insights unavailable")
    return insights

@st.cache_data
def forecast_sales_prophet(df, periods=12):
    try:
        sales_df = df.groupby(pd.Grouper(key='Date', freq='M'))['AmountSpent'].sum().reset_index()
        sales_df.rename(columns={'Date': 'ds', 'AmountSpent': 'y'}, inplace=True)
        model = Prophet()
        model.fit(sales_df)
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    except Exception as e:
        st.error(f"Forecasting error: {e}")
        return pd.DataFrame()

def generate_forecast_insights(forecast_df):
    insights = []
    if not forecast_df.empty:
        start_val = forecast_df['yhat'].iloc[0]
        end_val = forecast_df['yhat'].iloc[-1]
        growth = ((end_val - start_val) / start_val) * 100
        start_date = forecast_df['ds'].iloc[0].date()
        end_date = forecast_df['ds'].iloc[-1].date()
        insights.append(f"Projected growth from {start_date} to {end_date}: **{growth:.1f}%**")
        avg_lower = forecast_df['yhat_lower'].mean()
        avg_upper = forecast_df['yhat_upper'].mean()
        conf_range = avg_upper - avg_lower
        insights.append(f"Average confidence interval range: **±{conf_range/2:,.0f}** units")
        trend_direction = "upward" if growth > 0 else "downward"
        insights.append(f"The overall sales trend is **{trend_direction}** based on historical patterns.")
    else:
        insights.append("No forecast data available.")
    return insights

def calculate_new_product_projections(rfm_data, filtered_data, new_price, discount, forecast_months, growth_rate):
    if rfm_data.empty or filtered_data.empty or new_price <= 0 or forecast_months < 1:
        return pd.DataFrame(), pd.DataFrame()
    
    effective_price = new_price * (1 - discount/100)
    hist_end = filtered_data['Date'].max()
    historical_months = ((hist_end - filtered_data['Date'].min()).days // 30) + 1

    if 'Monetary' not in rfm_data.columns:
        st.warning("RFM data missing 'Monetary' column; cannot compute new product projections.")
        return pd.DataFrame(), pd.DataFrame()

    segment_analysis = rfm_data.groupby('Segment').agg(
        CustomerCount=('CustomerID', 'nunique'),
        AvgSpending=('Monetary', 'mean')
    ).reset_index()
    
    segment_analysis['MonthlyUnitsPerCustomer'] = (segment_analysis['AvgSpending'] / effective_price / historical_months).round(1)
    segment_analysis['MonthlyRevenuePerCustomer'] = segment_analysis['MonthlyUnitsPerCustomer'] * effective_price
    
    total_units = (segment_analysis['MonthlyUnitsPerCustomer'] * segment_analysis['CustomerCount']).sum()
    total_revenue = (segment_analysis['MonthlyRevenuePerCustomer'] * segment_analysis['CustomerCount']).sum()
    
    projections = []
    current_date = hist_end + pd.DateOffset(months=1)
    for month in range(forecast_months):
        growth_factor = (1 + growth_rate/100) ** month
        projections.append({
            'Month': current_date.strftime('%Y-%m'),
            'ProjectedUnits': total_units * growth_factor,
            'ProjectedRevenue': total_revenue * growth_factor,
            'GrowthFactor': growth_factor
        })
        current_date += pd.DateOffset(months=1)
    
    return pd.DataFrame(projections), segment_analysis

def generate_business_recommendations(rfm_data, monthly_data):
    recommendations = []
    if rfm_data is not None and not rfm_data.empty:
        seg_counts = rfm_data['Segment'].value_counts().to_dict()
        total_customers = rfm_data['CustomerID'].nunique()
        loyal = seg_counts.get('Loyal', 0)
        repeat = seg_counts.get('Repeat', 0)
        at_risk = seg_counts.get('At Risk', 0)
        discount = seg_counts.get('Discount Shoppers', 0)
        others = seg_counts.get('Others', 0)

        if loyal / total_customers > 0.20:
            recommendations.append("A large portion of customers are Loyal. Consider exclusive loyalty programs or VIP perks to maintain their satisfaction.")
        else:
            recommendations.append("Loyal customers are few. Consider strengthening retention via personalized rewards and early product access.")

        if at_risk / total_customers > 0.15:
            recommendations.append("High number of At Risk customers. Run reactivation campaigns with targeted offers to win them back.")
        else:
            recommendations.append("At Risk customers are lower in number; keep them engaged with timely reminders and surveys.")

        if discount / total_customers > 0.25:
            recommendations.append("Many Discount Shoppers. Use limited-time promotions or bundle deals to encourage higher spending.")
        else:
            recommendations.append("Discount Shoppers are moderate. Consider small targeted discounts to nudge them toward higher-value purchases.")

        if repeat / total_customers > 0.20:
            recommendations.append("Good base of Repeat customers. Encourage them to become Loyal with membership or upsell strategies.")
        else:
            recommendations.append("Fewer Repeat customers. Nurture them with follow-up campaigns and loyalty incentives.")
    
    if monthly_data is not None and not monthly_data.empty:
        avg_churn = monthly_data['ChurnRate'].mean()
        avg_repeat = monthly_data['RepeatRate'].mean()
        if avg_churn > 20:
            recommendations.append("High churn rate detected. Focus on customer retention through enhanced support and personalized communication.")
        if avg_repeat < 40:
            recommendations.append("Low repeat rate observed. Consider implementing referral programs or subscription models to boost repeat purchases.")
    else:
        recommendations.append("Monthly sales data is insufficient for churn/repeat insights.")

    return recommendations

# ------------------------------
# VISUALIZATION FUNCTIONS
# ------------------------------
def show_segmentation():
    st.subheader("🎯 Customer Segmentation")
    if st.session_state.rfm is not None and not st.session_state.rfm.empty:
        fig = px.pie(
            st.session_state.rfm,
            names='Segment',
            title="Customer Distribution by Segment",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        csv_seg = st.session_state.rfm.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Customer Segmentation as CSV",
            data=csv_seg,
            file_name='customer_segmentation.csv',
            mime='text/csv'
        )
    else:
        st.warning("No segmentation data available")

def show_sales_metrics():
    st.subheader("📈 Sales Performance")
    if st.session_state.filtered_data is not None:
        monthly_data = calculate_churn_repeat(st.session_state.filtered_data)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_data['Month'],
            y=monthly_data['TotalSales'],
            name='Total Sales',
            marker_color='#4CAF50'
        ))
        fig.add_trace(go.Scatter(
            x=monthly_data['Month'],
            y=monthly_data['ChurnRate'],
            name='Churn Rate (%)',
            line=dict(color='#FF5722', width=2),
            mode='lines+markers'
        ))
        fig.add_trace(go.Scatter(
            x=monthly_data['Month'],
            y=monthly_data['RepeatRate'],
            name='Repeat Rate (%)',
            line=dict(color='#2196F3', width=2),
            mode='lines+markers'
        ))
        fig.update_layout(
            title='Month-on-Month Sales vs Churn vs Repeat',
            xaxis_title='Month',
            yaxis_title='Sales ($)',
            yaxis2=dict(
                title='Rate (%)',
                overlaying='y',
                side='right'
            ),
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🔍 Key Sales Insights")
        for insight in generate_sales_insights(monthly_data):
            st.markdown(f"- {insight}")

        st.subheader("💡 Business Recommendations")
        recs = generate_business_recommendations(st.session_state.rfm, monthly_data)
        for r in recs:
            st.markdown(f"- {r}")
    else:
        st.warning("Load data to view sales metrics")

def show_forecast():
    st.subheader("🔮 Historical Sales Forecast")
    if st.session_state.filtered_data is not None:
        forecast = forecast_sales_prophet(st.session_state.filtered_data, periods=12)
        if not forecast.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='Forecast'
            ))
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                fill=None,
                mode='lines',
                line_color='lightgrey',
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                fill='tonexty',
                mode='lines',
                line_color='lightgrey',
                showlegend=False
            ))
            fig.update_layout(
                title='Sales Forecast with Prophet',
                xaxis_title='Date',
                yaxis_title='Sales Amount',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            insights = generate_forecast_insights(forecast)
            st.markdown("### Forecast Insights")
            for point in insights:
                st.markdown(f"- {point}")
            
            st.session_state.forecast = forecast
        else:
            st.warning("Forecasting failed.")
    else:
        st.warning("No data available for forecasting")

def show_new_product_projection():
    st.subheader("🚀 New Product Sales Projection")
    if st.session_state.rfm is None or st.session_state.filtered_data is None:
        st.warning("Run segmentation and load data first!")
        return
    
    with st.form("new_product_form"):
        new_price = st.number_input("New Product Price ($)", min_value=1.0, value=99.99)
        discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=10.0)
        forecast_months = st.selectbox("Forecast Period (Months)", [3, 6, 12])
        growth_rate = st.slider("Expected Monthly Growth Rate (%)", min_value=-20.0, max_value=50.0, value=5.0)
        submitted = st.form_submit_button("Generate Projection")
    
    if submitted:
        projections, seg_analysis = calculate_new_product_projections(
            st.session_state.rfm,
            st.session_state.filtered_data,
            new_price,
            discount,
            forecast_months,
            growth_rate
        )
        if not projections.empty:
            st.session_state.new_product_proj = projections
            
            st.markdown("### New Product Projection Data")
            st.dataframe(
                projections.style.format({
                    'ProjectedUnits': '{:,.0f}',
                    'ProjectedRevenue': '${:,.2f}',
                    'GrowthFactor': '{:.2f}'
                }),
                use_container_width=True
            )
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=projections['Month'],
                y=projections['ProjectedUnits'],
                mode='lines+markers',
                name='Projected Units',
                line=dict(color='green', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=projections['Month'],
                y=projections['ProjectedRevenue'],
                mode='lines+markers',
                name='Projected Revenue',
                line=dict(color='blue', width=3)
            ))
            fig.update_layout(
                title='New Product Sales Projection',
                xaxis_title='Month',
                yaxis_title='Values',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            csv_proj = projections.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download New Product Projection as CSV",
                data=csv_proj,
                file_name='new_product_projection.csv',
                mime='text/csv'
            )
        else:
            st.warning("Projection calculation failed.")

def save_dataset_to_db(df, dataset_name):
    csv_data = df.to_csv(index=False)
    upload_time = datetime.now()
    with engine.connect() as conn:
        query = text("""
            INSERT INTO datasets (dataset_name, upload_time, data)
            VALUES (:dataset_name, :upload_time, :data)
        """)
        conn.execute(query, {"dataset_name": dataset_name, "upload_time": upload_time, "data": csv_data})
        conn.commit()

@st.cache_data
def load_saved_datasets():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT dataset_id, dataset_name, upload_time FROM datasets"))
        datasets = pd.DataFrame(result.fetchall(), columns=result.keys())
    return datasets

def load_dataset_from_db(dataset_id):
    with engine.connect() as conn:
        query = text("SELECT data FROM datasets WHERE dataset_id = :dataset_id")
        result = conn.execute(query, {"dataset_id": dataset_id}).fetchone()
    if result:
        csv_data = result[0]
        return pd.read_csv(StringIO(csv_data))
    return None

# ------------------------------
# SIDEBAR: DATA UPLOAD & CONFIG
# ------------------------------
st.sidebar.header("Data Controls")
upload_choice = st.sidebar.radio("Choose Option", options=["Upload New File", "Load Saved Dataset"])
if upload_choice == "Upload New File":
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded_file:
        file_type = 'csv' if uploaded_file.name.endswith('.csv') else 'excel'
        st.sidebar.info("Automatically detecting column names...")
        new_df = process_data(uploaded_file, file_type)
        if new_df is not None:
            st.sidebar.success("New file processed successfully!")
            # Option to save the dataset
            dataset_name = st.sidebar.text_input("Dataset Name", value=uploaded_file.name)
            if st.sidebar.button("Save Dataset to Database"):
                save_dataset_to_db(new_df, dataset_name)
                st.sidebar.success("Dataset saved to database!")
            st.session_state.raw_data = new_df
            # Apply date filter
            min_date = new_df['Date'].min().date()
            max_date = new_df['Date'].max().date()
            dates = st.sidebar.date_input("Select Analysis Period", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if isinstance(dates, tuple) and len(dates) == 2:
                start, end = pd.to_datetime(dates[0]), pd.to_datetime(dates[1])
                st.session_state.filtered_data = new_df[(new_df['Date'] >= start) & (new_df['Date'] <= end)]
else:
    datasets_df = load_saved_datasets()
    if not datasets_df.empty:
        st.sidebar.write("Saved Datasets:")
        dataset_option = st.sidebar.selectbox("Select Dataset", options=datasets_df["dataset_id"].tolist(), format_func=lambda x: f"{datasets_df[datasets_df['dataset_id']==x]['dataset_name'].values[0]} ({datasets_df[datasets_df['dataset_id']==x]['upload_time'].values[0]})")
        if st.sidebar.button("Load Dataset"):
            loaded_df = load_dataset_from_db(dataset_option)
            if loaded_df is not None:
                st.session_state.raw_data = loaded_df
                min_date = loaded_df['Date'].min().date()
                max_date = loaded_df['Date'].max().date()
                dates = st.sidebar.date_input("Select Analysis Period", value=(min_date, max_date), min_value=min_date, max_value=max_date)
                if isinstance(dates, tuple) and len(dates) == 2:
                    start, end = pd.to_datetime(dates[0]), pd.to_datetime(dates[1])
                    st.session_state.filtered_data = loaded_df[(loaded_df['Date'] >= start) & (loaded_df['Date'] <= end)]
            else:
                st.sidebar.error("Failed to load dataset.")
    else:
        st.sidebar.info("No saved datasets found.")

# Button to run business segmentation
if st.sidebar.button("Run Business Segmentation"):
    if st.session_state.filtered_data is not None:
        st.session_state.rfm = perform_simple_rfm_segmentation(st.session_state.filtered_data)
        st.sidebar.success("Segmentation completed!")
    else:
        st.sidebar.error("Filtered data not available. Adjust your date range.")

# ------------------------------
# MAIN DASHBOARD DISPLAY
# ------------------------------
st.title("Retail Analytics Dashboard")
if st.session_state.filtered_data is not None:
    show_segmentation()
    st.markdown("---")
    show_sales_metrics()
    st.markdown("---")
    show_forecast()
    st.markdown("---")
    show_new_product_projection()
else:
    st.info("Please upload a file or load a saved dataset and configure settings from the sidebar.")
