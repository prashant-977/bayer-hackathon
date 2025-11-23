"""
Live Interactive Demo for Stakeholders
Run with: streamlit run demo.py
"""

import streamlit as st
import pandas as pd
from context2visual import generate_visualization

st.set_page_config(page_title="Context2Visual Demo", layout="wide")

st.title("🎯 Context2Visual - Live Demo")
st.markdown("---")

# Sidebar: Select scenario
st.sidebar.header("Select Demo Scenario")
scenario = st.sidebar.selectbox(
    "Choose a scenario:",
    [
        "Status Distribution",
        "Time Trend",
        "Handling Time",
        "Text Extraction",
        "Same Data - Different Questions"
    ]
)

# ============================================================================
# SCENARIO 1: Status Distribution
# ============================================================================
if scenario == "Status Distribution":
    st.header("📊 Scenario 1: Status Distribution")
    
    st.markdown("""
    **Use Case:** Team leader wants to see breakdown of observation statuses
    
    **Question:** "Show status distribution"
    """)
    
    # Sample data
    df = pd.DataFrame({
        'observation': [
            'Liukas lattia käytävällä',
            'Portaat liukkaat',
            'Kaapeli kulkuväylällä',
            'Vuotava hana',
            'Liukas lattia taukotilassa',
            'Hylly ylikuormitettu',
            'Palovaroitin ei toimi',
            'Kemikaaliroiske lattialla'
        ],
        'status': [
            'Archived', 'Implemented', 'Archived', 'Commenting',
            'Archived', 'Implemented', 'In Progress', 'Archived'
        ]
    })
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Input Data")
        st.dataframe(df, height=300)
        st.metric("Total Observations", len(df))
    
    with col2:
        st.subheader("📊 Generated Visualization")
        
        prompt = st.text_input("User Prompt:", "Show status distribution", key="s1")
        
        if st.button("Generate", key="b1"):
            with st.spinner("Generating..."):
                fig = generate_visualization(prompt, df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("✅ Bar chart showing status breakdown")
                else:
                    st.warning("No visualization generated")

# ============================================================================
# SCENARIO 2: Time Trend
# ============================================================================
elif scenario == "Time Trend":
    st.header("📈 Scenario 2: Temporal Trend")
    
    st.markdown("""
    **Use Case:** HSE team wants to see if observations are increasing/decreasing
    
    **Question:** "Show trend over time"
    """)
    
    df = pd.DataFrame({
        'observation': [f'Observation {i}' for i in range(1, 21)],
        'created': pd.date_range('2024-05-01', periods=20, freq='2D'),
        'status': ['Archived'] * 20
    })
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Input Data")
        st.dataframe(df[['observation', 'created']].head(10), height=300)
        st.metric("Total Observations", len(df))
        st.metric("Date Range", f"{df['created'].min().date()} to {df['created'].max().date()}")
    
    with col2:
        st.subheader("📊 Generated Visualization")
        
        prompt = st.text_input("User Prompt:", "Show trend over time", key="s2")
        
        if st.button("Generate", key="b2"):
            with st.spinner("Generating..."):
                fig = generate_visualization(prompt, df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("✅ Time series showing observation trend")
                else:
                    st.warning("No visualization generated")

# ============================================================================
# SCENARIO 3: Handling Time
# ============================================================================
elif scenario == "Handling Time":
    st.header("⏱️ Scenario 3: Handling Time Analysis")
    
    st.markdown("""
    **Use Case:** Manager wants to know how long it takes to resolve observations
    
    **Question:** "What was the average handling time?"
    """)
    
    df = pd.DataFrame({
        'title': [
            'Vesilammikko käytävällä',
            'Heiluva kaide rappusissa',
            'Puutteellinen suoja-asuste',
            'Tavaraa portaisiin',
            'Kemikaaliroiske'
        ],
        'havainto_pvm': ['2024-01-10', '2024-01-18', '2024-01-25', '2024-02-02', '2024-02-09'],
        'havainto_käsitelty_pvm': ['2024-01-11', '2024-01-25', '2024-01-26', '2024-02-03', '2024-02-10']
    })
    
    # Convert to datetime
    df['havainto_pvm'] = pd.to_datetime(df['havainto_pvm'])
    df['havainto_käsitelty_pvm'] = pd.to_datetime(df['havainto_käsitelty_pvm'])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Input Data")
        st.dataframe(df, height=300)
        
        # Calculate durations for display
        durations = (df['havainto_käsitelty_pvm'] - df['havainto_pvm']).dt.days
        st.metric("Average Duration", f"{durations.mean():.1f} days")
        st.metric("Range", f"{durations.min()}-{durations.max()} days")
    
    with col2:
        st.subheader("📊 Generated Visualization")
        
        prompt = st.text_input("User Prompt:", "What was the average handling time?", key="s3")
        
        if st.button("Generate", key="b3"):
            with st.spinner("Generating..."):
                fig = generate_visualization(prompt, df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("✅ Histogram with average/median lines")
                else:
                    st.warning("No visualization generated")

# ============================================================================
# SCENARIO 4: Text Extraction
# ============================================================================
elif scenario == "Text Extraction":
    st.header("🏷️ Scenario 4: Category Extraction from Text")
    
    st.markdown("""
    **Use Case:** No explicit categories in data, extract from Finnish text
    
    **Question:** "What are the most common types of observations?"
    """)
    
    df = pd.DataFrame({
        'observation': [
            'Liukas lattia käytävällä, liukastumisvaara',
            'Portaat erittäin liukkaat talvella',
            'Kaapeli kulkuväylällä kompastumisvaara',
            'Verkkokaapeli käytävän poikki',
            'Vuotava hana taukotilassa',
            'Vesivahinko vuotavan hanan takia',
            'Liukas lattia taukotilassa',
            'Portaiden kaide heiluu vaarallisesti',
            'Sähköjohto irti seinästä',
            'Kaapeli lattialla toimistossa'
        ]
    })
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Input Data (Text Only)")
        st.dataframe(df, height=300)
        st.info("ℹ️ No categorical columns - will extract from text using TF-IDF")
    
    with col2:
        st.subheader("📊 Generated Visualization")
        
        prompt = st.text_input("User Prompt:", "What are the most common types?", key="s4")
        
        if st.button("Generate", key="b4"):
            with st.spinner("Extracting categories from text..."):
                fig = generate_visualization(prompt, df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("✅ Categories automatically extracted: Liukas, Kaapeli, Portaat, Vuotava, etc.")
                else:
                    st.warning("No visualization generated")

# ============================================================================
# SCENARIO 5: Same Data, Different Questions
# ============================================================================
else:  # Same Data - Different Questions
    st.header("🎯 Scenario 5: Same Data, Different Analytical Angles")
    
    st.markdown("""
    **Use Case:** Different team members ask different questions about SAME observations
    
    **Demonstrates:** Context-aware visualization selection
    """)
    
    # Comprehensive dataset
    df = pd.DataFrame({
        'observation': ['Liukas lattia', 'Portaat liukkaat', 'Kaapeli'] * 5,
        'status': ['Archived', 'Implemented', 'Archived'] * 5,
        'created': pd.date_range('2024-05-01', periods=15, freq='2D'),
        'handled': pd.date_range('2024-05-03', periods=15, freq='2D')
    })
    
    st.subheader("📋 Same Input Data (15 observations)")
    st.dataframe(df.head(10), height=200)
    
    st.markdown("---")
    st.subheader("📊 Different Questions → Different Visualizations")
    
    # Question selector
    question = st.radio(
        "Choose a question:",
        [
            "Show status distribution",
            "Show trend over time",
            "What was average handling time?"
        ]
    )
    
    if st.button("Generate Visualization", key="b5"):
        with st.spinner("Generating..."):
            fig = generate_visualization(question, df)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Explain what happened
                if "status" in question:
                    st.success("✅ Generated BAR CHART - shows status categories")
                elif "trend" in question:
                    st.success("✅ Generated TIME SERIES - shows observations over time")
                elif "time" in question:
                    st.success("✅ Generated HISTOGRAM - shows duration distribution")
                
                st.info("💡 Same data, but visualization adapts to the question!")
            else:
                st.warning("No visualization generated")

# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.markdown("""
### 🎯 Key Features Demonstrated:
- ✅ Automatic chart type selection based on question
- ✅ Works with Finnish text (no English required)
- ✅ Extracts categories from text using TF-IDF
- ✅ Calculates durations automatically
- ✅ Non-technical user friendly (clear labels, annotations)
- ✅ Same data → different visualizations based on context
""")