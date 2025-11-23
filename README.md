# 🌟 Context2Visual  
### *AI-powered, context-aware visualizations for Bayer’s HSE RAG system*

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-Enabled-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/AI-Context%20Analysis-blueviolet" />
</p>

---

## 📖 Overview

**Context2Visual** is a modular Python component that adds **automatic, context-aware visualizations** to an existing HSE RAG (Retrieval-Augmented Generation) application.

It turns this:

```text
User Prompt → SQL → DataFrame → LLM Summary

into this:

User Prompt → SQL → DataFrame
        ↘ Context2Visual ↙
     LLM Summary  +  Interactive Charts

Users never have to say “give me a bar chart” — the module looks at the prompt and the DataFrame and decides what to draw.

------------------------------------------------------------------------------------------------------------

🎯 Key Features

- Zero-click visualizations
  Charts are generated automatically based on context.

- Intent-aware logic
  Detects whether the user is asking about:
  - trends over time
  - distributions
  - handling / processing time
  - category breakdowns (status, location, topics)
  - urgency

- Data-aware
  Inspects the DataFrame for:
    - timestamps
    - numeric columns
    - categorical columns
    - free-text fields
    - duration (start–end date)

- Streamlit-ready
  Returns Plotly Figure objects ready for st.plotly_chart().

- Minimal integration effort
  Plugs into the existing Bayer HSE RAG frontend with just a few lines.

------------------------------------------------------------------------------------------

🏗️ Project Structure

This is the structure of the repository:

bayer-hackathon/
│
├── context2visual.py        # Core module: intent detection + chart pipeline
├── quick_test.py            # Small script to manually test the module
├── requirements.txt         # Python dependencies
├── PROJECT_STRUCTURE.md     # Extra details on layout (internal notes)
├── README.md                # This file
│
├── docs/                    # Documentation, etc.
│   └── ...                  # e.g., API_REFERENCE.md, ARCHITECTURE.md, INSIGHT_EXTRACTION.MD..
│
├── examples/                # Example prompts + DataFrames / notebooks
│   └── ...                  # e.g., basic_usage.py, compare_with_llm_summary.py, generate_more_test_data.py...
│
├── integration/             # Example integration into the reference HSEBot app
│   └── ...                  # e.g., components_with_viz.py
│
└── tests/                   # Tests (manual & automated)
    └── ...                  # e.g., test_llm_alignment.py, test_same_data_angles.py, test_visualization.py...

----------------------------------------------------------------------------------------------

⚙️ Installation

Clone the repo:-------------------
git clone https://github.com/your-username/bayer-hackathon.git
cd bayer-hackathon

Create and activate a virtual environment (recommended):------------
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

Install dependencies:-------------------------------
pip install -r requirements.txt

----------------------------------------------------------------------------------------------------

🧩 How It Works (High Level)

The core logic lives in context2visual.py and has four main layers:

1. Context Analysis

  - Reads the user prompt
  - Detects intent: trend, distribution, categories, duration, urgency
  - Inspects DataFrame:
    - finds timestamp columns (created, handled, *_date, *_pvm)
    - identifies text columns (observation, havainto, etc.)
    - notes categorical vs numeric columns

2. Visualization Type Selection

Maps intent + data → chart type. For example:

If user asks about                    And data has	              We use 
“trend”, “over time”, “change”,     Date column	                  📈 Line chart
 “per month”	

“status distribution”, “which       Categorical column	          📊 Bar chart
category…”

“handling time”, “how long”,        Two date columns	            📉 Histogram
“käsittelyaika”

Vague query	                        Anything at all	              🧩 Fallback “best guess”

3. Data Preparation

  - Converts timestamps to proper datetime
  - Derives:
      - observation month / week
      - processing duration in days
      - Extracts rough categories from text (via TF–IDF or simple keyword rules)
      - Aggregates counts and averages as needed

4. Visualization Creation (Plotly)

Outputs Plotly Figure objects with:
  - Clear titles
  - Friendly labels
  - Annotations (e.g. averages, medians, totals)
  - Good defaults for non-technical users

-------------------------------------------------------------------------------------------------------

🧪 Usage Examples

Minimal Python example:----------------------- 

import pandas as pd
from context2visual import generate_visualizations

# Example: load a scenario from examples/
df = pd.read_excel("examples/example_scenario.xlsx")
prompt = "What is the distribution of handling times for these observations?"

figs = generate_visualizations(prompt, df)

for fig in figs:
    fig.show()

Streamlit integration:---------------------------

In the existing HSEBot app, after the DataFrame is retrieved:

import streamlit as st
from context2visual import generate_visualizations

# Existing code:
df = load_data(generated_query, params)
st.dataframe(df)

# NEW: Visualization
with st.spinner("Generating visualization..."):
    figures = generate_visualizations(user_input, df)

for fig in figures:
    st.plotly_chart(fig, use_container_width=True)

# Existing summary code continues:
result = analyze_observations(...)
st.write(result)

🙏 Acknowledgements

This project was developed for the Bayer SinceAI Hackathon, focusing on on-demand visualization of HSE data in an existing RAG-based application.


