"""Quick test with inline data - no external files needed"""

import pandas as pd
from context2visual import generate_visualization

# Create sample data inline
df = pd.DataFrame({
    'observation': [
        'Liukas lattia käytävällä',
        'Portaat liukkaat',
        'Kaapeli kulkuväylällä',
        'Vuotava hana',
        'Liukas lattia'
    ],
    'status': ['Archived', 'Implemented', 'Archived', 'Commenting', 'Archived'],
    'created': pd.date_range('2024-05-01', periods=5, freq='D'),
    'handled': pd.date_range('2024-05-03', periods=5, freq='D')
})

# Test 1: Bar chart
print("Test 1: Status distribution")
fig1 = generate_visualization("Show status distribution", df)
print(f"✅ Result: {type(fig1).__name__ if fig1 else 'None'}\n")

# Test 2: Time series
print("Test 2: Trend over time")
fig2 = generate_visualization("Show trend over time", df)
print(f"✅ Result: {type(fig2).__name__ if fig2 else 'None'}\n")

# Test 3: Histogram
print("Test 3: Handling time")
fig3 = generate_visualization("What was average handling time?", df)
print(f"✅ Result: {type(fig3).__name__ if fig3 else 'None'}\n")

print("✅ All tests completed!")