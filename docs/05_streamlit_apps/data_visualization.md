# Data Visualization

## Overview

Streamlit provides powerful tools for creating interactive data visualizations. It integrates seamlessly with popular plotting libraries like Matplotlib, Plotly, Altair, and offers its own native charting capabilities.

## Native Streamlit Charts

### Line Chart

```python
import streamlit as st
import pandas as pd
import numpy as np

# Generate sample data
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['A', 'B', 'C']
)

# Simple line chart
st.line_chart(chart_data)

# With specific columns
st.line_chart(chart_data, y=['A', 'B'])
```

### Area Chart

```python
import streamlit as st
import pandas as pd
import numpy as np

data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['Series 1', 'Series 2', 'Series 3']
)

st.area_chart(data)
```

### Bar Chart

```python
import streamlit as st
import pandas as pd

data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'],
    'Values': [10, 25, 15, 30]
})

st.bar_chart(data.set_index('Category'))
```

### Scatter Chart

```python
import streamlit as st
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'size': np.random.rand(100) * 100
})

st.scatter_chart(data, x='x', y='y', size='size')
```

## Matplotlib Integration

### Basic Matplotlib

```python
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Create figure
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, np.cos(x), label='cos(x)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.set_title('Trigonometric Functions')

# Display in Streamlit
st.pyplot(fig)
```

### Multiple Subplots

```python
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Subplot 1: Line
axes[0, 0].plot(np.random.randn(50).cumsum())
axes[0, 0].set_title('Line Plot')

# Subplot 2: Histogram
axes[0, 1].hist(np.random.randn(1000), bins=30)
axes[0, 1].set_title('Histogram')

# Subplot 3: Scatter
axes[1, 0].scatter(np.random.randn(50), np.random.randn(50))
axes[1, 0].set_title('Scatter Plot')

# Subplot 4: Bar
axes[1, 1].bar(['A', 'B', 'C', 'D'], [3, 7, 2, 5])
axes[1, 1].set_title('Bar Plot')

plt.tight_layout()
st.pyplot(fig)
```

## Plotly Interactive Charts

### Basic Plotly

```python
import streamlit as st
import plotly.express as px
import pandas as pd

# Sample data
df = px.data.gapminder()

# Create interactive chart
fig = px.scatter(
    df.query("year == 2007"),
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    hover_name="country",
    log_x=True,
    size_max=60,
    title="Life Expectancy vs GDP (2007)"
)

st.plotly_chart(fig, use_container_width=True)
```

### Line Chart with Plotly

```python
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

# Create data
dates = pd.date_range('2023-01-01', periods=100)
df = pd.DataFrame({
    'Date': dates,
    'Value A': np.cumsum(np.random.randn(100)),
    'Value B': np.cumsum(np.random.randn(100))
})

fig = px.line(df, x='Date', y=['Value A', 'Value B'],
              title='Time Series Data')
fig.update_layout(hovermode='x unified')

st.plotly_chart(fig, use_container_width=True)
```

### Interactive Bar Chart

```python
import streamlit as st
import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D', 'E'],
    '2022': [10, 15, 8, 22, 18],
    '2023': [12, 18, 10, 25, 20]
})

fig = px.bar(df, x='Category', y=['2022', '2023'],
             barmode='group', title='Year-over-Year Comparison')

st.plotly_chart(fig, use_container_width=True)
```

### 3D Plots

```python
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd

# Generate 3D data
n = 100
df = pd.DataFrame({
    'x': np.random.randn(n),
    'y': np.random.randn(n),
    'z': np.random.randn(n),
    'category': np.random.choice(['A', 'B', 'C'], n)
})

fig = px.scatter_3d(df, x='x', y='y', z='z', color='category',
                     title='3D Scatter Plot')

st.plotly_chart(fig, use_container_width=True)
```

## Altair Charts

### Basic Altair

```python
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

# Create data
source = pd.DataFrame({
    'x': range(100),
    'y': np.random.randn(100).cumsum()
})

# Create chart
chart = alt.Chart(source).mark_line().encode(
    x='x',
    y='y'
).properties(
    title='Altair Line Chart'
)

st.altair_chart(chart, use_container_width=True)
```

### Interactive Altair

```python
import streamlit as st
import altair as alt
import pandas as pd

# Sample data
source = pd.DataFrame({
    'a': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
    'b': [28, 55, 43, 91, 81, 53, 19, 87, 52]
})

# Interactive bar chart with tooltip
chart = alt.Chart(source).mark_bar().encode(
    x='a',
    y='b',
    tooltip=['a', 'b']
).interactive()

st.altair_chart(chart, use_container_width=True)
```

### Layered Charts

```python
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

# Create data
x = np.linspace(0, 10, 50)
source = pd.DataFrame({
    'x': x,
    'sin': np.sin(x),
    'cos': np.cos(x)
})

# Create base
base = alt.Chart(source).encode(x='x')

# Layer multiple lines
chart = base.mark_line(color='blue').encode(y='sin') + \
        base.mark_line(color='red').encode(y='cos')

st.altair_chart(chart, use_container_width=True)
```

## Map Visualizations

### Basic Map

```python
import streamlit as st
import pandas as pd
import numpy as np

# Sample coordinates
df = pd.DataFrame({
    'lat': np.random.uniform(37, 38, 100),
    'lon': np.random.uniform(-122, -121, 100)
})

st.map(df)
```

### Plotly Maps

```python
import streamlit as st
import plotly.express as px

# US map with data
df = px.data.gapminder().query("year == 2007")

fig = px.choropleth(
    df,
    locations="iso_alpha",
    color="lifeExp",
    hover_name="country",
    color_continuous_scale=px.colors.sequential.Plasma,
    title="Life Expectancy by Country"
)

st.plotly_chart(fig, use_container_width=True)
```

### PyDeck Maps

```python
import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np

# Sample data
df = pd.DataFrame({
    'lat': np.random.uniform(37.7, 37.8, 1000),
    'lon': np.random.uniform(-122.5, -122.4, 1000)
})

# Create deck
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=37.75,
        longitude=-122.45,
        zoom=11,
        pitch=50
    ),
    layers=[
        pdk.Layer(
            'HexagonLayer',
            data=df,
            get_position='[lon, lat]',
            radius=100,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True
        )
    ]
))
```

## Real-time Updates

### Animated Charts

```python
import streamlit as st
import numpy as np
import time

# Placeholder for chart
chart_placeholder = st.empty()

# Animate
for i in range(100):
    # Generate new data
    data = np.random.randn(20, 3)

    # Update chart
    chart_placeholder.line_chart(data)

    # Pause
    time.sleep(0.1)
```

### Streaming Data

```python
import streamlit as st
import pandas as pd
import numpy as np
import time

# Initialize empty dataframe
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['value'])

# Create chart placeholder
chart = st.line_chart(st.session_state.data)

# Add data points
if st.button('Start Streaming'):
    for i in range(50):
        new_row = pd.DataFrame({'value': [np.random.randn()]})
        st.session_state.data = pd.concat([st.session_state.data, new_row])
        chart.add_rows(new_row)
        time.sleep(0.1)
```

## Interactive Dashboards

### Dashboard Example

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Sales Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Date Range", [])
region = st.sidebar.multiselect("Region", ["North", "South", "East", "West"])

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", "$1.2M", "+12%")
col2.metric("Orders", "1,234", "+5%")
col3.metric("Customers", "567", "+8%")
col4.metric("Avg Order", "$97", "-2%")

# Charts row
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales by Region")
    df = pd.DataFrame({
        'Region': ['North', 'South', 'East', 'West'],
        'Sales': [300, 250, 400, 350]
    })
    fig = px.pie(df, values='Sales', names='Region')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Monthly Trend")
    df = pd.DataFrame({
        'Month': pd.date_range('2023-01', periods=12, freq='M'),
        'Sales': np.random.randint(80, 120, 12)
    })
    fig = px.line(df, x='Month', y='Sales')
    st.plotly_chart(fig, use_container_width=True)
```

## Exercises

1. Create an interactive dashboard with filters that update charts
2. Build a real-time streaming data visualization
3. Create a geographic visualization with custom data
4. Build a multi-chart comparison tool

## Additional Resources

- [Streamlit Charts](https://docs.streamlit.io/library/api-reference/charts)
- [Plotly Documentation](https://plotly.com/python/)
- [Altair Documentation](https://altair-viz.github.io/)
