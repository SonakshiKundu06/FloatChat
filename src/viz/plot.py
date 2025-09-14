import plotly.express as px
import folium

def plot_profile(df, float_id):
    """Plot temperature & salinity vs pressure for a given float"""
    df_float = df[df["platform_number"] == float_id]

    fig = px.line(
        df_float,
        x="temp",
        y="pres",
        color="year",
        labels={"temp": "Temperature (°C)", "pres": "Pressure (dbar)"},
        title=f"Temperature Profile for Float {float_id}"
    )
    fig.update_yaxes(autorange="reversed")  # depth goes down
    return fig

def plot_salinity(df, float_id):
    """Plot salinity vs pressure for a given float"""
    df_float = df[df["platform_number"] == float_id]

    fig = px.line(
        df_float,
        x="psal",
        y="pres",
        color="year",
        labels={"psal": "Salinity (PSU)", "pres": "Pressure (dbar)"},
        title=f"Salinity Profile for Float {float_id}"
    )
    fig.update_yaxes(autorange="reversed")
    return fig

def plot_float_map(df):
    """Plot float positions on a map"""
    m = folium.Map(location=[0, 0], zoom_start=2)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            popup=f"Float {row['platform_number']} ({row['year']})",
            color="blue",
            fill=True,
        ).add_to(m)
    return m
