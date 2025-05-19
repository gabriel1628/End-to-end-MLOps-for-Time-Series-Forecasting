import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go


sns.set_theme(style="darkgrid")

st.set_page_config(page_title="Enefit - Energy consumption", layout="centered")

# this is the header
st.title("Energy Consumption Report")
# st.markdown(
#     " **tel:** 050 000 0000 **| website:** https://www.example.com **| email:** mailto:example@mail.com"
# )

# Custom app width
# https://discuss.streamlit.io/t/change-main-app-width/85715
st.html(
    """
    <style>
        .stMainBlockContainer {
            max-width:60rem;
        }
    </style>
    """
)


@st.cache_data
def load_data():
    df = pd.read_parquet("./data/data.parquet")
    consumption = df[df["is_consumption"] == 1].drop(columns=["is_consumption"])
    consumption.rename(
        columns={"prediction_unit_id": "segment_id", "target": "consumption"},
        inplace=True,
    )
    del df
    y = consumption[["consumption", "datetime"]].copy().set_index("datetime")
    y_day_mean = y.resample("D").mean()
    return consumption, y_day_mean


consumption, y_day_mean = load_data()

# segment = st.selectbox(
#     "Select a segment ID", ["All"] + consumption["segment_id"].unique().tolist()
# )

cell_hover = {  # for row hover use <tr> instead of <td>
    "selector": "td:hover",
    "props": [("background-color", "#ffffb3")],
}
index_names = {
    "selector": ".index_name",
    "props": "font-style: italic; color: darkgrey; font-weight:normal;",
}
headers = {
    "selector": "th:not(.index_name)",
    "props": "background-color: #000066; color: white;",
}
columns = ["segment_id", "consumption", "county"]
stats = ["mean", "std", "min", "max"]
s = (
    consumption[columns]
    .describe()
    .T.style.format(precision=2, thousands=",", decimal=".")
)
# s = s.set_table_styles([cell_hover, index_names, headers])
st.dataframe(
    s,
    use_container_width=True,
)

# with st.expander("Statistics", expanded=True):
# cell_hover = {  # for row hover use <tr> instead of <td>
#     "selector": "td:hover",
#     "props": [("background-color", "#ffffb3")],
# }
# index_names = {
#     "selector": ".index_name",
#     "props": "font-style: italic; color: darkgrey; font-weight:normal;",
# }
# headers = {
#     "selector": "th:not(.index_name)",
#     "props": "background-color: #000066; color: white;",
# }
# columns = ["segment_id", "consumption", "county"]
# stats = ["mean", "std", "min", "max"]
# s = (
#     consumption[columns]
#     .describe()
#     .T[stats]
#     .style.format(precision=3, thousands=",", decimal=".")
# )
# # s = s.set_table_styles([cell_hover, index_names, headers])
# st.dataframe(
#     s,
#     use_container_width=True,
# )

# colourcode = [
#     "#FFCF8B",
#     "#F0F2F6",
# ]
# data = consumption[columns].describe().T.reset_index()
# table = go.Table(
#     columnorder=[i for i in range(0, data.shape[1])],
#     columnwidth=[18, 12],
#     header=dict(
#         values=list(data.columns),
#         font=dict(size=15, color="white"),
#         fill_color="#264653",
#         line_color="rgba(255,255,255,0.2)",
#         align=["left", "center"],
#         # text wrapping
#         height=40,
#     ),
#     cells=dict(
#         values=[data[K].tolist() for K in data.columns],
#         font=dict(size=12),
#         align=["left", "center"],
#         fill_color=colourcode,
#         line_color="rgba(255,255,255,0.2)",
#         height=30,
#     ),
# )
# fig = go.Figure(data=[table])
# fig.update_layout(
#     title_text="Statistics",
#     title_font_color="#264653",
#     title_x=0,
#     margin=dict(l=0, r=10, b=10, t=30),
#     height=480,
# )

# st.plotly_chart(fig, use_container_width=True)


col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    sns.histplot(consumption, x="consumption", log_scale=True, ax=ax1)
    ax1.set_title("Energy consumption distribution")
    st.pyplot(fig1)

with col2:
    consumption_sample = consumption[consumption["segment_id"].isin([0, 1, 2])]
    fig2, ax2 = plt.subplots()
    sns.kdeplot(
        consumption_sample,
        x="consumption",
        hue="segment_id",
        log_scale=True,
        fill=True,
        palette="Set1",
        linewidth=0,
        alpha=0.4,
        ax=ax2,
    )
    ax2.set_title("Energy consumption distribution by segment")
    ax2.set_xlabel("Energy consumption")
    ax2.set_ylabel("Count")
    st.pyplot(ax2.get_figure())


fig3, ax3 = plt.subplots(figsize=(10, 4))
y_day_mean.ffill().plot.line(ax=ax3, color="indigo", legend=False)
ax3.set_title("Average daily energy consumption per segment")
ax3.set_xlabel("Date", fontsize=10)
ax3.set_ylabel("Energy consumption (MWh)", fontsize=10)
st.pyplot(ax3.get_figure())
