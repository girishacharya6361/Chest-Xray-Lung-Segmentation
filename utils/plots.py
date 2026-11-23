"""Plotly charts in the project's navy / teal palette."""
import plotly.graph_objects as go
import plotly.express as px

PALETTE = ["#0B2545", "#13A89E", "#8DA9C4", "#F2A541", "#3A506B"]


def _style(fig, h=380):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(family="Inter, Segoe UI, sans-serif", color="#1f2a3c"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.2))
    return fig


def grouped_bar(df, x, cols, title, yrange=(0, 1.05)):
    fig = go.Figure()
    for i, c in enumerate(cols):
        fig.add_bar(name=c, x=df[x], y=df[c], marker_color=PALETTE[i % len(PALETTE)],
                    text=[f"{v:.3f}" for v in df[c]], textposition="outside")
    fig.update_layout(barmode="group", title=title)
    if yrange:
        fig.update_yaxes(range=list(yrange))
    return _style(fig)


def simple_bar(df, x, y, title, ytitle):
    fig = px.bar(df, x=x, y=y, color=x, color_discrete_sequence=PALETTE, text=y, title=title)
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title=ytitle, xaxis_title=None)
    return _style(fig, 340)


def loss_curves(epochs):
    ep = [e["epoch"] for e in epochs]
    fig = go.Figure()
    fig.add_scatter(x=ep, y=[e["train_loss"] for e in epochs], name="Training loss", line=dict(color="#0B2545", width=3))
    fig.add_scatter(x=ep, y=[e["val_loss"] for e in epochs], name="Validation loss", line=dict(color="#13A89E", width=3))
    fig.update_layout(title="Training and validation loss (Dice + BCE)", xaxis_title="Epoch", yaxis_title="Loss")
    return _style(fig)


def dice_curve(epochs):
    ep = [e["epoch"] for e in epochs]
    fig = go.Figure()
    fig.add_scatter(x=ep, y=[e["val_dice"] for e in epochs], name="Val Dice", line=dict(color="#13A89E", width=3))
    fig.add_scatter(x=ep, y=[e["val_iou"] for e in epochs], name="Val IoU", line=dict(color="#0B2545", width=3))
    fig.update_layout(title="Validation Dice / IoU", xaxis_title="Epoch", yaxis_range=[0, 1.02])
    return _style(fig)


def spectrum(img, title):
    fig = px.imshow(img, color_continuous_scale="Teal", title=title)
    fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=40, b=0), height=320)
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig
