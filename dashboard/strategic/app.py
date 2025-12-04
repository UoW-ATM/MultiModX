import dash
from layout import create_layout
from callbacks import register_callbacks

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.title = "Strategic Multimodal Dashboard"

app.layout = create_layout()

register_callbacks(app)

if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run(debug=True)
