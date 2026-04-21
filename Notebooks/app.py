import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from run_model import predict_patient

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Required for Render

app.layout = dbc.Container([
    html.H1("❤️ Heart Disease Risk Predictor", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Age"),
            dbc.Input(id="age", type="number", value=50),

            dbc.Label("Sex (1 = Male, 0 = Female)"),
            dbc.Input(id="sex", type="number", value=1),

            dbc.Label("Chest Pain Type (cp)"),
            dbc.Input(id="cp", type="number", value=0),

            dbc.Label("Resting Blood Pressure"),
            dbc.Input(id="trestbps", type="number", value=120),

            dbc.Label("Cholesterol"),
            dbc.Input(id="chol", type="number", value=200),

            dbc.Label("Fasting Blood Sugar (1 = True, 0 = False)"),
            dbc.Input(id="fbs", type="number", value=0),
        ]),

        dbc.Col([
            dbc.Label("Resting ECG"),
            dbc.Input(id="restecg", type="number", value=0),

            dbc.Label("Max Heart Rate"),
            dbc.Input(id="thalach", type="number", value=150),

            dbc.Label("Exercise Induced Angina"),
            dbc.Input(id="exang", type="number", value=0),

            dbc.Label("Oldpeak"),
            dbc.Input(id="oldpeak", type="number", value=1.0),

            dbc.Label("Slope"),
            dbc.Input(id="slope", type="number", value=1),

            dbc.Label("CA"),
            dbc.Input(id="ca", type="number", value=0),

            dbc.Label("Thal"),
            dbc.Input(id="thal", type="number", value=1),
        ])
    ]),

    html.Br(),

    dbc.Button("Predict Risk", id="predict-btn", color="primary", className="w-100"),

    html.Br(), html.Br(),

    html.Div(id="output-result", className="text-center fs-4")
])

@app.callback(
    Output("output-result", "children"),
    Input("predict-btn", "n_clicks"),
    State("age", "value"),
    State("sex", "value"),
    State("cp", "value"),
    State("trestbps", "value"),
    State("chol", "value"),
    State("fbs", "value"),
    State("restecg", "value"),
    State("thalach", "value"),
    State("exang", "value"),
    State("oldpeak", "value"),
    State("slope", "value"),
    State("ca", "value"),
    State("thal", "value"),
)
def make_prediction(n_clicks, age, sex, cp, trestbps, chol, fbs,
                    restecg, thalach, exang, oldpeak, slope, ca, thal):

    if n_clicks is None:
        return ""

    patient = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    result = predict_patient(patient, model_key='rf')

    return html.Div([
        html.H3(f"Prediction: {result['risk_label']}"),
        html.P(f"Probability: {result['probability']}"),
        html.P(f"Model Used: {result['model']}")
    ])

if __name__ == "__main__":
    app.run(debug=True)