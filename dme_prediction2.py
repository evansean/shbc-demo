import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd 
import math
import pickle 


app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # for deployment, if needed

# Load training data/model
df = pd.read_csv('./data/RF_training_200dataset_1oct.csv')
snellen_lkup = pd.read_excel('./data/logmar_snellen_lookup.xlsx')
snellen_lkup.set_index('logmar', inplace = True)
    
# Load the trained Random Forest model from pickle for post_thick
with open('./models/rf_thick_seed21.pkl', 'rb') as f:
    rf_thick = pickle.load(f)
    
# Load the trained Random Forest model from pickle for post_logmar
with open('./models/rf_logmar_seed21.pkl', 'rb') as f:
    rf_logmar = pickle.load(f) 

snellen_num = 6    

def make_panel(tag: str, title_main: str, title_sub: str):
    """
    One card/panel. `tag` keeps IDs unique ('A', 'B', etc.).
    `title_main` and `title_sub` render the two-line centered heading.
    """
    return html.Div(
        className="form-card",
        children=[
            html.Div(
                className="title-wrapper",
                children=[
                    html.Div(title_main, className="title-main"),
                    html.Div(title_sub,  className="title-sub"),
                ],
            ),

            # Row 1: Age | HbA1c
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="col",
                        children=dcc.Input(
                            id=f"age_{tag}",
                            type="number",
                            placeholder="Enter Age",
                            className="inp"
                        ),
                    ),
                    html.Div(
                        className="col",
                        children=dcc.Input(
                            id=f"hba1c_{tag}",
                            type="number",
                            placeholder="Enter HbA1c",
                            className="inp"
                        ),
                    ),
                ],
            ),

            # Row 2: Thickness | ( Num / Den ) as a single 'input-group'
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="col",
                        children=dcc.Input(
                            id=f"thickness_{tag}",
                            type="number",
                            placeholder="Enter macular thickness",
                            className="inp"
                        ),
                    ),
                    html.Div(
                        className="col",
                        children=html.Div(
                            className="input-group",
                            children=[
                                html.Span("Snellen VA: 6", style={'font-size':24}, className="label"),
                                # dcc.Input(
                                #     id=f"snellen_num_{tag}",
                                #     type="number",
                                #     value = 6,
                                #     # placeholder="Snell. Num",
                                #     className="inp inp-small"
                                # ),
                                html.Span("/", className="slash"),
                                dcc.Input(
                                    id=f"snellen_den_{tag}",
                                    type="number",
                                    placeholder="Denominator",
                                    className="inp inp-small"
                                ),
                            ],
                        ),
                    ),
                ],
            ),

            # Buttons (flush right)
            html.Div(
                className="buttons",
                children=[
                    html.Button("Submit", id=f"submit_{tag}", n_clicks=0, className="btn"),
                    html.Button("Clear",  id=f"clear_{tag}",  n_clicks=0, className="btn btn-secondary"),
                ],
            ),

            # Results
            html.Div(id=f"result_{tag}", className="result"),
        ],
    )

# Layout - two side-by-side panels
app.layout = html.Div(
    className="page",
    children=[
        html.Div(                       
            className="panel-row",
            children=[
                make_panel("A", "Ask AI:", "Will my eyesight improve after an injection?"),
                make_panel("B", "Automated CDR for Glaucoma Detection", ""),
            ],
        ),
    ],
)

# -------- Validation helper --------
# def validate_inputs(age, hba1c, thickness, snellen_num, snellen_den):
def validate_inputs(age, hba1c, thickness, snellen_den):    
    """
    Returns (ok: bool, errors: list[str]).
    Rules:
      - Age in 1..99
      - HbA1c in 3.0..20.0   (adjust if you prefer)
      - Macular thickness in 100..1200 µm (adjust if you prefer)
      - Snellen: numerator and denominator must be BOTH filled or BOTH empty.
                 If both filled: each > 0 and reasonable bounds.
    """
    errs = []

    # Age
    if age is not None and not (21 <= float(age) <= 99):
        errs.append("Age must be between 21 and 99.")

    # HbA1c
    if hba1c is not None and not (1.0 <= float(hba1c) <= 20.0):
        errs.append("HbA1c must be between 1.0 and 20.0.")

    # Macular thickness
    if thickness is not None and not (50 <= float(thickness) <= 1200):
        errs.append("Macular thickness must be between 50 and 1200 μm.")

    # VA
    # if snellen_num !=6:
    #     errs.append("Expected Snellen numerator is 6.")
    if snellen_den is not None and not (3 <= float(snellen_den) <= 60):
        errs.append("Snellen denominator should be between 3 and 60.")             

    # Snellen pair (num/den)
    # num_filled = snellen_num is not None and str(snellen_num) != ""
    # den_filled = snellen_den is not None and str(snellen_den) != ""

    # if num_filled ^ den_filled:
    #     errs.append("Snellen fields must both be filled or both left empty.")
    # elif num_filled and den_filled:
    #     try:
    #         n = float(snellen_num)
    #         d = float(snellen_den)
    #         if n <= 0 or d <= 0:
    #             errs.append("Snellen numerator and denominator must be positive numbers.")
    #         # if not (1 <= n <= 24):
    #         #     errs.append("Snellen numerator should be between 3 and 24.")
    #         if not (3 <= d <= 60):
    #             errs.append("Snellen denominator should be between 3 and 60.")
    #     except Exception:
    #         errs.append("Snellen entries must be numeric.")

    return (len(errs) == 0, errs)


# Callbacks (same logic bound to each panel) 
def register_callbacks(tag: str):
    @app.callback(
        Output(f"result_{tag}", "children"),
        [
            Input(f"submit_{tag}", "n_clicks"),
            Input(f"clear_{tag}", "n_clicks"),
        ],
        [
            State(f"age_{tag}", "value"),
            State(f"hba1c_{tag}", "value"),
            State(f"thickness_{tag}", "value"),
            # State(f"snellen_num_{tag}", "value"),
            State(f"snellen_den_{tag}", "value"),
        ],
        prevent_initial_call=True,
    )
    # def _handle(submit_clicks, clear_clicks, age, hba1c, thickness, snellen_num, snellen_den):
    def _handle(submit_clicks, clear_clicks, age, hba1c, thickness, snellen_den):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        which = ctx.triggered[0]["prop_id"].split(".")[0]

        # Clear inputs
        if which == f"clear_{tag}":
            return ""
        
        # Submit → validate before predicting
        # ok, errs = validate_inputs(age, hba1c, thickness, snellen_num, snellen_den)
        ok, errs = validate_inputs(age, hba1c, thickness, snellen_den)
        if not ok:
            # Show error block and DO NOT run prediction
            return html.Div(
                [
                    html.Div("Inputs are not valid. Please fix the following:", className="result-msg error"),
                    html.Ul([html.Li(e) for e in errs], className="error-list")
                ]
            )
        
        if snellen_den is None:
            logmar = None 
        else:     
            snellen_num = 6
        # if snellen_den is not None and snellen_num is not None:
            logmar = float(math.log10(snellen_den / snellen_num))  
        # else:
        #     logmar = None               

        msg1, msg2 = update_outputs(
            submit_clicks, age, hba1c, thickness, logmar)
        
        return [msg1, msg2]



def update_outputs(n_clicks_submit, age, HbA1c, pre_macular_thickness, pre_logmar):
    
    # Only make a prediction if Enter is pressed (n_clicks > 0)

    if n_clicks_submit == 0:
        return "", ""     
    
    # Check if ALL inputs are empty
    if all(v is None for v in [age, HbA1c, pre_macular_thickness, pre_logmar]):
        return (
            html.Div("⚠️ Please fill in at least one input field.", className="result-msg"),
            ""
        )

    # Fill missing inputs with mean values
    input_data = {
        'age': age if age is not None else round(df['age'].mean(), 0),
        'HbA1c': HbA1c if HbA1c is not None else df['pre_HbA1c'].mean(),
        'pre_macular_thickness': pre_macular_thickness if pre_macular_thickness is not None else df['pre_thick'].mean(),
        'pre_logmar': pre_logmar if pre_logmar is not None else df['pre_logmar'].mean(),
    }

    # Prepare the input features (use the input values or the mean if None)
    features = pd.DataFrame([[
        input_data['age'],
        input_data['HbA1c'],
        df['f1'].mean(),  
        df['f2'].mean(),
        df['f3'].mean(),
        df['f4'].mean(),
        input_data['pre_macular_thickness'],
        input_data['pre_logmar']
    ]], columns=['age', 'pre_HbA1c', 'f1', 'f2', 'f3', 'f4', 'pre_thick', 'pre_logmar'])  
    
    features_diff_logmar = pd.DataFrame([[
        input_data['age'],
        input_data['HbA1c'],
        df['f1'].mean(),  
        df['f2'].mean(),
        df['f3'].mean(),
        df['f4'].mean(),
        input_data['pre_macular_thickness']
    ]], columns=['age', 'pre_HbA1c', 'f1', 'f2', 'f3', 'f4', 'pre_thick'])  

    # Make predictions using the Random Forest model
    predicted_logmar = float(rf_logmar.predict(features))
    predicted_thick = float(rf_thick.predict(features))  

    # predicted_logmar = round(predicted_logmar, 2)
    # predicted_logmar = [predicted_logmar if (predicted_logmar*100)%2==0 else predicted_logmar+0.01]
    # predicted_va = snellen_lkup.at[predicted_logmar, 'Snellen']
    closest_idx = snellen_lkup.index.get_indexer([predicted_logmar], method="nearest")[0]
    predicted_va = snellen_lkup.iloc[closest_idx]['Snellen']
     
    # Prepare the text outputs
    prediction_text1 = f"The predicted Snellen visual acuity is {predicted_va}."  
    prediction_text2 = f"The predicted post-treatment thickness is {predicted_thick:.2f}um."

    # Prepare the text outputs as two cards:
    msg1 = html.Div([
        html.Div("The predicted Snellen visual acuity is", className="result-label"),
        html.Div(f"{predicted_va}", className="result-value")
    ], className="result-msg")

    msg2 = html.Div([
        html.Div("The predicted post-treatment thickness is", className="result-label"),
        html.Div(f"{predicted_thick:.2f} µm", className="result-value")
    ], className="result-msg")

    return msg1, msg2
    

# register for both A and B panels
register_callbacks("A")
register_callbacks("B")


if __name__ == "__main__":
    app.run(debug=False)
