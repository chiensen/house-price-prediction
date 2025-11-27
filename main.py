import pickle

import gradio as gr
import pandas as pd


def checkbox_handler(is_checked):
    output = int(is_checked)
    return output


def predict_house_price(
        bedrooms,
        bathrooms,
        sqft_living,
        sqft_lot,
        floors,
        waterfront,
        view,
        condition,
        grade,
        yr_built,
        yr_renovated,
        lat,
        long):

    ohe = pickle.load(open("model/encoder.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    model = pickle.load(open("model/xgb_r_model.pkl", "rb"))

    input_data = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "grade": grade,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "lat": lat,
        "long": long
    }

    print(input_data)

    X_new = pd.DataFrame([input_data])

    categorical = ['waterfront', 'view']
    numeric = [col for col in X_new.columns if col not in categorical]

    X_new_cat = ohe.transform(X_new[categorical])
    encoded_df = pd.DataFrame(X_new_cat, columns=ohe.get_feature_names_out(categorical))

    numeric_df = X_new.drop(columns=categorical)
    numeric_df[numeric] = scaler.transform(numeric_df[numeric])

    final_df = pd.concat([numeric_df, encoded_df], axis=1)
    print(final_df.columns)

    price_prediction = model.predict(final_df)
    print(price_prediction)

    price = price_prediction[0]

    return "${:0,.2f}".format(price)


with gr.Blocks() as demo:
    _waterfront_var = gr.State(0)

    gr.Markdown("# House Price Prediction in King County, USA")

    with gr.Row():
        with gr.Column():

            gr.Markdown("Location")
            with gr.Row():
                _latitude = gr.Number(value=47.2, label='Latitude', info='deg', minimum=47.15, maximum=47.78,
                                      step=0.1)
                _longitude = gr.Number(value=-122.52, label='Longitude', info='deg', minimum=-122.52, maximum=-121.32,
                                       step=0.1)

            gr.Markdown("Year")
            _year_built = gr.Number(
                value=2015, label='Year Built', info='Year', minimum=1900, maximum=2015, step=1, precision=0)
            _year_renovated = gr.Number(
                value=0, label='Year Renovated', info='Year', minimum=0, maximum=2015, step=1, precision=0)

            gr.Markdown("Floor Areas")
            _living_square_feet = gr.Number(value=500, label="Living Floor Area", info="Square Feet", precision=0)
            _lot_square_feet = gr.Number(value=800, label="Lot Floor Area", info="Square Feet", precision=0)

            gr.Markdown("Floors")
            _floors = gr.Number(value=1.0, label="Floors", info="No.", step=0.5, minimum=1.0, maximum=3.5, precision=1)

        with gr.Column():
            gr.Markdown("Rooms")
            _bedrooms = gr.Number(value=2, label="Bedrooms", info="No.", step=1, minimum=2, precision=0)
            _bathrooms = gr.Number(value=1, label="Bathrooms", info="No.", step=1, minimum=1, precision=0)

            gr.Markdown("Waterfront")
            _waterfront = gr.Checkbox(value=False, label="Waterfront")

            gr.Markdown("Views")
            _views = gr.Number(value=0, label="View", info="No. (1 to 4)", step=1, minimum=0, maximum=4, precision=0)

            gr.Markdown("Condition")
            _condition = gr.Number(value=3, label="House Condition", info="1 to 5", step=1, minimum=1, maximum=5, precision=0)

            gr.Markdown("Grade")
            _grade = gr.Number(value=1, label="House Grade", info="1 to 13", step=1, minimum=1, maximum=13, precision=0)

    with gr.Row():
        _submit_button = gr.Button("Submit")
    with gr.Row():
        _results_box = gr.Textbox(label="Estimated Price")

    _waterfront.change(checkbox_handler, inputs=_waterfront, outputs=_waterfront_var)
    _submit_button.click(
        predict_house_price,
        inputs=(
             _bedrooms,
             _bathrooms,
             _living_square_feet,
             _lot_square_feet,
             _floors,
             _waterfront_var,
             _views,
             _condition,
             _grade,
             _year_built,
             _year_renovated,
             _latitude,
             _longitude),
        outputs=_results_box)


if __name__ == "__main__":
    demo.queue().launch(css="footer {visibility: hidden}", share=False)
