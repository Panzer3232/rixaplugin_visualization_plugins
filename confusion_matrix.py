from rixaplugin.decorators import plugfunc, worker_init
import rixaplugin
import rixaplugin.sync_api as api
import pandas as pd
import numpy as np
import shap
import torch
import torch.nn as nn
import os
import joblib
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import json
import logging
import plotly.express as px
from lime.lime_tabular import LimeTabularExplainer
import re
import dice_ml
from dice_ml import Dice
from dice_ml.model_interfaces.base_model import BaseModel

# Define plugin variables
model_path = rixaplugin.variables.PluginVariable("model_path", str, default="nn_model_hotel_state_dict.pth")
scaler_path = rixaplugin.variables.PluginVariable("scaler_path", str, default="scaler_hotel.pkl")
X_test_path = rixaplugin.variables.PluginVariable("X_test_path", str, default="x_test_hotel.csv")
y_test_path = rixaplugin.variables.PluginVariable("y_test_path", str, default="y_test_hotel.csv")

# Initialize logger for debugging
your_logger = logging.getLogger("rixa.plugin_logger")
your_logger.setLevel(logging.INFO)

# Define the paths for the label encoder files
label_encoder_files = {
    "country": "/home/ies/ashri/RIXA-2/rixaplugin/rixaplugin/label_encoder_country.pkl",
    "deposit_type": "/home/ies/ashri/RIXA-2/rixaplugin/rixaplugin/label_encoder_deposit_type.pkl",
    "hotel": "/home/ies/ashri/RIXA-2/rixaplugin/rixaplugin/label_encoder_hotel.pkl",
    "market_segment": "/home/ies/ashri/RIXA-2/rixaplugin/rixaplugin/label_encoder_market_segment.pkl",
    "reserved_room_type": "/home/ies/ashri/RIXA-2/rixaplugin/rixaplugin/label_encoder_reserved_room_type.pkl"
}

# Global variables to hold model, scaler, and current datapoint information
model = None
scaler = None
X_test_unscaled = None
y_test = None
datapoints_list =  [8723, 10233, 52028, 16948, 16376, 13260, 27378, 22730, 23855, 28133,
    30195, 56168, 60052, 85498, 37601, 55780, 76664, 61754, 68227, 68689,
    71076, 78335, 80601, 81743, 82393, 84583, 96373, 99102, 106685, 111235]

#datapoints_list = [85498, 106685]
current_datapoint_index = 0 
current_datapoint_id = None
features = [
    'deposit_type', 'lead_time', 'country', 'total_of_special_requests', 'adr',
    'arrival_date_week_number', 'market_segment', 'arrival_date_day_of_month',
    'previous_cancellations', 'arrival_date_month', 'stays_in_week_nights',
    'booking_changes', 'stays_in_weekend_nights', 'reserved_room_type',
    'adults', 'hotel', 'children'
]
display_order = [
    'hotel', 'reserved_room_type', 'lead_time', 'deposit_type', 'market_segment',
    'arrival_date_week_number', 'arrival_date_day_of_month', 'arrival_date_month',
    'adults', 'children', 'country',
    'previous_cancellations', 'booking_changes', 'total_of_special_requests',
    'stays_in_week_nights', 'stays_in_weekend_nights',
    'adr'
]

INTEGER_FEATURES = {
    "lead_time",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "previous_cancellations",
    "booking_changes",
    "total_of_special_requests",
    "stays_in_week_nights",
    "stays_in_weekend_nights",
    "adults",
    "children",
    "reserved_room_type",
    "market_segment",
    "country",
    "hotel",
    "deposit_type", 
    "arrival_date_month"
}

label_encoders = {}

def load_label_encoders():
    """
    Loads all label encoders as specified in label_encoder_files and stores them in the label_encoders dictionary.
    """
    for feature, path in label_encoder_files.items():
        label_encoders[feature] = joblib.load(path)

# Load label encoders during initialization
load_label_encoders()

# Device selection: GPU 3 or 4 if available, otherwise CPU
device = torch.device("cuda:3" if torch.cuda.is_available() and torch.cuda.device_count() > 3 else
                      "cuda:4" if torch.cuda.is_available() and torch.cuda.device_count() > 4 else
                      "cpu")
your_logger.info(f"Using device: {device}")

# Define the Neural Network Model class
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(hidden_size3, hidden_size4)
        self.relu4 = nn.ReLU()
        self.layer5 = nn.Linear(hidden_size4, hidden_size5)
        self.relu5 = nn.ReLU()
        self.layer6 = nn.Linear(hidden_size5, hidden_size6)
        self.relu6 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size6, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu1(out)
        out = self.layer2(out)
        out = self.relu2(out)
        out = self.layer3(out)
        out = self.relu3(out)
        out = self.layer4(out)
        out = self.relu4(out)
        out = self.layer5(out)
        out = self.relu5(out)
        out = self.layer6(out)
        out = self.relu6(out)
        out = self.output_layer(out)
        return out
    
    def predict_proba(self, x):
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)

@worker_init()
def initialize_model_and_data():
    """
    Initializes the model, scaler, and test data for the plugin.
    """
    global model, scaler, X_test_unscaled, y_test

    # Load the model and set to evaluation mode
    input_size = len(features)
    hidden_sizes = [64, 128, 256, 512, 256, 128]
    num_classes = 2
    model_instance = NeuralNet(input_size, *hidden_sizes, num_classes)
    model_instance.load_state_dict(torch.load(model_path.get(), map_location=device))
    model_instance.eval().to(device)
    model = model_instance
    your_logger.info("Model loaded and set to evaluation mode.")

    # Load scaler
    scaler = joblib.load(scaler_path.get())
    your_logger.info("Scaler loaded successfully.")

    # Load test data
    X_test_unscaled = pd.read_csv(X_test_path.get())
    y_test = pd.read_csv(y_test_path.get(), header=0).values.ravel()
    X_test_unscaled['reservation_status'] = y_test

    your_logger.info("Test data loaded.")

def decode_features(data):
    """
    Decodes categorical features from their encoded numeric values to original labels using loaded label encoders.
    Only applies decoding to features that have a corresponding label encoder.
    """
    decoded_data = data.copy()
    room_type_mapping = {
        'A': 'Standard Room', 'B': 'Deluxe Room', 'C': 'Family Room',
        'D': 'Suite Room', 'E': 'Executive Room', 'F': 'Superior Room',
        'G': 'Penthouse Suite', 'H': 'Junior Suite', 'L': 'Luxury Room',
        'P': 'Presidential Suite'
    }
    
    int_to_month = {
        0: 'January', 1: 'February', 2: 'March', 3: 'April',
        4: 'May', 5: 'June', 6: 'July', 7: 'August',
        8: 'September', 9: 'October', 10: 'November', 11: 'December'
    }

    # Inverse-transform for label-encoded features
    for feature, encoder in label_encoders.items():
        if feature in decoded_data:
            decoded_data[feature] = encoder.inverse_transform(decoded_data[feature].astype(int))
    
    # room type mapping for display
    if 'reserved_room_type' in decoded_data:
        decoded_data['reserved_room_type'] = (
            decoded_data['reserved_room_type']
            .map(room_type_mapping)
            .fillna(decoded_data['reserved_room_type'])
        )

    #Decode month codes 
    if 'arrival_date_month' in decoded_data:
        decoded_data['arrival_date_month'] = (
            decoded_data['arrival_date_month']
            .map(int_to_month)
            .fillna(decoded_data['arrival_date_month'])
        )

    return decoded_data

def filter_data(data, **kwargs):
    """
    Filters the dataset based on specified criteria. Supports filtering by features and target variable.
    
    :param data: The DataFrame to filter.
    :param kwargs: Conditions for filtering (e.g., lead_time=('>=', 10), hotel=1).
    :return: Filtered DataFrame.
    """
    status_map = {'Canceled': 0, 'Check-Out': 1}
    for key, value in kwargs.items():
        if key in data.columns:
            if key == 'reservation_status' and isinstance(value, str):
                value = status_map.get(value, value)
            if isinstance(value, tuple) and len(value) == 2:
                op, val = value
                if op == '>=':
                    data = data[data[key] >= val]
                elif op == '<=':
                    data = data[data[key] <= val]
                elif op == '>':
                    data = data[data[key] > val]
                elif op == '<':
                    data = data[data[key] < val]
                elif op == '==':
                    data = data[data[key] == val]
            else:
                data = data[data[key] == value]
    return data


# Define a DiCE-compatible model wrapper
class CustomPyTorchModel(BaseModel):
    def __init__(self, predict_fn, backend='PYT'):
        super().__init__(model=None, backend=backend)
        self.predict_fn = predict_fn

    def get_output(self, input_instance, transform_data=False):
        return self.predict_fn(input_instance)


def format_feature_name(name):
    formatted_name = name.replace("_", " ").title()
    if name.lower() == "adr": 
        return "ADR"
    return formatted_name  

def generate_datapoint_info():
    """
    Generates the explanation and JSON structure for the current datapoint including confidence score and feature values.
    Interprets current_datapoint_id as an 'original_id' in X_test_unscaled.
    """
    global current_datapoint_id

    # 1) Find the row in X_test_unscaled where original_id matches current_datapoint_id
    matching_rows = X_test_unscaled.index[X_test_unscaled["original_id"] == current_datapoint_id]
    if len(matching_rows) == 0:
        raise ValueError(f"No row found with original_id={current_datapoint_id}")
    row_idx = matching_rows[0]

    # 2) true target label from y_test
    #    We also need to use row_idx here so we match the same row
    true_target = 'Canceled' if y_test[row_idx] == 0 else 'Check-Out'

    # 3) Extract features from that row, decode them, scale them for model input
    features_data = X_test_unscaled.loc[row_idx, features]
    decoded_features_data = decode_features(features_data.to_frame().T).iloc[0]
    scaled_features = scaler.transform(features_data.values.reshape(1, -1))
    scaled_features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)

    # 4) Model predictions and confidence
    with torch.no_grad():
        output = model(scaled_features_tensor)
        confidence_score = torch.nn.functional.softmax(output, dim=1).max().item() * 100
        confidence_score = round(confidence_score, 2)
        predicted_class = output.argmax().item()

    predicted_class_str = 'Cancellation' if predicted_class == 0 else 'Check-Out'

    # 5) Prepare JSON structure
    datapoint_info = {
        "role": "datapoint",
        "survey": {
                "title": "Survey",
                "questions": [
                      {
                        "id": "q1",
                        "text": "How confident are you in your decision?",
                        "scaleMin": 1,
                        "scaleMax": 5,
                        "labelMin": "not confident at all",
                        "labelMax": "very confident",
                      },
                      {
                        "id": "q2",
                        "text": "How useful did you find the AI assistant's responses?",
                        "scaleMin": 1,
                        "scaleMax": 5,
                        "labelMin": "not useful at all",
                        "labelMax": "very useful",
                      },
                      {
                        "id": "q3",
                        "text": "How satisfied were you with the overall support from the AI assistant?",
                        "scaleMin": 1,
                        "scaleMax": 5,
                        "labelMin": "not satisfied at all",
                        "labelMax": "very satisfied",
                      },
                ],
            },
        "content": {
            "prediction": predicted_class_str,
            "confidence": confidence_score,
            "true_target": true_target,
            "data": {
                format_feature_name(feature): {
                    "title": format_feature_name(feature),
                    "value": decoded_features_data[feature]
                }
                for feature in display_order
            },
            "description":"Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.",
            
        }
    }

    # 6) Explanation text
    explanation = f"The confidence score of this datapoint is {confidence_score}%. "
    explanation += f"This means the model is {confidence_score}% confident that the booking will result in a {predicted_class_str.lower()}."
    explanation += "\nFeature values:\n"
    for feature in display_order:
        explanation += f"- {format_feature_name(feature)}: {decoded_features_data[feature]}\n"

    return explanation.strip(), datapoint_info

def unify_feature_name(name: str) -> str:
    """
    Converts a user-typed feature name into a canonical format 
    that matches the dataset's column names.
    E.g., "Market Segment" -> "market_segment"
          "Deposit Type"   -> "deposit_type"
    """
    result = name.strip().lower().replace(" ", "_")
    while "__" in result:
        result = result.replace("__", "_")
    return result


@plugfunc()
def reset(datapoint_id=None):
    """
    Resets the current datapoint index to the beginning of the datapoints_list,
    so that the next call to next_datapoint() will start from the first datapoint.
    If datapoint_id is provided, jumps directly to that original_id in datapoints_list.
    """
    global current_datapoint_index, current_datapoint_id
    
    if datapoint_id is not None:
        if datapoint_id in datapoints_list:
            current_datapoint_index = datapoints_list.index(datapoint_id)
        else:
            raise ValueError(f"Datapoint ID {datapoint_id} not found in datapoints_list.")
    else:
        current_datapoint_index = 0

    current_datapoint_id = datapoints_list[current_datapoint_index]
    
    # Generate datapoint information and explanation
    explanation, datapoint_info = generate_datapoint_info()
    
    # Display the datapoint information as JSON on the frontend
    api.display(custom_msg=json.dumps(datapoint_info, ensure_ascii=True))
    your_logger.info("Datapoint index has been reset or updated based on 'datapoint_id' argument.")
    
    # Update the frontend banner with settings
    settings_dic = {"role": "global_settings", "content": {"show_banner": True}}
    api.display(custom_msg=json.dumps(settings_dic, ensure_ascii=True))
    
    return explanation.strip()

@plugfunc()
def show_datapoint(datapoint_id=None):
    global current_datapoint_index, current_datapoint_id
    
    if datapoint_id is not None:
        if datapoint_id in datapoints_list:
            current_datapoint_index = datapoints_list.index(datapoint_id)
        else:
            raise ValueError(f"Datapoint ID {datapoint_id} not found in datapoints_list.")
    current_datapoint_id = datapoints_list[current_datapoint_index]

    explanation, datapoint_info = generate_datapoint_info()
    
    api.display(custom_msg=json.dumps(datapoint_info, ensure_ascii=True))
    your_logger.info(f"Datapoint information displayed for original_id: {current_datapoint_id}")
    
    settings_dic = {"role": "global_settings", "content": {"show_banner": True}}
    api.display(custom_msg=json.dumps(settings_dic, ensure_ascii=True))
    
    return explanation.strip()


@plugfunc()
def next_datapoint(datapoint_id=None, username="", datapoint_choice="",answers="", **kwargs):
    """
    Moves to the next datapoint (or a specified datapoint), 
    and updates the explanation for the chatbot.
    """
    global current_datapoint_index, current_datapoint_id

    if datapoint_id is not None:
        if datapoint_id in datapoints_list:
            current_datapoint_index = datapoints_list.index(datapoint_id)
        else:
            raise ValueError(f"Datapoint ID {datapoint_id} not found in datapoints_list.")
    else:
        current_datapoint_index = (current_datapoint_index + 1) % len(datapoints_list)

    current_datapoint_id = datapoints_list[current_datapoint_index]

    explanation, datapoint_info = generate_datapoint_info()
    
    api.display(custom_msg=json.dumps(datapoint_info, ensure_ascii=True))
    your_logger.info(f"Datapoint information displayed for original_id: {current_datapoint_id}")

    settings_dic = {"role": "global_settings", "content": {"show_banner": True}}
    api.display(custom_msg=json.dumps(settings_dic, ensure_ascii=True))

    log_file = "/home/ies/ashri/selections/ashri.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Username: {username}, Datapoint Choice: {datapoint_choice}, Answers:{answers}\n"

    with open(log_file, "a") as f:
        f.write(log_entry)

    your_logger.info(f"Selection logged in ashri.txt: {log_entry.strip()}")
    
    return explanation.strip()


@plugfunc()
def explain_with_lime():
    """
    Generates a LIME explanation for the current datapoint (original_id) in X_test_unscaled.
    """
    global current_datapoint_id

    try:
        # 1) Find the row where 'original_id' == current_datapoint_id
        matching_rows = X_test_unscaled.index[X_test_unscaled["original_id"] == current_datapoint_id]
        if len(matching_rows) == 0:
            raise ValueError(f"No row found with original_id={current_datapoint_id}")
        row_idx = matching_rows[0]

        # 2) Extract the features for that row
        features_data = X_test_unscaled.loc[row_idx, features].values.reshape(1, -1)

        # 3) Scale the features for LIME
        scaled_features = scaler.transform(features_data)
        your_logger.info(f"Generating LIME explanation for original_id: {current_datapoint_id}")

        # 4) Define a wrapper function for model predictions (compatible with LIME)
        def predict_proba(X):
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            model.eval()
            with torch.no_grad():
                probabilities = torch.nn.functional.softmax(model(X_tensor), dim=1).cpu().numpy()
            return probabilities

        # 5) Build the background dataset for LIME from all rows in X_test_unscaled
        #    (scaled version)
        background_data = scaler.transform(X_test_unscaled[features])
        lime_explainer = LimeTabularExplainer(
            background_data,
            feature_names=[format_feature_name(feature).replace("_", " ") for feature in features],
            class_names=['Cancelation', 'Check-Out'],
            mode='classification',
            random_state=42
        )

        # 6) Generate the explanation for the single datapoint
        explanation = lime_explainer.explain_instance(scaled_features[0], predict_proba, num_features=len(features))

        # 7) Format and clean feature names for display
        explanation_list = [
            (format_feature_name(feature).replace("_", " "), contribution)
            for feature, contribution in explanation.as_list() if contribution != 0
        ]
        # Remove any numerical constraints from the feature names
        cleaned_feature_names = [
            re.sub(r"(<|>|<=|>=|==|!=)?\s?-?\d*\.?\d*", "", feat).strip().replace("=", "")
            for feat, _ in explanation_list
        ]
        contributions = [contrib for _, contrib in explanation_list]

        # 8) Plot only the impactful features with cleaned names
        fig = px.bar(
            x=contributions,
            y=cleaned_feature_names,
            orientation='h',
            labels={'x': 'Contribution to Prediction', 'y': 'Feature'},
            title='LIME Explanation',
            color=contributions,
            color_continuous_scale=["red", "green"]
        )
        fig.update_layout(dragmode=False)

        # Disable mouse scroll zoom and remove zoom/pan buttons in the mode bar
        config_dict = {
          'displayModeBar': False,
          'scrollZoom': False,
          'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
        }

        # 9) Generate and display the interactive HTML plot
        html = fig.to_html(include_plotlyjs=False, full_html=False, config=config_dict)
        api.display(html="<!--LIME-->" + html)

        # 10) Convert LIME explanation to text for LLM summary
        lime_text_explanation = explanation.as_list()
        text_summary = "LIME Explanation:\n"
        text_summary += "\n".join([f"{feat.replace('_', ' ')}: {val:.4f}" for feat, val in lime_text_explanation])

        return text_summary

    except Exception as e:
        error_message = f"An error occurred in LIME explanation: {str(e)}"
        api.display(html=f"<h3>{error_message}</h3>")
        return error_message


@plugfunc()
def generate_counterfactual_explanations(num_counterfactuals: int = 5):
    """
    Generates counterfactual explanations for the current datapoint in the test dataset.
    """
    global current_datapoint_id, scaler, model, features

    try:
        your_logger.info(f"Generating counterfactual explanations for datapoint ID: {current_datapoint_id}")

        # Load test data
        X_test = pd.read_csv(X_test_path.get())
        y_test = pd.read_csv(y_test_path.get(), header=0, names=["reservation_status"])
        
        # 2) Find the row in X_test that has original_id == current_datapoint_id
        matching_rows = X_test.index[X_test["original_id"] == current_datapoint_id]
        if len(matching_rows) == 0:
            raise ValueError(f"No row found in X_test with original_id={current_datapoint_id}")
        row_idx = matching_rows[0]

        # 3) Build the query instance from that row
        query_instance = X_test.iloc[[row_idx]].copy()

        # 4) For the predicted class, we also look up the row in X_test_unscaled
        matching_rows_unscaled = X_test_unscaled.index[X_test_unscaled["original_id"] == current_datapoint_id]
        if len(matching_rows_unscaled) == 0:
            raise ValueError(f"No row found in X_test_unscaled with original_id={current_datapoint_id}")
        row_idx_unscaled = matching_rows_unscaled[0]

        # Define continuous and categorical features
        continuous_features = ['lead_time', 'adr']
        categorical_features = [
            'deposit_type', 'country', 'total_of_special_requests', 'arrival_date_week_number',
            'market_segment', 'previous_cancellations', 'booking_changes', 'reserved_room_type',
            'adults', 'hotel', 'children', 'arrival_date_day_of_month', 'arrival_date_month',
            'stays_in_week_nights', 'stays_in_weekend_nights'
        ]

        # Define the prediction function using the global scaler
        def predict_fn(x):
            # Use only the features the model expects
            x_features = x[features].copy()
            # Use the global scaler instead of the local one
            x_tensor = torch.tensor(scaler.transform(x_features.values), dtype=torch.float32).to(device)
            with torch.no_grad():
                probs = model.predict_proba(x_tensor).cpu().numpy()
            return probs

        # Initialize DiCE data interface
        d = dice_ml.Data(
            dataframe=pd.concat([X_test[features], y_test], axis=1),
            continuous_features=continuous_features,
            outcome_name='reservation_status'
        )
        custom_dice_model = CustomPyTorchModel(predict_fn=predict_fn, backend="PYT")
        custom_dice_model.model_type = 'classifier'
        exp = Dice(d, custom_dice_model)
        
        # Extract the features from X_test_unscaled for model prediction
        features_data = X_test_unscaled.loc[row_idx_unscaled, features]
        scaled_features_tensor = torch.tensor(scaler.transform(features_data.values.reshape(1, -1)), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            output = model(scaled_features_tensor)
            predicted_class = output.argmax().item()

        # Generate counterfactuals with features the model expects
        counterfactuals = exp.generate_counterfactuals(query_instance[features], total_CFs=num_counterfactuals, desired_class="opposite")

        # Process and display counterfactuals
        counterfactuals_data = counterfactuals.cf_examples_list[0].final_cfs_df
        
        # Use model prediction instead of true label
        counterfactuals_data['reservation_status'] = 1 - predicted_class
        
        # Create original instance with the model's prediction
        original_instance = query_instance[features].copy()
        original_instance['reservation_status'] = predicted_class

        # Decode features for display
        original_instance_decoded = decode_features(original_instance)
        counterfactuals_data_decoded = decode_features(counterfactuals_data)

        # Combine decoded original and counterfactual data for display
        visual_df = pd.concat([original_instance_decoded, counterfactuals_data_decoded], ignore_index=True)

        # Replace `1` and `0` in `reservation_status` with `Check-Out` and `Canceled`
        visual_df['reservation_status'] = visual_df['reservation_status'].replace({1: 'Check-Out', 0: 'Cancelation'})

        # Add row labels
        row_labels = ['Original Instance'] + [f'Counterfactual {i}' for i in range(1, len(visual_df))]
        visual_df.insert(0, 'Instance', row_labels)

        # Apply `format_feature_name` to all feature column names
        visual_df.columns = [format_feature_name(col) if col != "Instance" else col for col in visual_df.columns]

        # Replace matching values in counterfactuals with '-'
        for col in visual_df.columns[1:]:
            for i in range(1, len(visual_df)):
                if visual_df.loc[i, col] == visual_df.loc[0, col]:
                    visual_df.loc[i, col] = "-"

        # Filter out columns where all counterfactual rows contain only dashes
        columns_to_keep = []
        for col in visual_df.columns[1:]:
            if visual_df.loc[1:, col].ne("-").any():
                columns_to_keep.append(col)
        
        visual_df = visual_df[columns_to_keep]

        html_table = visual_df.to_html(index=False, escape=False, border=1)
        styled_html_table = f"""
        <style>
           table {{border-collapse: collapse; width: 100%;}}
           th, td {{border: 1px solid black; padding: 10px; text-align: center;}}
        </style>
        {html_table}
        """
        api.display(html="<!--COUNTERFACTUAL-->"+styled_html_table)

        # Create explanation text
        explanation_text = f"Here are the counterfactual explanations for data instance:\n\n"
        for i, row in visual_df.iterrows():
            row_type = "Original Instance" if i == 0 else f"Counterfactual {i}"
            changes = [f"{col}: {val}" for col, val in row.items() if val != '-' and col != 'Instance']
            explanation_text += f"\n- **{row_type}**: " + ', '.join(changes)

        return explanation_text.strip()
    
    except Exception as e:
        error_message = f"An error occurred in counterfactual explanation: {str(e)}"
        api.display(html=f"<h3>{error_message}</h3>")
        return error_message



import os

@plugfunc()
def draw_importances_dashboard(**kwargs) -> str:
    """
    Draws a Shapely feature importances dashboard, allowing filtering by feature columns and target variable.
    
    Example usage:
    - draw_importances_dashboard(): Full dataset.
    - draw_importances_dashboard(hotel=1, market_segment=3): Filtered dataset.
    
    :param kwargs: Filter criteria to apply to the dataset.
    :return: Explanation with HTML-embedded bar chart of feature importances for LLM.
    """
    try:
        # Check if saved SHAP plot exists; if so, read and display it.
        SHAP_PLOT_FILE = "shap_plot.html"
        if os.path.exists(SHAP_PLOT_FILE):
            with open(SHAP_PLOT_FILE, "r") as f:
                saved_html = f.read()
            api.display(html="<!--SHAP-->" + saved_html)
            return "Displayed saved SHAP plot."
        
        print("Generating feature importance dashboard...")

        # Apply filtering to dataset
        X_test_filtered_df = filter_data(X_test_unscaled.copy(), **kwargs) if kwargs else X_test_unscaled.copy()
        if X_test_filtered_df.empty:
            return "The filtered dataset is empty. Please provide valid filter parameters."

        # Remove 'reservation_status' from the features when scaling
        y_test_filtered = X_test_filtered_df.pop('reservation_status').values

        # --- Minimal change: if 'original_id' is present, drop it before scaling
        if 'original_id' in X_test_filtered_df.columns:
            X_test_filtered_df.drop('original_id', axis=1, inplace=True)

        # Use only a random sample of 100 records
        if len(X_test_filtered_df) > 100:
            X_test_filtered_df = X_test_filtered_df.sample(n=100, random_state=42)
            y_test_filtered = y_test_filtered[X_test_filtered_df.index]

        # Scale the features using the scaler
        try:
            X_test_scaled = scaler.transform(X_test_filtered_df[features])  # Select only the feature columns for scaling
        except ValueError as e:
            return f"Error during scaling: {e}"

        # Convert to tensor for model input
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

        # Use SHAP for feature importance
        background = X_test_tensor[:80]  # Use the first 80 samples as the background
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(X_test_tensor)

        # Calculate mean absolute SHAP values for feature importance
        shap_mean_abs = np.mean(np.abs(shap_values), axis=(0, 2))
        sorted_idx = np.argsort(shap_mean_abs)[::-1]
        sorted_features = [features[i] for i in sorted_idx]
        sorted_shap_values = shap_mean_abs[sorted_idx]

        # Decode and format feature names
        decoded_sorted_features = decode_features(pd.DataFrame([sorted_features])).iloc[0].tolist()
        formatted_sorted_features = [format_feature_name(feature) for feature in decoded_sorted_features]

        fig = px.bar(
            x=formatted_sorted_features,
            y=sorted_shap_values,
            labels={'x': 'Feature', 'y': 'Shapley Value'},
            title='Shapley Values'
        )

        # Disable drag interactions (thus preventing zoom or pan)
        fig.update_layout(dragmode=False)

        # Disable mouse scroll zoom and remove zoom/pan buttons in the mode bar
        config_dict = {
            'displayModeBar': False,
            'scrollZoom': False,
            'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d',
                                       'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
        }

        html = fig.to_html(include_plotlyjs=False, full_html=False, config=config_dict)
        api.display(html="<!--SHAP-->" + html)

        # Save the generated HTML for future calls
        with open(SHAP_PLOT_FILE, "w") as f:
            f.write(html)

        # Explanation for LLM with formatted feature names
        explanation = f"""
        Feature Importance Analysis:

        This dashboard visualizes the feature importance based on Shapley values, illustrating which features have the most significant impact on the model's predictions.
        
        All Feature Importances:
        """ + "\n".join([f"- {feature}: {importance:.4f}" for feature, importance in zip(formatted_sorted_features, sorted_shap_values)]) + "\n\n" + \
        "Higher Shapley values indicate greater influence on the model's predictions."

        return explanation.strip()

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        api.display(html=f"<h3>{error_message}</h3>")
        return error_message



@plugfunc()
def draw_pdp_dashboard(pdp_features: list, target_class: int = 0, **kwargs) -> str:
    """
    Draws Partial Dependence Plots (PDP) for selected features on the same plot.
    Decodes categorical feature values in the LLM explanation (e.g. "Portugal" instead of "22").
    If no target_class is provided, the user should choose between 'Cancelation' (0) or 'Check-Out' (1).

    :param pdp_features: List of features for PDPs (e.g., ["Market Segment", "deposit type"]).
    :param target_class: Target class for PDP (0 = 'Cancelation', 1 = 'Check-Out').
    :param kwargs: Optional filter parameters (e.g., lead_time=('>=', 10)).
    :return: Text explanation string (the PDP plot is displayed via api.display).
    """

    try:
        print("Generating PDP Dashboard...")

        # 1) Match user-typed feature names to actual columns in 'features'
        validated_pdp_features = []
        for user_feat in pdp_features:
            unified = unify_feature_name(user_feat)
            matched_feature = next(
                (f for f in features if unify_feature_name(f) == unified),
                None
            )
            if matched_feature:
                validated_pdp_features.append(matched_feature)
            else:
                print(f"Invalid feature: {user_feat}")
                return f"The feature '{user_feat}' is not valid."
        pdp_features = validated_pdp_features

        # 2) Apply optional filters to the dataset
        X_filtered = filter_data(X_test_unscaled.copy(), **kwargs) if kwargs else X_test_unscaled.copy()
        if X_filtered.empty:
            print("Filtered dataset is empty.")
            return "The filtered dataset is empty."

        # 3) Prepare human-readable names
        formatted_pdp_features = [format_feature_name(f) for f in pdp_features]
        grid_resolution = 70

        def calculate_partial_dependence(feature: str, target_cls: int):
            """
            Calculates partial dependence for a single numeric feature, returning
            (feature_values, avg_predictions).
            """
            original_values = X_filtered[feature]
            min_val = original_values.min()
            max_val = original_values.max()

            if feature == "market_segment":
               # only codes are only 0–6:
               valid_max = len(label_encoders["market_segment"].classes_) - 2
               max_val = min(max_val, valid_max)

            # build an integer grid
            if feature in INTEGER_FEATURES:
                possible_values = np.arange(int(min_val), int(max_val) + 1, 1, dtype=int)
                if len(possible_values) > grid_resolution:
                    indices = np.linspace(0, len(possible_values) - 1, grid_resolution, dtype=int)
                    feature_values = possible_values[indices]
                else:
                    feature_values = possible_values
            else:
                # Treat as continuous
                feature_values = np.linspace(min_val, max_val, grid_resolution)

            avg_predictions = np.zeros(len(feature_values))

            
            for i, val in enumerate(feature_values):
                X_copy_unscaled = X_filtered.copy()
                X_copy_unscaled[feature] = val
                X_scaled = scaler.transform(X_copy_unscaled[features])
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

                with torch.no_grad():
                    preds = model(X_tensor)
                    preds_softmax = torch.nn.functional.softmax(preds, dim=1)
                    avg_predictions[i] = preds_softmax[:, target_cls].mean().item()

            return feature_values, avg_predictions

        # 4) Build the data for plotting and the explanation
        plot_data = []
        explanation = ""
        target_label = "Cancelation" if target_class == 0 else "Check-Out"
        explanation += f"Partial Dependence Plot for target class '{target_label}':\n"

        for feature, formatted_feature in zip(pdp_features, formatted_pdp_features):
            feature_values, avg_predictions = calculate_partial_dependence(feature, target_class)

            # Decode feature values for categorical features
            decoded_values = []
            for val in feature_values:
                X_explain = X_filtered.iloc[:1].copy() 
                X_explain[feature] = val
                X_explain_decoded = decode_features(X_explain)
                decoded_val = X_explain_decoded[feature].iloc[0]
                
            
                if isinstance(decoded_val, (int, float)) and feature not in label_encoders:
                    if feature in INTEGER_FEATURES:
                        decoded_val = str(int(round(decoded_val)))
                    else:
                        decoded_val = f"{decoded_val:.2f}"
                
                decoded_values.append(decoded_val)

            # Build a for the Plotly line chart with both original and decoded values
            df = pd.DataFrame({
                "Original Value": feature_values,
                "Decoded Value": decoded_values,
                "Average Prediction": avg_predictions,
                "Feature": formatted_feature
            })
            plot_data.append(df)

            # We want to show 12 sample points in the textual explanation
            if len(feature_values) > 1:
                explanation_indices = np.linspace(0, len(feature_values) - 1, 12, dtype=int)
            else:
                explanation_indices = [0]

            explanation += f"\nFeature: {formatted_feature}\n"
            min_display = feature_values[0]
            max_display = feature_values[-1]
            explanation += (f"  Range: from {min_display:.2f} to {max_display:.2f}.\n"
                            "  Below are 12 sample points across this range:\n")

            # 5) For each sample point, use the already decoded values
            for idx in explanation_indices:
                val = feature_values[idx]
                pred = avg_predictions[idx]
                decoded_val = decoded_values[idx]
                
                explanation += f"   - Feature Value: {decoded_val}, Average Prediction: {pred:.4f}\n"

        # 6) If no data was built, return an error
        if not plot_data:
            return "No valid features provided."

        # 7) Combine data for the plot
        full_df = pd.concat(plot_data, ignore_index=True)

        # 8) Create Plotly figure with decoded values for categorical features
        fig = px.line(
            full_df,
            x="Original Value",  
            y="Average Prediction",
            color="Feature",
            title=f"Partial Dependence Plot for [{', '.join(formatted_pdp_features)}] ({target_label})"
        )

        # Update x-axis labels with decoded values
        for feature_name in formatted_pdp_features:
            feature_df = full_df[full_df['Feature'] == feature_name]
            
           
            is_categorical = all(isinstance(val, str) for val in feature_df['Decoded Value']) and not all(
                str(orig) == decoded for orig, decoded in zip(feature_df['Original Value'], feature_df['Decoded Value'])
            )
            
            if is_categorical:
                
                if len(feature_df) > 12:
                    indices = np.linspace(0, len(feature_df) - 1, 12, dtype=int)
                    subset_df = feature_df.iloc[indices]
                else:
                    subset_df = feature_df
                
                # Create a trace-specific axis
                fig.update_traces(
                    x=feature_df['Original Value'],
                    customdata=feature_df['Decoded Value'],
                    hovertemplate='Value: %{customdata}<br>Prediction: %{y:.4f}',
                    selector=dict(name=feature_name)
                )
                
                # Only update x-axis if this is the only feature (otherwise it gets confusing)
                if len(formatted_pdp_features) == 1:
                    fig.update_xaxes(
                        tickmode='array',
                        tickvals=subset_df['Original Value'].tolist(),
                        ticktext=subset_df['Decoded Value'].tolist(),
                        title="Feature Value"
                    )
        
        fig.update_layout(dragmode=False)

        # Disable mouse scroll zoom and remove zoom/pan buttons in the mode bar
        config_dict = {
            'displayModeBar': False,
            'scrollZoom': False,
            'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
        }
        # 9) Display the plot in the UI
        html = fig.to_html(include_plotlyjs=False, full_html=False, config=config_dict)
        api.display(html="<!--PDP-->" + html)

        # 10) Return the textual explanation
        return explanation.strip()

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        api.display(html=f"<h3>{error_message}</h3>")
        return error_message

@plugfunc()
def generate_histogram(feature_name, split_by_target=False):
    """
    Generates a histogram for the specified feature, with an option to split by target variable.
    Returns both an interactive HTML plot and a text summary of histogram values for LLM interpretation.
    
    Args:
        feature_name (str): Name of the feature to generate a histogram for.
        split_by_target (bool, optional): Whether to split the histogram by target variable. Defaults to False.
    
    Returns:
        str: A description of the generated histogram with summary statistics.
    """
    global X_test_unscaled, y_test
    
    try:
        # Use the test data
        data = X_test_unscaled.copy()
        
        # Ensure feature name exists (using a canonical form if needed)
        if not feature_name:
            return "Please specify a feature name. Plotting all features simultaneously is not supported."
        if feature_name not in data.columns:
            feature_name = unify_feature_name(feature_name)
            if feature_name not in data.columns:
                return f"Feature '{feature_name}' not found in the dataset."
        
        # Format feature name for display
        feature_title = format_feature_name(feature_name)
        your_logger.info(f"Generating histogram for feature: {feature_name}, split_by_target: {split_by_target}")
        
        # Add target variable if needed
        if split_by_target:
            data['reservation_status'] = y_test
            target_labels = {0: 'Cancelation', 1: 'Check-Out'}
            data['target_label'] = data['reservation_status'].map(target_labels)
        
        # Apply full decoding: room type mapping, month name conversion, etc.
        data = decode_features(data)
        
        # Specific filtering:
        if feature_name == "market_segment":
            data = data[data[feature_name] != "Undefined"]
        if feature_name == "country":
            data = data[data[feature_name] != "Unknown"]
        
        # Determine if the feature is categorical (dtype object)
        is_categorical = (data[feature_name].dtype == object)
        
        summary_text = []  # For text summary
        
        # Create the plot based on whether the feature is categorical or numeric
        if is_categorical:
            # Build the histogram for categorical features
            if split_by_target:
                # For arrival_date_month, supply a custom order
                if feature_name == "arrival_date_month":
                    fig = px.histogram(
                        data, 
                        x=feature_name, 
                        color='target_label',
                        title=f'Distribution of {feature_title} by Reservation Status',
                        labels={'count': 'Count', feature_name: feature_title},
                        color_discrete_map={'Cancelation': 'salmon', 'Check-Out': 'skyblue'},
                        barmode='group',
                        category_orders={
                            feature_name: [
                                "January", "February", "March", "April", "May", "June",
                                "July", "August", "September", "October", "November", "December"
                            ]
                        }
                    )
                else:
                    fig = px.histogram(
                        data, 
                        x=feature_name, 
                        color='target_label',
                        title=f'Distribution of {feature_title} by Reservation Status',
                        labels={'count': 'Count', feature_name: feature_title},
                        color_discrete_map={'Cancelation': 'salmon', 'Check-Out': 'skyblue'},
                        barmode='group'
                    )
                summary = data.groupby(['target_label', feature_name]).size().reset_index(name='count')
                summary_text.append(f"Distribution of {feature_title} by reservation status:")
                categories = sorted(data[feature_name].unique())
                for category in categories:
                    summary_text.append(f"\n{category}:")
                    for status in ['Cancelation', 'Check-Out']:
                        count = summary[(summary['target_label'] == status) & (summary[feature_name] == category)]['count'].sum()
                        percentage = 100 * count / len(data[data['target_label'] == status])
                        summary_text.append(f"  - {status}: {count} bookings ({percentage:.1f}%)")
            else:
                # Categorical data without splitting by target.
                if feature_name == "arrival_date_month":
                    fig = px.histogram(
                        data, 
                        x=feature_name, 
                        title=f'Distribution of {feature_title}',
                        labels={'count': 'Count', feature_name: feature_title},
                        color_discrete_sequence=['skyblue'],
                        category_orders={
                            feature_name: [
                                "January", "February", "March", "April", "May", "June",
                                "July", "August", "September", "October", "November", "December"
                            ]
                        }
                    )
                else:
                    fig = px.histogram(
                        data, 
                        x=feature_name, 
                        title=f'Distribution of {feature_title}',
                        labels={'count': 'Count', feature_name: feature_title},
                        color_discrete_sequence=['skyblue']
                    )
                summary = data[feature_name].value_counts().sort_index()
                summary_text.append(f"Distribution of {feature_title}:")
                for category, count in summary.items():
                    percentage = 100 * count / len(data)
                    summary_text.append(f"  - {category}: {count} bookings ({percentage:.1f}%)")
            
            # Rotate x-axis labels for better readability
            fig.update_layout(xaxis_tickangle=-45)
        
        else:
            # Handle numeric features
            if split_by_target:
                fig = px.histogram(
                    data, 
                    x=feature_name, 
                    color='target_label',
                    title=f'Distribution of {feature_title} by Reservation Status',
                    labels={'count': 'Count', feature_name: feature_title},
                    color_discrete_map={'Cancelation': 'salmon', 'Check-Out': 'skyblue'},
                    opacity=0.7
                )
                summary_text.append(f"Distribution of {feature_title} by reservation status:")
                for status in ['Cancelation', 'Check-Out']:
                    status_data = data[data['target_label'] == status][feature_name]
                    summary_text.append(f"\n{status}:")
                    summary_text.append(f"  - Count: {len(status_data)}")
                    summary_text.append(f"  - Min: {status_data.min():.2f}")
                    summary_text.append(f"  - Max: {status_data.max():.2f}")
                    summary_text.append(f"  - Mean: {status_data.mean():.2f}")
                    summary_text.append(f"  - Median: {status_data.median():.2f}")
                    summary_text.append(f"  - 25th percentile: {status_data.quantile(0.25):.2f}")
                    summary_text.append(f"  - 75th percentile: {status_data.quantile(0.75):.2f}")
            else:
                fig = px.histogram(
                    data, 
                    x=feature_name, 
                    title=f'Distribution of {feature_title}',
                    labels={'count': 'Count', feature_name: feature_title},
                    color_discrete_sequence=['skyblue']
                )
                mean_val = data[feature_name].mean()
                fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                              annotation_text=f"Mean: {mean_val:.2f}", 
                              annotation_position="top right")
                summary_text.append(f"Distribution of {feature_title}:")
                summary_text.append(f"  - Min: {data[feature_name].min():.2f}")
                summary_text.append(f"  - Max: {data[feature_name].max():.2f}")
                summary_text.append(f"  - Mean: {data[feature_name].mean():.2f}")
                summary_text.append(f"  - Median: {data[feature_name].median():.2f}")
                summary_text.append(f"  - 25th percentile: {data[feature_name].quantile(0.25):.2f}")
                summary_text.append(f"  - 75th percentile: {data[feature_name].quantile(0.75):.2f}")
                hist_values, bin_edges = np.histogram(data[feature_name], bins=10)
                summary_text.append("\nHistogram bins:")
                for i in range(len(hist_values)):
                    bin_start = bin_edges[i]
                    bin_end = bin_edges[i+1]
                    count = hist_values[i]
                    percentage = 100 * count / len(data)
                    summary_text.append(f"  - {bin_start:.2f} to {bin_end:.2f}: {count} bookings ({percentage:.1f}%)")
        
        # For features other than arrival_date_month, use the trace order
        if feature_name != "arrival_date_month":
            fig.update_xaxes(categoryorder="trace")

        fig.update_layout(dragmode=False)
        config_dict = {
            'displayModeBar': False,
            'scrollZoom': False,
            'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d',
                                       'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
        }    
        
        html = fig.to_html(include_plotlyjs=False, full_html=False, config=config_dict)
        plot_id = f"HISTOGRAM_{feature_name}" + ("_BY_TARGET" if split_by_target else "")
        api.display(html=f"<!--{plot_id}-->{html}")
        
        text_summary = "\n".join(summary_text)
        return_message = f"Generated histogram for {feature_title}"
        if split_by_target:
            return_message += " split by reservation status (Cancelation vs Check-Out)."
        else:
            return_message += "."
        
        return f"{return_message}\n\n{text_summary}"
    
    except Exception as e:
        error_message = f"An error occurred while generating the histogram: {str(e)}"
        your_logger.error(error_message)
        api.display(html=f"<h3>{error_message}</h3>")
        return error_message



