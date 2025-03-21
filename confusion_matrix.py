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
    "arrival_date_month": "/home/ies/ashri/RIXA-2/rixaplugin/rixaplugin/label_encoder_arrival_date_month.pkl",
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
datapoints_list = [16505, 2342, 11486, 234, 23171, 4225, 12117, 10677, 14757, 3364, 332, 7788, 890, 19666, 761, 22828, 12571, 8921, 6001, 17964]
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
    "deposit_type"
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
    y_test = pd.read_csv(y_test_path.get(), header=None).values.ravel()
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
        'G': 'Penthouse Suite', 'H': 'Junior Suite', 'I': 'Luxury Room',
        'J': 'Presidential Suite'
    }
    
    for feature, encoder in label_encoders.items():
        if feature in decoded_data:
            decoded_data[feature] = encoder.inverse_transform(decoded_data[feature].astype(int))
    
    # Apply room type mapping for display
    if 'reserved_room_type' in decoded_data:
        decoded_data['reserved_room_type'] = decoded_data['reserved_room_type'].map(room_type_mapping).fillna(decoded_data['reserved_room_type'])

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
    """
    global current_datapoint_id

    # true target label
    true_target = 'Canceled' if y_test[current_datapoint_id] == 0 else 'Check-Out'

    # Extract features, decode them, and perform scaling for model input
    features_data = X_test_unscaled.iloc[current_datapoint_id][features]
    decoded_features_data = decode_features(features_data.to_frame().T).iloc[0]  # Decode feature values
    scaled_features = scaler.transform(features_data.values.reshape(1, -1))
    scaled_features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)

    #  model predictions and confidence
    with torch.no_grad():
        output = model(scaled_features_tensor)
        confidence_score = torch.nn.functional.softmax(output, dim=1).max().item() * 100
        confidence_score = round(confidence_score, 2)
        predicted_class = output.argmax().item()

    # predicted class to label
    predicted_class_str = 'Canceled' if predicted_class == 0 else 'Check-Out'

    # Prepare JSON structure 
    datapoint_info = {
        "role": "datapoint",
        "content": {
            "prediction": predicted_class_str,
            "confidence": confidence_score,
            "true_target": true_target,
            "data": {
                format_feature_name(feature): {"title": format_feature_name(feature), "value": decoded_features_data[feature]}
                for feature in display_order
            }
        }
    }

    # explanation text
    explanation = f"The confidence score of this datapoint is {confidence_score}%. "
    explanation += f"This means the model is {confidence_score}% confident that the booking will result in a {predicted_class_str.lower()}."
    explanation += "\nFeature values:\n"
    for feature in display_order:
        explanation += f"- {format_feature_name(feature)}: {decoded_features_data[feature]}\n"

    return explanation.strip(), datapoint_info



@plugfunc()
def reset(datapoint_id=None):
    """
    Resets the current datapoint index to the beginning of the datapoints_list,
    so that the next call to next_datapoint() will start from the first datapoint.
    Whenever reset is called explanation and datapoint_info is updated each time make sure it shows data of updated current datapoint.
    """
    global current_datapoint_index, current_datapoint_id
    
    if datapoint_id is not None:
        # If datapoint_id is provided, find its index in the datapoints_list
        if datapoint_id in datapoints_list:
            current_datapoint_index = datapoints_list.index(datapoint_id)
        else:
            raise ValueError(f"Datapoint ID {datapoint_id} not found in datapoints_list.")
    else:
        # Reset to the initial state
        current_datapoint_index = 0
    current_datapoint_id = datapoints_list[current_datapoint_index]
    
    # Generate datapoint information and explanation
    explanation, datapoint_info = generate_datapoint_info()
    
    # Display the datapoint information as JSON on the frontend
    api.display(custom_msg=json.dumps(datapoint_info, ensure_ascii=True))
    your_logger.info("Datapoint index has been reset to the first entry in datapoints_list.")
    
    # Update the frontend banner with settings
    settings_dic = {"role": "global_settings", "content": {"show_banner": True}}
    api.display(custom_msg=json.dumps(settings_dic, ensure_ascii=True))
    your_logger.info("Frontend banner updated.")
    
    return explanation.strip()

@plugfunc()
def show_datapoint(datapoint_id=None):
    global current_datapoint_index, current_datapoint_id
    # Move to the next datapoint
    if datapoint_id is not None:
        # If datapoint_id is provided, find its index in the datapoints_list
        if datapoint_id in datapoints_list:
            current_datapoint_index = datapoints_list.index(datapoint_id)
        else:
            raise ValueError(f"Datapoint ID {datapoint_id} not found in datapoints_list.")
    current_datapoint_id = datapoints_list[current_datapoint_index]

    # Generate datapoint information and explanation
    explanation, datapoint_info = generate_datapoint_info()
    
    # Display the datapoint information as JSON on the frontend
    api.display(custom_msg=json.dumps(datapoint_info, ensure_ascii=True))
    your_logger.info(f"Datapoint information displayed on frontend for datapoint id: {current_datapoint_id}")
    
    # Update the frontend banner with settings
    settings_dic = {"role": "global_settings", "content": {"show_banner": True}}
    api.display(custom_msg=json.dumps(settings_dic, ensure_ascii=True))
    your_logger.info("Frontend banner updated.")
    
    return explanation.strip()


@plugfunc()
def next_datapoint(datapoint_id=None, username="", datapoint_choice="",**kwargs ):
    """
    Moves to the next datapoint and updates the explanation for the chatbot.
    Whenever this function is called or confirm button is clicked explanation and datapoint_info is updated each time make sure it shows data of updated current datapoint.
    """
    global current_datapoint_index, current_datapoint_id

    # Move to the next datapoint
    if datapoint_id is not None:
        # If datapoint_id is provided, set the current index based on it
        if datapoint_id in datapoints_list:
            current_datapoint_index = datapoints_list.index(datapoint_id)
        else:
            raise ValueError(f"Datapoint ID {datapoint_id} not found in datapoints_list.")
    else:
        # Default behavior: Move to the next datapoint
        current_datapoint_index = (current_datapoint_index + 1) % len(datapoints_list)

    current_datapoint_id = datapoints_list[current_datapoint_index]

    # Generate datapoint information and explanation
    explanation, datapoint_info = generate_datapoint_info()
    
    # Display the datapoint information as JSON on the frontend
    api.display(custom_msg=json.dumps(datapoint_info, ensure_ascii=True))
    your_logger.info(f"Datapoint information displayed on frontend for datapoint id: {current_datapoint_id}")
    
    # Update the frontend banner with settings
    settings_dic = {"role": "global_settings", "content": {"show_banner": True}}
    api.display(custom_msg=json.dumps(settings_dic, ensure_ascii=True))
    your_logger.info("Frontend banner updated.")

     # Define the file path
    log_file = "/home/ies/ashri/selections/ashri.txt"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Log the selection with a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Username: {username}, Datapoint Choice: {datapoint_choice}\n"

    with open(log_file, "a") as f:
        f.write(log_entry)

    your_logger.info(f"Selection logged in ashri.txt: {log_entry.strip()}")
    
    return explanation.strip()


@plugfunc()
def explain_with_lime():
    """
    Generates a LIME explanation for the current datapoint ID.
    """
    global current_datapoint_id

    try:
        # Scale the features for the current datapoint
        features_data = X_test_unscaled.iloc[current_datapoint_id][features].values.reshape(1, -1)
        scaled_features = scaler.transform(features_data)
        your_logger.info(f"Generating LIME explanation for datapoint ID: {current_datapoint_id}")

        # Wrapper function for model predictions compatible with LIME
        def predict_proba(X):
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            model.eval()
            with torch.no_grad():
                probabilities = torch.nn.functional.softmax(model(X_tensor), dim=1).cpu().numpy()
            return probabilities

        # Initialize the LIME explainer on the scaled dataset with modified feature names
        lime_explainer = LimeTabularExplainer(
            scaler.transform(X_test_unscaled[features]), 
            feature_names=[format_feature_name(feature).replace("_", " ") for feature in features],
            class_names=['Canceled', 'Check-Out'],
            mode='classification',
            random_state=42
        )

        # Generate the explanation for the current datapoint
        explanation = lime_explainer.explain_instance(scaled_features[0], predict_proba, num_features=len(features))

        # Format and clean feature names for display
        explanation_list = [(format_feature_name(feature).replace("_", " "), contribution) for feature, contribution in explanation.as_list() if contribution != 0]

        # Remove any numerical ranges, constraints, and equals signs from feature names
        cleaned_feature_names = [re.sub(r"(<|>|<=|>=|==|!=)?\s?-?\d*\.?\d*", "", feature).strip().replace("=", "") for feature, _ in explanation_list]
        contributions = [contribution for _, contribution in explanation_list]

        # Plot only the impactful features with cleaned names
        fig = px.bar(x=contributions, y=cleaned_feature_names, orientation='h', 
                     labels={'x': 'Contribution to Prediction', 'y': 'Feature'},
                     title='LIME Explanation', 
                     color=contributions, 
                     color_continuous_scale=["red", "green"])

        # Generate and display the interactive HTML plot
        html = fig.to_html(include_plotlyjs=False, full_html=False, config={'displayModeBar': False})
        api.display(html="<!--LIME-->"+html)

        # Convert LIME explanation to text for LLM summary
        lime_text_explanation = explanation.as_list()
        text_summary = f"LIME Explanation:\n"
        text_summary += "\n".join([f"{feature.replace('_', ' ')}: {contribution:.4f}" for feature, contribution in lime_text_explanation])

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
        y_test = pd.read_csv(y_test_path.get(), header=None, names=["reservation_status"])

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

        # Select the current datapoint instance to explain
        query_instance = X_test.iloc[[current_datapoint_id]].copy()
        
        # Get model prediction directly using the same approach as in next_datapoint
        features_data = X_test_unscaled.iloc[current_datapoint_id][features]
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
        visual_df['reservation_status'] = visual_df['reservation_status'].replace({1: 'Check-Out', 0: 'Canceled'})

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
        print("Generating feature importance dashboard...")

        # Apply filtering to dataset
        X_test_filtered_df = filter_data(X_test_unscaled.copy(), **kwargs) if kwargs else X_test_unscaled.copy()

        if X_test_filtered_df.empty:
            return "The filtered dataset is empty. Please provide valid filter parameters."

        # Ensure 'reservation_status' is not in the features when scaling
        y_test_filtered = X_test_filtered_df.pop('reservation_status').values

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

        fig = px.bar(x=formatted_sorted_features, y=sorted_shap_values, 
                     labels={'x': 'Feature', 'y': 'Shapley Value'},
                     title='Shapley Values')

        # Convert the Plotly figure to an HTML string
        html = fig.to_html(include_plotlyjs=False, full_html=False, config={'displayModeBar': False})
        api.display(html="<!--SHAP-->"+html)

        # Explanation for LLM with formatted feature names
        explanation = f"""
        Feature Importance Analysis:

        This dashboard visualizes the feature importance based on Shapely values, illustrating which features have the most significant impact on the model's predictions. 
        
        All Feature Importances:
        """ + "\n".join([f"- {feature}: {importance:.4f}" for feature, importance in zip(formatted_sorted_features, sorted_shap_values)]) + "\n\n" + \
        "Higher Shapely values indicate greater influence on the model's predictions."

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
    If no target_class is provided, the user should choose between 'Canceled' (0) or 'Check-Out' (1).

    :param pdp_features: List of features for PDPs (e.g., ["Market Segment", "deposit type"]).
    :param target_class: Target class for PDP (0 = 'Canceled', 1 = 'Check-Out').
    :param kwargs: Optional filter parameters (e.g., lead_time=('>=', 10)).
    :return: Text explanation string (the PDP plot is displayed via api.display).
    """

    try:
        print("Generating PDP Dashboard...")

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

            # If it's an integer feature, build an integer grid
            if feature in INTEGER_FEATURES:
                possible_values = np.arange(min_val, max_val + 1, 1, dtype=float)
                if len(possible_values) > grid_resolution:
                    indices = np.linspace(0, len(possible_values) - 1, grid_resolution, dtype=int)
                    feature_values = possible_values[indices]
                else:
                    feature_values = possible_values
            else:
                # Treat as continuous
                feature_values = np.linspace(min_val, max_val, grid_resolution)

            avg_predictions = np.zeros(len(feature_values))

            # For partial dependence, we set the entire column to 'val'
            # and then compute the average predicted probability for 'target_cls'.
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
        target_label = "Canceled" if target_class == 0 else "Check-Out"
        explanation += f"Partial Dependence Plot for target class '{target_label}':\n"

        for feature, formatted_feature in zip(pdp_features, formatted_pdp_features):
            feature_values, avg_predictions = calculate_partial_dependence(feature, target_class)

            # Build a DataFrame for the Plotly line chart
            df = pd.DataFrame({
                "Feature Value": feature_values,
                "Average Prediction": avg_predictions,
                "Feature": formatted_feature
            })
            plot_data.append(df)

            # We want to show 10 sample points in the textual explanation
            if len(feature_values) > 1:
                explanation_indices = np.linspace(0, len(feature_values) - 1, 10, dtype=int)
            else:
                explanation_indices = [0]

            explanation += f"\nFeature: {formatted_feature}\n"
            min_display = feature_values[0]
            max_display = feature_values[-1]
            explanation += (f"  Range: from {min_display:.2f} to {max_display:.2f}.\n"
                            "  Below are 10 sample points across this range:\n")

            # 5) For each sample point, decode the categorical label if applicable
            for idx in explanation_indices:
                val = feature_values[idx]
                # 5a) For the explanation, decode the feature if we have a label encoder
                #     or room type mapping. We'll do that by creating a 1-row copy 
                #     from X_filtered, setting 'feature' to 'val', then calling decode_features.
                X_explain = X_filtered.iloc[:1].copy()  # single row
                X_explain[feature] = val

                # decode_features will convert numeric codes to strings (e.g. country code -> "Portugal")
                X_explain_decoded = decode_features(X_explain)
                decoded_val = X_explain_decoded[feature].iloc[0]

                # 5b) If it's an integer feature but not label-encoded, we may want to show it as int
                #     But decode_features will leave it numeric if there's no label encoder
                #     so we handle fallback formatting if it's still numeric
                if isinstance(decoded_val, (int, float)) and feature not in label_encoders:
                    if feature in INTEGER_FEATURES:
                        decoded_val = str(int(round(decoded_val)))
                    else:
                        decoded_val = f"{decoded_val:.2f}"

                # 5c) Get the partial dependence average prediction
                pred = avg_predictions[idx]

                explanation += f"   - Feature Value: {decoded_val}, Average Prediction: {pred:.4f}\n"

        # 6) If no data was built, return an error
        if not plot_data:
            return "No valid features provided."

        # 7) Combine data for the plot
        full_df = pd.concat(plot_data, ignore_index=True)

        # 8) Plotly line chart
        fig = px.line(
            full_df,
            x="Feature Value",
            y="Average Prediction",
            color="Feature",
            title=f"Partial Dependence Plot for [{', '.join(formatted_pdp_features)}] ({target_label})"
        )

        # 9) Display the plot in the UI
        html = fig.to_html(include_plotlyjs=False, full_html=False, config={'displayModeBar': False})
        api.display(html="<!--PDP-->" + html)

        # 10) Return the textual explanation
        return explanation.strip()

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        api.display(html=f"<h3>{error_message}</h3>")
        return error_message

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
def generate_histogram(feature_name, split_by_target=False):
    """
    Generates a histogram for the specified feature, with an option to split by target variable.
    
    Args:
        feature_name (str): Name of the feature to generate histogram for.
        split_by_target (bool, optional): Whether to split the histogram by target variable. Defaults to False.
    
    Returns:
        str: Description of the generated histogram
    """
    global X_test_unscaled, y_test
    
    try:
        # Use the test data
        data = X_test_unscaled.copy()
        
        # Check if a specific feature was requested
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
            # Prepare target variable labels
            target_labels = {0: 'Canceled', 1: 'Check-Out'}
            data['target_label'] = data['reservation_status'].map(target_labels)
        
        # Handle categorical features
        if feature_name in label_encoders:
            # Decode the categorical feature
            feature_data = data[feature_name].copy()
            data[feature_name] = pd.Series(label_encoders[feature_name].inverse_transform(feature_data.astype(int)))
            
            if split_by_target:
                # Create grouped bar chart for categorical data with target split
                fig = px.histogram(data, x=feature_name, color='target_label',
                                  title=f'Distribution of {feature_title} by Reservation Status',
                                  labels={'count': 'Count', feature_name: feature_title},
                                  color_discrete_map={'Canceled': 'salmon', 'Check-Out': 'skyblue'},
                                  barmode='group')
            else:
                # Create a count plot for categorical data without target split
                fig = px.histogram(data, x=feature_name, 
                                  title=f'Distribution of {feature_title}',
                                  labels={'count': 'Count', feature_name: feature_title},
                                  color_discrete_sequence=['skyblue'])
            
            # Rotate x-axis labels for better readability for categorical variables
            fig.update_layout(xaxis_tickangle=-45)
        else:
            # Handle numeric features
            if split_by_target:
                # Create overlapping histograms for numeric data with target split
                fig = px.histogram(data, x=feature_name, color='target_label',
                                 title=f'Distribution of {feature_title} by Reservation Status',
                                 labels={'count': 'Count', feature_name: feature_title},
                                 color_discrete_map={'Canceled': 'salmon', 'Check-Out': 'skyblue'},
                                 opacity=0.7)
            else:
                # Create a standard histogram for numeric data without target split
                fig = px.histogram(data, x=feature_name, 
                                 title=f'Distribution of {feature_title}',
                                 labels={'count': 'Count', feature_name: feature_title},
                                 color_discrete_sequence=['skyblue'])
                
                # Add a mean line to numeric histograms
                mean_val = data[feature_name].mean()
                fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                              annotation_text=f"Mean: {mean_val:.2f}", 
                              annotation_position="top right")
        
        # Generate the interactive HTML plot
        html = fig.to_html(include_plotlyjs=False, full_html=False, config={'displayModeBar': False})
        
        # Use a unique identifier for the histogram
        plot_id = f"HISTOGRAM_{feature_name}" + ("_BY_TARGET" if split_by_target else "")
        api.display(html=f"<!--{plot_id}-->{html}")
        
        if split_by_target:
            return f"Generated histogram for {feature_title} split by reservation status (Canceled vs Check-Out)."
        else:
            return f"Generated histogram for {feature_title}."
    
    except Exception as e:
        error_message = f"An error occurred while generating the histogram: {str(e)}"
        your_logger.error(error_message)
        api.display(html=f"<h3>{error_message}</h3>")
        return error_message
