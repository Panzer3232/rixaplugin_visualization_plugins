from rixaplugin.decorators import plugfunc, worker_init
import rixaplugin
import rixaplugin.sync_api as api
import pandas as pd
import numpy as np
import shap
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import StandardScaler
import json
import logging
from lime.lime_tabular import LimeTabularExplainer
import re
import dice_ml
from dice_ml import Dice
from dice_ml.model_interfaces.base_model import BaseModel

# plugin variables
model_path = rixaplugin.variables.PluginVariable("model_path", str, default="nn_model_hotel_state_dict.pth")
scaler_path = rixaplugin.variables.PluginVariable("scaler_path", str, default="scaler_hotel.pkl")
X_test_path = rixaplugin.variables.PluginVariable("X_test_path", str, default="x_test_hotel.csv")
y_test_path = rixaplugin.variables.PluginVariable("y_test_path", str, default="y_test_hotel.csv")

# logger for debugging
your_logger = logging.getLogger("rixa.plugin_logger")
your_logger.setLevel(logging.INFO)

# paths for the label encoder files
label_encoder_files = {
    "arrival_date_month": "/home/ies/ashri/RIXA-2/rixaplugin/rixaplugin/label_encoder_arrival_date_month.pkl",
    "country": "/home/ies/ashri/RIXA-2/rixaplugin/rixaplugin/label_encoder_country.pkl",
    "deposit_type": "/home/ies/ashri/RIXA-2/rixaplugin/rixaplugin/label_encoder_deposit_type.pkl",
    "hotel": "/home/ies/ashri/RIXA-2/rixaplugin/rixaplugin/label_encoder_hotel.pkl",
    "market_segment": "/home/ies/ashri/RIXA-2/rixaplugin/rixaplugin/label_encoder_market_segment.pkl",
    "reserved_room_type": "/home/ies/ashri/RIXA-2/rixaplugin/rixaplugin/label_encoder_reserved_room_type.pkl"
}

model = None
scaler = None
X_test_unscaled = None
y_test = None
datapoints_list = [106, 2342, 78, 234, 12, 45, 9876, 10234, 1, 23, 332, 675, 22132, 2329, 761, 442, 5673, 15345, 18221, 991]
current_datapoint_index = -1  
current_datapoint_id = None
features = [
    'deposit_type', 'lead_time', 'country', 'total_of_special_requests', 'adr',
    'arrival_date_week_number', 'market_segment', 'arrival_date_day_of_month',
    'previous_cancellations', 'arrival_date_month', 'stays_in_week_nights',
    'booking_changes', 'stays_in_weekend_nights', 'reserved_room_type',
    'adults', 'hotel', 'children'
]

label_encoders = {}

def load_label_encoders():
    """
    Loads all label encoders as specified in label_encoder_files and stores them in the label_encoders dictionary.
    """
    for feature, path in label_encoder_files.items():
        label_encoders[feature] = joblib.load(path)

load_label_encoders()

# Device selection
device = torch.device("cuda:3" if torch.cuda.is_available() and torch.cuda.device_count() > 3 else
                      "cuda:4" if torch.cuda.is_available() and torch.cuda.device_count() > 4 else
                      "cpu")
your_logger.info(f"Using device: {device}")

# Neural Network Model class
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

    input_size = len(features)
    hidden_sizes = [64, 128, 256, 512, 256, 128]
    num_classes = 2
    model_instance = NeuralNet(input_size, *hidden_sizes, num_classes)
    model_instance.load_state_dict(torch.load(model_path.get(), map_location=device))
    model_instance.eval().to(device)
    model = model_instance
    your_logger.info("Model loaded and set to evaluation mode.")

    scaler = joblib.load(scaler_path.get())
    your_logger.info("Scaler loaded successfully.")

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
    
    for feature, encoder in label_encoders.items():
        if feature in decoded_data:
            # Decode the column if the encoder exists
            decoded_data[feature] = encoder.inverse_transform(decoded_data[feature].astype(int))

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

#  DiCE-compatible model wrapper
class CustomPyTorchModel(BaseModel):
    def __init__(self, predict_fn, backend='PYT'):
        super().__init__(model=None, backend=backend)
        self.predict_fn = predict_fn

    def get_output(self, input_instance, transform_data=False):
        return self.predict_fn(input_instance)

@plugfunc()
def reset():
    """
    Resets the current datapoint index to the beginning of the datapoints_list,
    so that the next call to next_datapoint() will start from the first datapoint.
    """
    global current_datapoint_index, current_datapoint_id
    
    # Reset to the initial state
    current_datapoint_index = 0
    current_datapoint_id = datapoints_list[current_datapoint_index]
    
    #  the true target label
    true_target = 'Canceled' if y_test[current_datapoint_id] == 0 else 'Check-Out'

    # Extract features, decode them
    features_data = X_test_unscaled.iloc[current_datapoint_id][features]
    decoded_features_data = decode_features(features_data.to_frame().T).iloc[0]  # Decode feature values
    scaled_features = scaler.transform(features_data.values.reshape(1, -1))
    scaled_features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)

    # model predictions and confidence
    with torch.no_grad():
        output = model(scaled_features_tensor)
        confidence_score = torch.nn.functional.softmax(output, dim=1).max().item() * 100
        confidence_score = round(confidence_score, 3)
        predicted_class = output.argmax().item()

    predicted_class_str = 'Canceled' if predicted_class == 0 else 'Check-Out'

    # JSON structure 
    datapoint_info = {
        "role": "datapoint",
        "content": {
            "prediction": predicted_class_str,
            "confidence": confidence_score,
            "true_target": true_target,
            "data": {
                feature: {"title": feature, "value": decoded_features_data[feature]} for feature in features
            }
        }
    }

    api.display(custom_msg=json.dumps(datapoint_info, ensure_ascii=True))
    your_logger.info("Datapoint index has been reset to the first entry in datapoints_list.")


@plugfunc()
def next_datapoint():
    global current_datapoint_index, current_datapoint_id

    # Move to the next datapoint
    current_datapoint_index = (current_datapoint_index + 1) % len(datapoints_list)
    current_datapoint_id = datapoints_list[current_datapoint_index]
    true_target = 'Canceled' if y_test[current_datapoint_id] == 0 else 'Check-Out'

    # Extract features, decode them
    features_data = X_test_unscaled.iloc[current_datapoint_id][features]
    decoded_features_data = decode_features(features_data.to_frame().T).iloc[0]  # Decode feature values
    scaled_features = scaler.transform(features_data.values.reshape(1, -1))
    scaled_features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)

    # model predictions and confidence
    with torch.no_grad():
        output = model(scaled_features_tensor)
        confidence_score = torch.nn.functional.softmax(output, dim=1).max().item() * 100
        confidence_score = round(confidence_score, 3)
        predicted_class = output.argmax().item()
    predicted_class_str = 'Canceled' if predicted_class == 0 else 'Check-Out'

    # JSON structure 
    datapoint_info = {
        "role": "datapoint",
        "content": {
            "prediction": predicted_class_str,
            "confidence": confidence_score,
            "true_target": true_target,
            "data": {
                feature: {"title": feature, "value": decoded_features_data[feature]} for feature in features
            }
        }
    }

    api.display(custom_msg=json.dumps(datapoint_info, ensure_ascii=True))
    your_logger.info(f"Datapoint information displayed on frontend for datapoint id: {current_datapoint_id}")
    
    settings_dic = {"role": "global_settings", "content": {"show_banner": True}}
    api.display(custom_msg=json.dumps(settings_dic, ensure_ascii=True))
    your_logger.info("Frontend banner updated.")
    

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

        # Initialize the LIME explainer on the scaled dataset
        lime_explainer = LimeTabularExplainer(
            scaler.transform(X_test_unscaled[features]), 
            feature_names=features,
            class_names=['Canceled', 'Check-Out'],
            mode='classification'
        )

        explanation = lime_explainer.explain_instance(scaled_features[0], predict_proba, num_features=len(features))
   
        explanation_list = [(feature, contribution) for feature, contribution in explanation.as_list() if contribution != 0]

     # Remove any numerical ranges, inequalities, or constraints from feature names
        cleaned_feature_names = [re.sub(r"(<|>|<=|>=|==|!=)?\s?-?\d*\.?\d*", "", feature).strip() for feature, _ in explanation_list]
        contributions = [contribution for _, contribution in explanation_list]

        # Plot only the impactful features with cleaned names
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(cleaned_feature_names, contributions, color=['red' if c < 0 else 'green' for c in contributions])
        ax.set_xlabel("Contribution to Prediction")
        ax.set_title(f"LIME Explanation")

        # Save the plot to a buffer and encode as base64 for HTML display
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')  # Use 'tight' to ensure labels fit
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Display the plot as an embedded HTML image
        html = f'<img src="data:image/png;base64,{img_base64}" style="background-color:white;height:100%; width:auto"/>'
        api.display(html=html)

        # Convert LIME explanation to text for LLM summary
        lime_text_explanation = explanation.as_list()
        text_summary = f"LIME Explanation :\n"
        text_summary += "\n".join([f"{feature}: {contribution:.4f}" for feature, contribution in lime_text_explanation])

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
    global current_datapoint_id

    try:
        your_logger.info(f"Generating counterfactual explanations for datapoint ID: {current_datapoint_id}")

        # Load test data
        X_test = pd.read_csv(X_test_path.get())
        y_test = pd.read_csv(y_test_path.get(), header=None, names=["reservation_status"])

        # continuous and categorical features
        continuous_features = ['lead_time', 'adr']
        categorical_features = [
            'deposit_type', 'country', 'total_of_special_requests', 'arrival_date_week_number',
            'market_segment', 'previous_cancellations', 'booking_changes', 'reserved_room_type',
            'adults', 'hotel', 'children', 'arrival_date_day_of_month', 'arrival_date_month',
            'stays_in_week_nights', 'stays_in_weekend_nights'
        ]
        all_features = continuous_features + categorical_features

        # Standardize continuous features
        scaler = StandardScaler()
        X_test[continuous_features] = scaler.fit_transform(X_test[continuous_features])

        # prediction function
        def predict_fn(x):
            x = x[all_features]
            x[continuous_features] = scaler.transform(x[continuous_features])
            x = x.apply(pd.to_numeric, errors='coerce').fillna(0)
            x_tensor = torch.tensor(x.values, dtype=torch.float32).to(device)
            with torch.no_grad():
                probs = model.predict_proba(x_tensor).cpu().numpy()
            return probs

        #  DiCE data interface
        d = dice_ml.Data(
            dataframe=pd.concat([X_test, y_test], axis=1),
            continuous_features=continuous_features,
            outcome_name='reservation_status'
        )
        custom_dice_model = CustomPyTorchModel(predict_fn=predict_fn, backend="PYT")
        custom_dice_model.model_type = 'classifier'
        exp = Dice(d, custom_dice_model)

        #  current datapoint instance to explain
        query_instance = X_test.iloc[[current_datapoint_id]].copy()
        query_instance = query_instance[all_features]

        # Generate counterfactuals
        counterfactuals = exp.generate_counterfactuals(query_instance, total_CFs=num_counterfactuals, desired_class="opposite")

        # Process and display counterfactuals
        counterfactuals_data = counterfactuals.cf_examples_list[0].final_cfs_df[all_features]
        counterfactuals_data['reservation_status'] = 1 - y_test.iloc[current_datapoint_id].values[0]
        counterfactuals_data[continuous_features] = scaler.inverse_transform(counterfactuals_data[continuous_features])
        counterfactuals_data[continuous_features] = counterfactuals_data[continuous_features].round(2)

        # Inverse transform the original instance continuous features
        original_instance = query_instance.copy()
        original_instance[continuous_features] = scaler.inverse_transform(original_instance[continuous_features])
        original_instance[continuous_features] = original_instance[continuous_features].round(2)
        original_instance['reservation_status'] = y_test.iloc[current_datapoint_id].values[0]

        # Decode only the encoded features for both original and counterfactual instances
        original_instance_decoded = decode_features(original_instance)
        counterfactuals_data_decoded = decode_features(counterfactuals_data)

        # Combine decoded original and counterfactual data for display
        visual_df = pd.concat([original_instance_decoded, counterfactuals_data_decoded], ignore_index=True)

        visual_df['reservation_status'] = visual_df['reservation_status'].replace({1: 'Check-Out', 0: 'Canceled'})

        row_labels = ['Original Instance'] + [f'Counterfactual {i}' for i in range(1, len(visual_df))]
        visual_df.insert(0, 'Instance', row_labels)

        for col in all_features:
            for i in range(1, len(visual_df)):
                if visual_df.loc[i, col] == visual_df.loc[0, col]:
                    visual_df.loc[i, col] = "-"

        columns_to_keep = ['Instance'] + [
            col for col in visual_df.columns[1:]
            if visual_df.loc[1:, col].ne("-").any()
        ]
        visual_df = visual_df[columns_to_keep]

        fig, ax = plt.subplots(figsize=(14, 16))  # Increase width for clarity
        ax.axis('tight')
        ax.axis('off')

        #  table with improved readability settings
        table = ax.table(cellText=visual_df.values, colLabels=visual_df.columns, cellLoc='center', loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)  
        table.scale(1.8, 1.6)  
        plt.title(f'Counterfactual Explanations', fontsize=18, weight='bold')

        #  plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150) 
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        
        # Display the counterfactual explanations
        html = f'<img src="data:image/png;base64,{img_base64}" style="background-color:white;height:100%; width:auto"/>'
        api.display(html=html)

        # Prepare explanation for LLM with only the relevant changes
        explanation = f"Here are the counterfactual explanations for data instance:\n\n"
        for i, row in visual_df.iterrows():
            row_type = "Original Instance" if i == 0 else f"Counterfactual {i}"
            changes = [f"{col}: {val}" for col, val in row.items() if val != '-' or col == 'reservation_status']
            explanation += f"\n- **{row_type}**: " + ', '.join(changes)

        return explanation.strip()
    
    except Exception as e:
        error_message = f"An error occurred in counterfactual explanation: {str(e)}"
        api.display(html=f"<h3>{error_message}</h3>")
        return error_message 


@plugfunc()
def draw_importances_dashboard(**kwargs) -> str:
    """
    Draws a SHAP-based feature importances dashboard, allowing filtering by feature columns and target variable.

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
            X_test_scaled = scaler.transform(X_test_filtered_df[features])  
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

        # Decode feature names
        decoded_sorted_features = decode_features(pd.DataFrame([sorted_features])).iloc[0].tolist()

        # Plot feature importance as a horizontal bar chart with smaller font size
        plt.figure(figsize=(12, 8))
        bars = plt.barh(decoded_sorted_features, sorted_shap_values)
        plt.xlabel("Shapley Value (Feature Importance)")
        plt.title("Feature Importance based on Shapley Values")
        plt.gca().invert_yaxis()
        
        # Set smaller font size for y-axis labels to prevent overlapping
        plt.yticks(fontsize=8)

        # Annotate bar values for clarity
        for bar, value in zip(bars, sorted_shap_values):
            plt.text(value, bar.get_y() + bar.get_height()/2, f'{value:.2f}', va='center')

        # Save plot to buffer and encode as base64 for HTML display
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)  
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Display the plot as an embedded HTML image
        html = f'<img src="data:image/png;base64,{img_base64}" style="background-color:white;width:100%"/>'
        api.display(html=html)

        # Explanation for LLM with decoded feature names
        explanation = f"""
        Feature Importance Analysis:

        This dashboard visualizes the feature importance based on Shapely values, illustrating which features have the most significant impact on the model's predictions. 
        
        All Feature Importances:
        """ + "\n".join([f"- {feature}: {importance:.4f}" for feature, importance in zip(decoded_sorted_features, sorted_shap_values)]) + "\n\n" + \
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

    Usage:
    - draw_pdp_dashboard(pdp_features=['lead_time'], target_class=0)

    :param pdp_features: List of features for PDPs.
    :param target_class: Target class for PDP (0 = 'Canceled', 1 = 'Check-Out').
    :param kwargs: Filter parameters.
    :return: Explanation and numerical data for LLM.
    """
    try:
        print("Generating PDP Dashboard...")
        
        # Verify if provided features are valid
        for feature in pdp_features:
            if feature not in features:
                print(f"Invalid feature: {feature}")
                return f"The feature '{feature}' is not valid."

        # Apply filters to the dataset
        X_filtered = filter_data(X_test_unscaled.copy(), **kwargs) if kwargs else X_test_unscaled.copy()
        if X_filtered.empty:
            print("Filtered dataset is empty.")
            return "The filtered dataset is empty."

        # Decode features for LLM explanation
        decoded_pdp_features = decode_features(pd.DataFrame([pdp_features])).iloc[0].tolist()

        # Function to calculate partial dependence
        def calculate_partial_dependence(model, X_unscaled, feature, target_class, grid_resolution=70):
            original_values = X_unscaled[feature]
            feature_values = np.linspace(original_values.min(), original_values.max(), grid_resolution)
            avg_predictions = np.zeros(grid_resolution)

            # Loop over each grid value for the feature
            for i, val in enumerate(feature_values):
                X_copy_unscaled = X_unscaled.copy()
                X_copy_unscaled[feature] = val
                X_scaled = scaler.transform(X_copy_unscaled[features])
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

                with torch.no_grad():
                    preds = model(X_tensor)
                    preds_softmax = torch.nn.functional.softmax(preds, dim=1)
                    avg_predictions[i] = preds_softmax[:, target_class].mean().item()

            return feature_values, avg_predictions

        # Initialize plot
        plt.figure(figsize=(12, 8))  # Larger plot size for better readability
        explanation = f"Partial Dependence Plot for target class '{'Canceled' if target_class == 0 else 'Check-Out'}':\n"

        # Calculate and plot PDP for each feature
        for feature, decoded_feature in zip(pdp_features, decoded_pdp_features):
            feature_values, avg_predictions = calculate_partial_dependence(
                model, X_filtered, feature, target_class, grid_resolution=70
            )
            plt.plot(feature_values, avg_predictions, label=f'PDP of {decoded_feature} (Target Class: {target_class})')
            
            # Append data for LLM text summary
            explanation += f"\nFeature: {decoded_feature}\n"
            for val, pred in zip(feature_values, avg_predictions):
                explanation += f"  - Feature Value: {val:.2f}, Average Prediction: {pred:.4f}\n"

        # Plot aesthetics
        plt.xlabel('Feature Value')
        plt.ylabel(f'Average Prediction Probability for Class {target_class}')
        plt.title(f'Partial Dependence Plot for Selected Features (Class {target_class})')
        plt.legend()
        
        # Save and display plot as an HTML image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        html = f'<img src="data:image/png;base64,{img_base64}" style="background-color:white;width:100%"/>'
        api.display(html=html)
        
        # Return the PDP analysis explanation
        return explanation.strip()

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        api.display(html=f"<h3>{error_message}</h3>")
        return error_message