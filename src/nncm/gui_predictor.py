"""
PySide6 GUI for multi-output Neural Network consequence prediction.

Requirements:
  pip install PySide6 tensorflow scikit-learn numpy
"""

import sys
import os
import numpy as np
import joblib
import tensorflow as tf

TARGET_COLUMNS = ["Release_rate", "Velocity", "Distance_to_LFL", "Flame_length"]
RAW_INPUT_COLUMNS = ["Pressure", "Temperature", "Orifice_diameter"]
FEATURE_EPS = 1e-6
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class ReleaseRatePredictor(QMainWindow):
    """Main window for Release Rate prediction GUI."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.target_meta = None
        self.target_columns = TARGET_COLUMNS.copy()
        self.feature_columns = RAW_INPUT_COLUMNS.copy()
        self.use_engineered_features = True
        self.log_columns = set()
        self.target_shifts = {col: 0.0 for col in self.target_columns}
        self.scale_targets = False
        self.result_labels = {}
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))

        # Paths to model files
        models_dir = os.path.join(project_root, "models")
        self.model_path = os.path.join(models_dir, "best_model.h5")
        self.scaler_path = os.path.join(models_dir, "scaler_X.joblib")
        self.scaler_y_path = os.path.join(models_dir, "scaler_y.joblib")
        self.meta_path = os.path.join(models_dir, "target_meta.joblib")
        
        self.init_ui()
        self.load_model()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("nncm")
        self.setGeometry(100, 100, 500, 400)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        central_widget.setLayout(main_layout)
        
        # Title
        title = QLabel("nncm")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # Input group
        input_group = QGroupBox("Input Parameters")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(15)
        
        # Temperature input
        temp_layout = QHBoxLayout()
        temp_label = QLabel("Temperature (°C):")
        temp_label.setMinimumWidth(150)
        self.temp_input = QLineEdit()
        self.temp_input.setPlaceholderText("e.g., 25.0")
        temp_layout.addWidget(temp_label)
        temp_layout.addWidget(self.temp_input)
        input_layout.addLayout(temp_layout)
        
        # Pressure input
        pressure_layout = QHBoxLayout()
        pressure_label = QLabel("Pressure (barg):")
        pressure_label.setMinimumWidth(150)
        self.pressure_input = QLineEdit()
        self.pressure_input.setPlaceholderText("e.g., 50.0")
        pressure_layout.addWidget(pressure_label)
        pressure_layout.addWidget(self.pressure_input)
        input_layout.addLayout(pressure_layout)
        
        # Orifice Diameter input
        orifice_layout = QHBoxLayout()
        orifice_label = QLabel("Orifice Diameter (mm):")
        orifice_label.setMinimumWidth(150)
        self.orifice_input = QLineEdit()
        self.orifice_input.setPlaceholderText("e.g., 100.0")
        orifice_layout.addWidget(orifice_label)
        orifice_layout.addWidget(self.orifice_input)
        input_layout.addLayout(orifice_layout)
        
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)
        
        # Predict button
        self.predict_button = QPushButton("Predict Consequences")
        self.predict_button.setMinimumHeight(40)
        predict_font = QFont()
        predict_font.setPointSize(12)
        predict_font.setBold(True)
        self.predict_button.setFont(predict_font)
        self.predict_button.clicked.connect(self.predict)
        main_layout.addWidget(self.predict_button)
        
        # Output group
        output_group = QGroupBox("Prediction Result")
        output_layout = QVBoxLayout()
        output_layout.setSpacing(10)
        
        self.output_rows_layout = QVBoxLayout()
        self.output_rows_layout.setSpacing(6)
        self.build_output_rows(self.target_columns)
        output_layout.addLayout(self.output_rows_layout)
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666666; padding: 5px;")
        main_layout.addWidget(self.status_label)
        
        # Add stretch to push everything to top
        main_layout.addStretch()
    
    def build_output_rows(self, columns):
        """Create or refresh the prediction output rows."""
        if not hasattr(self, "output_rows_layout"):
            return
        while self.output_rows_layout.count():
            item = self.output_rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.result_labels = {}
        result_font = QFont()
        result_font.setPointSize(14)
        result_font.setBold(True)
        for col in columns:
            pretty_name = col.replace("_", " ")
            row_layout = QHBoxLayout()
            label = QLabel(f"{pretty_name}:")
            label.setMinimumWidth(180)
            value_label = QLabel("--")
            value_label.setFont(result_font)
            value_label.setStyleSheet("color: #0066cc;")
            row_layout.addWidget(label)
            row_layout.addWidget(value_label)
            row_layout.addStretch()
            self.output_rows_layout.addLayout(row_layout)
            self.result_labels[col] = value_label
    
    def load_model(self):
        """Load the trained model, scaler, and metadata."""
        try:
            # Check if files exist
            if not os.path.exists(self.model_path):
                self.show_error(f"Model file not found: {self.model_path}")
                return False
            
            if not os.path.exists(self.scaler_path):
                self.show_error(f"Scaler file not found: {self.scaler_path}")
                return False
            
            if not os.path.exists(self.meta_path):
                self.show_error(f"Metadata file not found: {self.meta_path}")
                return False

            if not os.path.exists(self.meta_path):
                self.show_error(f"Metadata file not found: {self.meta_path}")
                return False
            
            # Load model
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            
            # Load scaler
            self.scaler_X = joblib.load(self.scaler_path)
            
            # Load metadata (must precede target scaler to know requirements)
            self.target_meta = joblib.load(self.meta_path)
            self.target_columns = self.target_meta.get("target_columns", self.target_columns)
            self.feature_columns = self.target_meta.get("feature_columns", self.feature_columns)
            self.use_engineered_features = bool(self.target_meta.get("use_engineered_features", self.use_engineered_features))
            self.log_columns = set(self.target_meta.get("log_columns", []))
            self.scale_targets = bool(self.target_meta.get("scale_targets", False))
            default_shifts = {col: 0.0 for col in self.target_columns}
            default_shifts.update(self.target_meta.get("shifts", {}))
            self.target_shifts = default_shifts
            self.build_output_rows(self.target_columns)

            if self.scale_targets:
                if not os.path.exists(self.scaler_y_path):
                    self.show_error(f"Target scaler file not found: {self.scaler_y_path}")
                    return False
                self.scaler_y = joblib.load(self.scaler_y_path)
            else:
                self.scaler_y = None
            
            self.status_label.setText("Model loaded successfully")
            self.status_label.setStyleSheet("color: #006600; padding: 5px;")
            return True
            
        except Exception as e:
            self.show_error(f"Error loading model: {str(e)}")
            return False
    
    def validate_inputs(self):
        """Validate user inputs."""
        try:
            temp = float(self.temp_input.text().strip())
            pressure = float(self.pressure_input.text().strip())
            orifice = float(self.orifice_input.text().strip())
            
            # Basic range validation (optional - adjust based on your training data ranges)
            if temp < -1000 or temp > 1000:
                raise ValueError("Temperature seems out of reasonable range")
            if pressure < 0 or pressure > 10000:
                raise ValueError("Pressure seems out of reasonable range")
            if orifice <= 0 or orifice > 10000:
                raise ValueError("Orifice diameter must be positive and reasonable")
            
            return temp, pressure, orifice
            
        except ValueError as e:
            if "could not convert" in str(e).lower():
                QMessageBox.warning(self, "Invalid Input", 
                                  "Please enter valid numbers for all fields.")
            else:
                QMessageBox.warning(self, "Invalid Input", str(e))
            return None
    
    def predict(self):
        """Make prediction using the loaded model."""
        if self.model is None or self.scaler_X is None or self.target_meta is None:
            QMessageBox.critical(self, "Error", 
                               "Model not loaded. Please check model files.")
            return
        
        # Validate inputs
        inputs = self.validate_inputs()
        if inputs is None:
            return
        
        temp, pressure, orifice = inputs
        
        try:
            feature_array = self.build_feature_vector(pressure, temp, orifice)
            
            # Scale inputs
            input_scaled = self.scaler_X.transform(feature_array)
            
            # Make prediction
            raw_prediction = self.concatenate_outputs(self.model.predict(input_scaled, verbose=0))[0]
            if self.scaler_y is not None and self.scale_targets:
                prediction_processed = self.scaler_y.inverse_transform(raw_prediction.reshape(1, -1))[0]
            else:
                prediction_processed = raw_prediction
            prediction = self.inverse_transform_prediction(prediction_processed)
            
            # Display results
            for idx, col in enumerate(self.target_columns):
                value = prediction[idx] if idx < len(prediction) else float("nan")
                label = self.result_labels.get(col)
                if label is not None:
                    label.setText(f"{value:.2f}")
            self.status_label.setText("Prediction completed successfully")
            self.status_label.setStyleSheet("color: #006600; padding: 5px;")
            
        except Exception as e:
            self.show_error(f"Error during prediction: {str(e)}")
            for label in self.result_labels.values():
                label.setText("--")
    
    def show_error(self, message):
        """Display error message."""
        QMessageBox.critical(self, "Error", message)
        self.status_label.setText("Error occurred")
        self.status_label.setStyleSheet("color: #cc0000; padding: 5px;")

    def build_feature_vector(self, pressure, temperature, orifice):
        """Build engineered feature vector consistent with training pipeline."""
        features = self.compute_feature_map(pressure, temperature, orifice)
        vector = []
        for name in self.feature_columns:
            if name not in features:
                raise KeyError(f"Missing engineered feature '{name}' in GUI computation")
            vector.append(features[name])
        return np.array([vector], dtype=np.float32)

    def compute_feature_map(self, pressure, temperature, orifice):
        """Replicates feature engineering logic from training script."""
        feature_map = {
            "Pressure": pressure,
            "Temperature": temperature,
            "Orifice_diameter": orifice,
        }
        if self.use_engineered_features:
            orifice_clamped = max(orifice, 1e-3)
            temp_k = temperature + 273.15
            feature_map["log_pressure"] = np.log1p(max(pressure, 0.0) + FEATURE_EPS)
            feature_map["log_temperature"] = np.log1p(max(temp_k, 0.0) + FEATURE_EPS)
            feature_map["log_orifice"] = np.log1p(orifice_clamped)
            feature_map["pressure_temperature"] = pressure * temperature
            feature_map["pressure_orifice"] = pressure * orifice_clamped
            feature_map["temperature_orifice"] = temperature * orifice_clamped
            feature_map["pressure_over_orifice"] = pressure / orifice_clamped
            feature_map["temperature_sq"] = temperature ** 2
        return feature_map

    @staticmethod
    def concatenate_outputs(predictions):
        """Convert list outputs from the Keras model to a single vector."""
        if isinstance(predictions, list):
            return np.hstack([np.asarray(p).reshape(-1, 1) for p in predictions])
        return np.asarray(predictions)

    def inverse_transform_prediction(self, prediction):
        """Inverse-transform prediction vector according to training metadata."""
        pred = np.array(prediction, dtype=float)
        if pred.ndim == 0:
            pred = pred.reshape(1)
        for idx, col in enumerate(self.target_columns):
            if idx >= pred.size:
                break
            if col in self.log_columns:
                shift = self.target_shifts.get(col, 0.0)
                pred[idx] = np.expm1(pred[idx]) - shift
        return pred


def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = ReleaseRatePredictor()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

