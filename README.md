# HandPi: Polish Sign Language Recognition

## Project Overview
HandPi is a machine learning project focused on recognizing Polish Sign Language (PSL) gestures using sensor data. The system analyzes data from both ADC (Analog-to-Digital Converter) channels that capture hand pressure/flex measurements and IMU (Inertial Measurement Unit) channels for spatial orientation, enabling accurate sign language interpretation.

## Features
- **Complete Alphabet Coverage**: Supports all 36 Polish alphabet signs, including special characters
- **Multi-Modal Sensing**: Integrates 10 ADC channels and 6 IMU channels for comprehensive gesture capture
- **Dynamic & Static Sign Recognition**: Distinguishes between static hand positions and dynamic movements
- **Custom Neural Network Architecture**: Utilizes a hybrid CNN-GRU model for temporal sequence analysis

## Technical Implementation
The system implements a deep learning approach with:
- Convolutional Neural Network (CNN) layers for feature extraction
- Gated Recurrent Unit (GRU) layers for sequential pattern recognition
- Batch normalization and regularization techniques to prevent overfitting
- Data augmentation capabilities for improved model generalization

## Model Architecture
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Input                        [(None, 75, 16)]          0         
BatchNormalization           (None, 75, 16)            64        
Conv1D                       (None, 73, 64)            3,136     
BatchNormalization           (None, 73, 64)            256       
GRU                          (None, 73, 75)            31,275    
BatchNormalization           (None, 73, 75)            300       
Conv1D                       (None, 71, 64)            14,464    
BatchNormalization           (None, 71, 64)            256       
GRU                          (None, 71, 75)            31,275    
Conv1D                       (None, 69, 64)            14,464    
BatchNormalization           (None, 69, 64)            256       
GRU                          (None, 75)                31,275    
BatchNormalization           (None, 75)                300       
Dense                        (None, 36)                2,736     
=================================================================
```

## Dataset
The system was trained on custom-collected gesture data from the HandPi glove device, capturing approximately 75 samples per gesture. The dataset includes:
- Pressure sensor readings from all fingers
- 3D orientation (Euler angles)
- 3D acceleration
- Labels for all 36 Polish alphabet characters

## Results
The model achieves significant accuracy in recognizing Polish Sign Language gestures, with evaluation metrics and confusion matrix visualization included in the codebase.

## Requirements
- Python 3.6+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Future Work
- Real-time inference on embedded devices
- Expansion to phrase-level recognition
- User-specific calibration for improved accuracy

## License
MIT License
