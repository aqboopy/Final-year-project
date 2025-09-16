# Breast Cancer Classification System Using Deep Learning

A comprehensive deep learning pipeline for automated breast cancer detection using mammography images from the CBIS-DDSM dataset. This project implements multiple CNN architectures with medical-specific optimizations and provides interpretability through Grad-CAM visualizations.

This project was completed using Jupyter Notebook.

## IMPORTANT: Model Downloads
Due to file size constraints, trained models and CBIS-DDSM Dataset are available at:[Download Link](https://drive.google.com/file/d/1HB0iXGExDf1c7klYEKFkAqAkH4Jxsqg4/view?usp=sharing)
- Best Model Name: continued_model.keras

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project develops and compares multiple deep learning models for breast cancer classification using mammography images. The system emphasizes medical-specific performance metrics (sensitivity, specificity) and provides visual interpretability to support clinical decision-making.

### Key Objectives
- Implement robust data preprocessing for mammography images
- Develop multiple CNN architectures optimized for medical imaging
- Balance sensitivity and specificity for clinical viability
- Provide model interpretability through Grad-CAM visualizations
- Establish systematic evaluation framework for medical AI

## Features

- **Multiple Model Architectures**: VGG16, EfficientNetB0, Custom CNN, Medical-optimized variants
- **Advanced Training Techniques**: Transfer learning, focal loss, cross-validation, ensemble methods
- **Medical-Specific Optimizations**: Class balancing, enhanced data augmentation, clinical metrics monitoring
- **Interpretability Tools**: Grad-CAM visualizations for attention analysis
- **Comprehensive Evaluation**: Sensitivity, specificity, accuracy, AUC metrics with clinical context

## Dataset

**CBIS-DDSM (Curated Breast Imaging Subset of DDSM)**
- Mammography images with pathology-confirmed labels
- Binary classification: Benign vs Malignant
- Training set: 2,864 samples
- Test set: 704 samples

### Data Structure
```
CBIS-DDSM/
├── csv/
│   ├── calc_case_description_train_set.csv
│   ├── mass_case_description_train_set.csv
│   ├── calc_case_description_test_set.csv
│   ├── mass_case_description_test_set.csv
│   └── dicom_info.csv
└── jpeg/
    └── [mammography images organized by study/series]
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, CPU training supported)
- 16GB+ RAM recommended

### Dependencies
```bash
pip install tensorflow>=2.10.0
pip install scikit-learn>=1.0.0
pip install pandas>=1.3.0
pip install numpy>=1.21.0
pip install matplotlib>=3.5.0
pip install opencv-python>=4.5.0
```

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/breast-cancer-classification.git
cd breast-cancer-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download CBIS-DDSM dataset and place in project directory

## Usage

### Quick Start
```python
# 1. Prepare data
prepare_data_only()

# 2. Train basic model
model, history = simple_train(epochs=10, model_type='simple_vgg')

# 3. Evaluate performance
results = evaluate_only(model_path='simple_best_model.keras')

# 4. Generate interpretability visualizations
run_working_gradcam('simple_best_model.keras', num_samples=10)
```

### Complete Pipeline
```python
# Run full improvement pipeline with cross-validation and ensemble
cv_metrics, comparison_results, best_model = run_complete_improvement()
```

### Model Comparison
```python
# Compare multiple models
compare_models()
```

## Model Architectures

### 1. Basic VGG16
- Transfer learning from ImageNet
- Frozen base layers with custom classifier
- Standard data augmentation

### 2. Enhanced Medical VGG16
- Medical-optimized architecture
- More unfrozen layers for medical imaging
- Enhanced data augmentation addressing mammography artifacts
- Larger classifier head for complex feature learning

### 3. EfficientNetB0
- Memory-efficient architecture
- Compound scaling approach
- Suitable for resource-constrained environments

### 4. Custom Lightweight CNN
- Designed for fast inference
- Reduced parameter count
- Maintained performance for deployment scenarios

## Results

### Final Model Comparison

| Model | Accuracy | Sensitivity | Specificity | AUC |
|-------|----------|-------------|-------------|-----|
| Basic VGG16 | 69.3% | 66.7% | 71.0% | 0.758 |
| Enhanced Medical | 64.3% | 90.9% | 47.2% | 0.791 |
| Continued Training | 71.4% | 62.7% | 77.1% | 0.766 |

### Key Findings
- **Continued Training Model** selected as baseline for balanced clinical performance
- **Enhanced Medical Model** achieves high sensitivity but generates excessive false positives
- **Cross-validation** confirms model robustness across different data splits

### Clinical Implications
The continued training model provides the most clinically viable balance between cancer detection (sensitivity) and false positive management (specificity), making it suitable for real-world screening applications.

## File Structure

```
breast-cancer-classification/
├── README.md
├── requirements.txt
├── notebooks/
│   └── breast_cancer_classification.ipynb
├── models/
│   ├── simple_best_model.keras
│   ├── enhanced_medical_model.keras
│   └── continued_model.keras
├── results/
│   ├── model_comparison_table.csv
│   ├── training_logs/
│   └── gradcam_visualizations/
└── docs/
    └── technical_documentation.md
```

## Key Functions

### Data Processing
- `prepare_data_only()`: Load and preprocess CBIS-DDSM dataset
- `create_robust_dataset()`: Create TensorFlow datasets with augmentation

### Model Training
- `simple_train()`: Basic training with standard parameters
- `run_enhanced_training()`: Medical-optimized training with focal loss
- `continue_training()`: Resume training from saved checkpoints

### Evaluation
- `evaluate_only()`: Comprehensive model evaluation
- `compare_models()`: Side-by-side model comparison
- `run_working_gradcam()`: Generate attention visualizations

## Performance Monitoring

The system includes comprehensive monitoring for:
- **Class Collapse Detection**: Prevents models from predicting single class
- **Medical Metrics Tracking**: Monitors sensitivity thresholds
- **Training Stability**: Automatic checkpointing and recovery
- **Resource Usage**: CPU/GPU optimization settings

## Interpretability

Grad-CAM visualizations provide insight into model decision-making by highlighting regions of attention in mammography images. This supports clinical validation and identifies potential model biases or artifacts.

## Limitations

- Binary classification only (benign vs malignant)
- Single dataset validation (CBIS-DDSM)
- Limited to mammography imaging modality
- Requires substantial computational resources for training

## Future Enhancements

- Multi-class classification including lesion subtypes
- Integration with clinical metadata
- External dataset validation
- Real-time inference optimization
- Regulatory compliance features for clinical deployment

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{breast_cancer_classification_2024,
  title={Breast Cancer Classification System Using Deep Learning},
  author={[Your Name]},
  year={2024},
  publisher={GitHub},
  url={https://github.com/yourusername/breast-cancer-classification}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CBIS-DDSM dataset providers
- TensorFlow and Keras development teams
- Medical imaging research community

---


**Disclaimer**: This system is for research purposes only and has not been validated for clinical use. Always consult qualified medical professionals for diagnostic decisions.

