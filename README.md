# Risk Factor Modeling Project

## MSc Public Health Data Science - SDS6108 Health Data Mining and Analytics

A comprehensive data science project analyzing behavioral risk factors from the Behavioral Risk Factor Surveillance System (BRFSS) dataset. This project demonstrates the complete workflow for health data mining and predictive modeling.

## Project Overview

This repository contains a complete data science pipeline for analyzing health risk factors, including:

- **Data Collection & Loading**: Processing BRFSS survey data from CSV and Excel sources
- **Data Cleaning & Preprocessing**: Handling missing values, outliers, and BRFSS-specific coding conventions
- **Exploratory Data Analysis**: Statistical analysis and visualization of health indicators
- **Feature Engineering**: Creating derived features like BMI, health scores, and risk factor counts
- **Predictive Modeling**: Building and evaluating machine learning models for health outcome prediction
- **Results Interpretation**: Translating findings into actionable public health insights

## Dataset

The project uses behavioral risk factor data containing:
- **441,457 survey responses**
- **23 health-related variables** covering demographics, health status, healthcare access, and behavioral risk factors
- Variables including general health, physical/mental health days, blood pressure, smoking status, exercise frequency, and more

## Project Structure

```
risk-factor-modeling/
├── SDS6108_Health_Data_Mining_Project.ipynb  # Main analysis notebook
├── README.md                                  # This file
├── src/                                       # Source code modules
│   ├── data_loading.py                        # Data loading utilities
│   ├── data_cleaning.py                       # Data preprocessing functions
│   ├── eda_analysis.py                        # Exploratory data analysis
│   ├── feature_engineering.py                 # Feature creation utilities
│   ├── model_training.py                      # Machine learning models
│   └── model_evaluation.py                    # Model evaluation metrics
├── data/                                      # Data directory
│   ├── raw/                                   # Raw data files
│   ├── processed/                             # Cleaned data files
│   └── results/                               # Model outputs and visualizations
├── docs/                                      # Documentation
│   ├── data_dictionary.md                     # Variable reference guide
│   └── methodology.md                         # Analysis methodology
├── notebooks/                                 # Additional notebooks
└── tests/                                     # Unit tests
```

## Features

### Variables Analyzed

**Demographic Variables:**
- State FIPS code, Sex, Age (proxy), Marital status
- Education level, Employment status, Veteran status
- Home ownership status

**Health Status Variables:**
- General health rating (1=Excellent to 5=Poor)
- Physical health days (past 30 days)
- Mental health days (past 30 days)
- Days poor health kept from usual activities

**Health Conditions:**
- High blood pressure history
- Current blood pressure medication use
- Difficulty walking or climbing stairs

**Health Behaviors:**
- Smoking history and current usage
- Exercise or physical activity frequency

**Healthcare Access:**
- Health insurance coverage
- Personal doctor availability
- Cost barriers to healthcare
- Routine checkup frequency

### Derived Features

- **BMI**: Calculated Body Mass Index from self-reported weight and height
- **BMI_CAT**: Categorized BMI (Underweight, Normal, Overweight, Obese)
- **Health Score**: Composite health assessment score
- **Risk Count**: Number of cardiovascular risk factors present
- **Access Score**: Healthcare access composite score

## Getting Started

### Prerequisites

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

### Installation

```bash
# Clone the repository
git clone https://github.com/OumaCavin/risk-factor-modeling.git
cd risk-factor-modeling

# Install dependencies
pip install -r requirements.txt
```

### Usage

Open the main Jupyter notebook to run the complete analysis:

```bash
jupyter notebook SDS6108_Health_Data_Mining_Project.ipynb
```

## Model Performance

### Best Model: Random Forest Classifier

| Metric | Score |
|--------|-------|
| Accuracy | ~85% |
| F1-Score (Poor Health) | ~0.65 |
| ROC-AUC | ~0.88 |
| Cross-Validation ROC-AUC | ~0.87 |

### Top Predictive Features

1. Physical health days (PHYSHLTH)
2. General health self-assessment (GENHLTH)
3. Difficulty walking (DIFFWALK)
4. Exercise frequency (EXERANY2)
5. Body Mass Index (BMI)

## Public Health Implications

The analysis reveals several key findings with public health significance:

1. **Physical Activity**: Regular exercise is strongly protective against poor health outcomes
2. **Chronic Disease Management**: Hypertension screening and management remain critical
3. **Mental Health Integration**: Mental and physical health are interconnected
4. **Healthcare Access**: Removing cost barriers improves health outcomes
5. **High-Risk Screening**: Identifying individuals with multiple risk factors enables early intervention

## Methodology

This project follows a comprehensive data science pipeline:

1. **Phase 1**: Project setup and data loading
2. **Phase 2**: Data exploration and quality assessment
3. **Phase 3**: Data cleaning and preprocessing
4. **Phase 4**: Exploratory data analysis
5. **Phase 5**: Feature engineering
6. **Phase 6**: Model development
7. **Phase 7**: Model evaluation
8. **Phase 8**: Results and conclusions

## Results

The final model successfully identifies individuals at risk for poor health outcomes with high accuracy. Key findings include:

- Strong correlation between physical inactivity and poor health
- Significant impact of healthcare access on health outcomes
- Importance of integrated approaches to physical and mental health
- Value of composite risk scores for population health screening

## Limitations

- Cross-sectional design limits causal inference
- Self-reported data subject to recall bias
- Class imbalance in target variable (poor health is minority class)
- Results specific to US adult population

## Future Work

- Apply resampling techniques to address class imbalance
- Explore ensemble methods (XGBoost, LightGBM)
- Implement SHAP values for better interpretability
- Link with clinical records for validation
- Develop web-based risk assessment tools

## Author

**Cavin Otieno**  
MSc Public Health Data Science Student

## Course Information

- **Course**: SDS6108 Health Data Mining and Analytics
- **Institution**: [Your University Name]
- **Academic Year**: 2024-2025

## License

This project is for educational purposes as part of the MSc Public Health Data Science program.

## Acknowledgments

- Behavioral Risk Factor Surveillance System (BRFSS)
- Course instructors and teaching assistants
- Peer reviewers and study group members
