# Methodology Documentation

## Analysis Approach

This document describes the methodology used in the risk factor modeling project for predicting poor health outcomes using behavioral risk factor data.

## Data Science Pipeline

The project follows a comprehensive 8-phase data science pipeline:

### Phase 1: Project Setup and Data Loading

**Objectives:**
- Establish project environment and configuration
- Load data from source files (CSV and Excel)
- Validate data integrity

**Activities:**
1. Import required libraries (pandas, numpy, matplotlib, seaborn, scikit-learn)
2. Configure visualization settings and random seeds
3. Load data from CSV/Excel files
4. Perform initial data quality checks

### Phase 2: Data Exploration and Quality Assessment

**Objectives:**
- Understand data structure and variable types
- Identify data quality issues
- Document data characteristics

**Activities:**
1. Examine data dimensions and structure
2. Analyze data types and distributions
3. Calculate missing value statistics
4. Compute descriptive statistics
5. Identify special value codes (BRFSS specific)
6. Generate data quality report

### Phase 3: Data Cleaning and Preprocessing

**Objectives:**
- Handle missing values appropriately
- Treat special codes and outliers
- Convert variables to proper types

**Activities:**
1. Convert BRFSS special values to appropriate representations
2. Apply missing value treatment strategies:
   - Mode imputation for categorical variables
   - Median imputation for numeric variables
   - Drop rows for variables with excessive missingness
3. Cap outliers to reasonable ranges
4. Optimize data types for memory efficiency
5. Remove duplicate records

### Phase 4: Exploratory Data Analysis

**Objectives:**
- Understand variable distributions
- Identify relationships between variables
- Generate hypotheses for modeling

**Activities:**
1. **Univariate Analysis:**
   - Distribution plots for health status variables
   - Frequency counts for categorical variables
   - Summary statistics for numeric variables

2. **Bivariate Analysis:**
   - Correlation matrix for numeric variables
   - Cross-tabulations for categorical variables
   - Grouped statistics by demographic segments

3. **Statistical Testing:**
   - Chi-square tests for categorical associations
   - Spearman correlations for ordinal variables
   - Cramér's V for effect size

4. **Multivariate Analysis:**
   - Health status by demographic groups
   - Risk factor patterns
   - BMI distribution by health status

### Phase 5: Feature Engineering

**Objectives:**
- Create meaningful derived features
- Prepare variables for modeling
- Define target variable

**Activities:**
1. **Feature Creation:**
   - Calculate BMI from weight and height
   - Create BMI categories
   - Build composite health score
   - Generate risk factor count
   - Calculate healthcare access score

2. **Target Variable Definition:**
   - POOR_HEALTH = (GENHLTH >= 4)

3. **Feature Selection:**
   - Correlation analysis with target
   - Remove highly correlated features
   - Select features with significant predictive power

### Phase 6: Model Development

**Objectives:**
- Build predictive models
- Compare multiple algorithms
- Optimize model hyperparameters

**Activities:**
1. **Data Preparation:**
   - Train-test split (80-20)
   - Stratified sampling to preserve class proportions
   - Feature scaling using StandardScaler

2. **Baseline Model:**
   - Logistic Regression with class weighting
   - Establishes minimum performance threshold

3. **Advanced Models:**
   - Decision Tree Classifier
   - Random Forest Classifier
   - Gradient Boosting Classifier

4. **Hyperparameter Tuning:**
   - GridSearchCV with cross-validation
   - Optimize for ROC-AUC score
   - Test multiple parameter combinations

### Phase 7: Model Evaluation

**Objectives:**
- Assess model performance
- Compare different models
- Identify most important features

**Activities:**
1. **Performance Metrics:**
   - Accuracy
   - F1-Score
   - ROC-AUC
   - Cross-validation scores

2. **Visual Evaluation:**
   - ROC curves
   - Precision-recall curves
   - Confusion matrices

3. **Feature Importance:**
   - Random Forest feature importance
   - Rank features by predictive power
   - Identify key risk factors

### Phase 8: Results and Conclusions

**Objectives:**
- Synthesize findings
- Generate public health insights
- Document limitations and future work

**Activities:**
1. Summary of key findings
2. Public health implications
3. Study limitations
4. Recommendations for future research

## Statistical Methods

### Missing Value Treatment

| Variable Type | Treatment Method | Rationale |
|--------------|------------------|-----------|
| Categorical | Mode imputation | Preserves distribution |
| Ordinal | Median imputation | Robust to outliers |
| Numeric | Median imputation | Robust to outliers |
| Special codes (88) | Convert to 0 | Represents "none" |
| Special codes (77, 99) | Convert to NaN | Non-response |

### Outlier Treatment

Outliers are identified using the IQR method and capped to reasonable ranges:

- **Physical/Mental health days**: 0-30 (valid range)
- **Weight**: 50-700 lbs (reasonable adult range)
- **Height**: 36-96 inches (3-8 feet)
- **Number of adults**: 1-20 (household reasonable range)

### Statistical Tests

**Chi-Square Test:**
- Used for testing independence between categorical variables
- Null hypothesis: Variables are independent
- Alternative hypothesis: Variables are associated
- Significance level: α = 0.05

**Spearman Correlation:**
- Used for ordinal variables and non-normal distributions
- Measures monotonic relationships
- Range: -1 to 1 (0 = no correlation)

**Cramér's V:**
- Effect size measure for chi-square test
- Range: 0 to 1
- Interpretation:
  - 0.00-0.10: Negligible
  - 0.10-0.20: Weak
  - 0.20-0.40: Moderate
  - 0.40-0.60: Relatively strong
  - 0.60-0.80: Strong
  - 0.80-1.00: Very strong

## Machine Learning Methods

### Models Used

1. **Logistic Regression**
   - Baseline model
   - Handles binary classification
   - Provides interpretable coefficients
   - Class weight adjustment for imbalance

2. **Decision Tree**
   - Non-linear model
   - Easy interpretation
   - Prone to overfitting
   - Max depth limited to 10

3. **Random Forest**
   - Ensemble of decision trees
   - Reduces overfitting
   - Handles feature interactions
   - Provides feature importance

4. **Gradient Boosting**
   - Sequential ensemble method
   - High predictive accuracy
   - Handles complex patterns
   - Computationally intensive

### Evaluation Metrics

**Accuracy:**
- Proportion of correct predictions
- Useful when classes are balanced

**F1-Score:**
- Harmonic mean of precision and recall
- Better for imbalanced datasets
- Formula: 2 * (Precision * Recall) / (Precision + Recall)

**ROC-AUC:**
- Area under Receiver Operating Characteristic curve
- Measures discrimination ability
- Threshold-independent
- Range: 0.5 (random) to 1.0 (perfect)

**Cross-Validation:**
- 5-fold stratified cross-validation
- Provides robust performance estimate
- Reduces variance in performance estimates

### Hyperparameter Tuning

**GridSearchCV Parameters:**
- Cross-validation folds: 3
- Scoring metric: ROC-AUC
- Parallel jobs: -1 (all cores)

**Random Forest Grid:**
- n_estimators: [100, 200]
- max_depth: [8, 10, 12]
- min_samples_split: [5, 10]
- min_samples_leaf: [2, 4]

## Reproducibility

### Random Seed

All random processes use seed 42 for reproducibility:
- NumPy random operations
- Train-test split
- Model initialization

### Data Version

- Original data: behavioral risk factor-selected.csv
- Cleaned data: cleaned_health_data.csv
- Analysis date: January 2026

## Limitations

1. **Cross-sectional design**: Cannot establish causality
2. **Self-reported data**: Subject to recall bias
3. **Class imbalance**: Poor health is minority class (~15%)
4. **Missing data**: Some responses are missing or refused
5. **Geographic scope**: US adult population only

## Software Environment

- Python 3.8+
- pandas 1.3+
- numpy 1.20+
- matplotlib 3.4+
- seaborn 0.11+
- scikit-learn 1.0+
- scipy 1.7+
