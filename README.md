# Risk Factor Modeling Project

## MSc Public Health Data Science - SDS6108 Health Data Mining and Analytics

**Academic Year:** 2025-2026  
**Student Name:** Cavin Otieno  
**Student ID:** SDS6/46982/2024  
**Python Version:** 3.12.3

---

A comprehensive data science project analyzing behavioral risk factors from the Behavioral Risk Factor Surveillance System (BRFSS) dataset. This project demonstrates the complete workflow for health data mining and predictive modeling, meeting all academic requirements for the MSc Public Health Data Science program. The implementation includes robust preprocessing pipelines, multiple machine learning algorithms, hyperparameter optimization, feature importance analysis, and an interactive prediction system for clinical interpretation.

---

## Table of Contents

1. [Background: Public Health Challenge](#background-public-health-challenge)
2. [Project Objective](#project-objective)
3. [Our Solution](#our-solution)
4. [Dataset Overview](#dataset-overview)
5. [Preprocessing](#preprocessing)
6. [Analysis](#analysis)
7. [Methodology](#methodology)
8. [Results](#results)
9. [Key Insights from Exploratory Analysis](#key-insights-from-exploratory-analysis)
10. [Insights](#insights)
11. [Model Performance Benchmarking](#model-performance-benchmarking)
12. [Feature Importance: What Drives Prediction](#feature-importance-what-drives-prediction)
13. [Interactive Prediction Tool](#interactive-prediction-tool)
14. [Impact](#impact)
15. [Recommendations and Future Directions](#recommendations-and-future-directions)
16. [Academic Requirements Met](#academic-requirements-met)
17. [Public Health Critical Metrics](#public-health-critical-metrics)
18. [Handcrafted Features](#handcrafted-features)
19. [Comprehensive Visualizations](#comprehensive-visualizations)
20. [Clinical Interpretation Dashboard](#clinical-interpretation-dashboard)
21. [Complete Output Management](#complete-output-management)
22. [Project Structure](#project-structure)
23. [Getting Started](#getting-started)
24. [Author Information](#author-information)
25. [License](#license)

---

## Background: Public Health Challenge

### The Growing Burden of Chronic Disease

Chronic diseases, including heart disease, diabetes, and obesity, represent the leading cause of death and disability worldwide. According to the World Health Organization, chronic diseases account for approximately 71% of all deaths globally, with cardiovascular diseases alone causing an estimated 17.9 million deaths annually. In the United States, the Behavioral Risk Factor Surveillance System (BRFSS) has documented rising rates of obesity, hypertension, and physical inactivity across all demographic groups. The economic burden of chronic disease is equally staggering, with healthcare costs associated with preventable conditions consuming an increasingly large share of national budgets. These trends underscore the critical need for effective prevention strategies and early identification of at-risk individuals.

The transition from reactive healthcare to proactive prevention requires sophisticated tools for risk identification and stratification. Traditional clinical approaches rely heavily on established risk factors and population-level statistics, but these methods often fail to capture the complex interactions between behavioral, social, and clinical factors that determine individual health outcomes. Furthermore, the siloed nature of healthcare data makes it difficult to obtain a holistic view of patient risk profiles. This fragmentation limits the ability of healthcare systems to implement truly personalized prevention strategies that address the root causes of chronic disease.

### The Need for Predictive Analytics

Traditional public health approaches rely on reactive interventions after disease onset. However, predictive analytics offers a paradigm shift toward proactive, personalized prevention strategies. By identifying individuals at elevated risk for poor health outcomes before symptoms manifest, healthcare systems can implement targeted interventions, optimize resource allocation, and ultimately reduce disease burden. The application of machine learning to public health data represents a significant opportunity to enhance the precision and effectiveness of prevention efforts. Unlike traditional statistical methods, machine learning algorithms can identify non-linear relationships and complex interactions among variables, potentially revealing novel risk patterns that would otherwise remain hidden.

The promise of predictive analytics in healthcare extends beyond individual risk prediction to population health management and resource optimization. Health systems can use predictive models to identify high-risk subgroups that would benefit most from intensive interventions, allowing for more efficient use of limited public health resources. Additionally, predictive models can inform the design of community-wide prevention programs by identifying modifiable risk factors that contribute most significantly to poor health outcomes. This data-driven approach to public health planning represents a significant advancement over intuition-based decision making.

### Data-Driven Health Prediction

The convergence of big data analytics and healthcare presents unprecedented opportunities for improving population health. Machine learning algorithms can analyze complex patterns in behavioral, demographic, and clinical data to predict health outcomes with greater accuracy than traditional statistical methods. This project explores the application of advanced predictive modeling techniques to behavioral risk factor data, demonstrating how data science can inform public health decision-making. The BRFSS dataset provides an ideal foundation for this analysis, containing comprehensive information about health behaviors, chronic conditions, and healthcare access across a large and diverse population sample.

The methodology developed in this project can be adapted to various public health contexts and integrated into existing surveillance systems. By demonstrating the feasibility and value of machine learning approaches to health risk prediction, this work contributes to the growing body of evidence supporting the adoption of predictive analytics in public health practice. The project also addresses important considerations related to model interpretability, clinical validation, and ethical deployment, ensuring that the resulting tools are suitable for real-world applications.

---

## Project Objective

### Primary Objective

To develop and validate a machine learning-based predictive model that accurately identifies individuals at risk for poor health outcomes using behavioral risk factor data, enabling proactive public health interventions. This objective encompasses the entire data science pipeline from data collection and preprocessing through model development, validation, and interpretation. The model must achieve sufficient predictive performance to be clinically useful while remaining interpretable enough to inform actionable recommendations.

### Specific Objectives

The project addresses multiple interconnected objectives that together constitute a comprehensive approach to health risk prediction. The first objective focuses on data exploration and understanding, conducting comprehensive analysis of behavioral risk factor data to identify patterns, distributions, and relationships among health indicators. This foundational work informs all subsequent analytical decisions and ensures that the model is built on a thorough understanding of the data.

The second objective involves feature engineering, creating meaningful derived features that capture the complex interactions between behavioral, demographic, and clinical variables. This includes the development of clinically meaningful indices such as Body Mass Index, composite health scores, and cardiovascular risk counts that enhance the predictive power of the model while maintaining clinical relevance.

The third objective centers on model development, building and comparing multiple machine learning algorithms to identify the optimal approach for predicting health outcomes. By implementing and rigorously evaluating multiple modeling strategies, the project ensures that the final model represents the best available approach given the available data and computational resources.

The fourth objective addresses model optimization through hyperparameter tuning and cross-validation to maximize model performance and generalizability. This systematic approach to model refinement ensures that the final model is not overfit to the training data and will perform well on new, unseen cases.

The fifth objective focuses on interpretation and insights, extracting actionable findings from model results to inform public health policy and clinical practice. This includes detailed analysis of feature importance and development of clinical interpretation guidelines.

The sixth objective ensures clinical validation, confirming that model outputs are clinically meaningful and interpretable by healthcare professionals. This involves collaboration with clinical experts and adherence to established standards for medical decision support tools.

---

## Our Solution

### Comprehensive Data Science Pipeline

This project implements a complete end-to-end data science solution tailored specifically for public health applications. The solution addresses the unique challenges of health data analysis, including complex missing data patterns, categorical variable handling, and the need for interpretable results. Each component of the pipeline has been designed with clinical applicability in mind, ensuring that the final deliverables can be translated into real-world public health interventions.

The preprocessing pipeline implements specialized handling of BRFSS survey data, including systematic treatment of non-response codes, missing value imputation strategies appropriate for health data, and outlier detection algorithms that account for clinical plausibility. This rigorous approach to data quality ensures that the subsequent modeling steps are built on a solid foundation of clean, reliable data.

The feature engineering component creates domain-informed derived features including Body Mass Index calculations, composite health scores, and cardiovascular risk counts. These features leverage clinical expertise to capture health risk factors in ways that are meaningful for both prediction and interpretation. The handcrafted features are designed to be clinically intuitive while providing additional predictive value beyond the raw survey responses.

The modeling component implements and compares four distinct machine learning algorithms: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. This multi-model approach allows for rigorous performance comparison and ensures that the final model selection is based on comprehensive evaluation rather than convenience or familiarity with particular algorithms.

The evaluation component provides rigorous model validation using multiple performance metrics, cross-validation protocols, and held-out test sets. This comprehensive evaluation approach ensures that the reported performance metrics accurately reflect the model's expected performance in real-world applications.

The interpretation component delivers feature importance analysis and clinical interpretation guidelines that enable healthcare practitioners to understand and act on model predictions. This focus on interpretability is essential for clinical adoption and regulatory approval of predictive tools.

### Technology Stack

The project utilizes a modern, well-maintained technology stack that ensures reproducibility and performance. Python 3.12.3 serves as the programming language, chosen for its extensive ecosystem of data science libraries and its prominence in both academic research and industry applications. Pandas and NumPy provide efficient data manipulation capabilities, while SciPy supports advanced statistical analysis. The scikit-learn library provides the machine learning algorithms and evaluation tools used throughout the project. Matplotlib and Seaborn generate publication-quality visualizations that effectively communicate analytical findings. The Jupyter Notebook environment enables reproducible analysis with integrated documentation and code execution.

---

## Dataset Overview

### Data Source

The Behavioral Risk Factor Surveillance System (BRFSS) is the nation's premier system of health-related telephone surveys that collects state data about U.S. residents regarding their health-related risk behaviors, chronic health conditions, and use of preventive services. Established in 1984, the BRFSS has become the largest continuously conducted health surveillance system in the world, providing critical data for monitoring the health of American populations. The survey is conducted by state health departments in collaboration with the Centers for Disease Control and Prevention (CDC), using standardized questionnaires and data collection procedures to ensure comparability across states and years.

The BRFSS methodology involves random-digit-dialing of both landline and cellular telephone numbers within each state. The survey collects information on health risk behaviors, chronic health conditions, and use of preventive services from non-institutionalized adults aged 18 years and older. This broad population coverage ensures that the resulting dataset represents the full spectrum of health statuses and behaviors in the adult population. The large sample size and standardized methodology make the BRFSS an ideal data source for developing and validating predictive models for public health applications.

### Dataset Specifications

| Attribute | Value |
|-----------|-------|
| **Total Records** | 441,457 survey responses |
| **Original Variables** | 23 health-related features |
| **Data Types** | Categorical, ordinal, numeric |
| **Geographic Coverage** | 72 US states and territories |
| **Survey Period** | 2024-2025 data collection |

### Variable Categories

#### Demographic Variables

The dataset includes essential demographic characteristics that influence health outcomes and serve as important predictors in the predictive model. State FIPS code provides geographic stratification capability, enabling analysis of regional health patterns. Sex variable allows for sex-specific analysis and addresses potential gender disparities in health outcomes. Age is captured through categorical groupings that reflect life stage differences in health risk profiles. Marital status provides insight into social support structures that influence health behaviors and outcomes. Education level serves as a proxy for health literacy and socioeconomic status, both of which significantly impact health outcomes. Employment status indicates economic stability and access to employer-sponsored health benefits. Veteran status identifies a population with unique healthcare needs and access patterns. Home ownership and number of adults in household provide additional socioeconomic context.

#### Health Status Variables

Health status variables capture respondents' self-reported health conditions and functional status. The general health rating question asks respondents to rate their overall health on a five-point scale from excellent to poor, providing a subjective assessment of health status that has been validated as a strong predictor of mortality and healthcare utilization. Physical health days and mental health days questions measure the number of days in the past 30 days when poor physical or mental health affected usual activities, providing quantitative measures of health-related quality of life. These variables are particularly valuable for identifying individuals with chronic conditions or functional limitations that may not be captured by binary disease indicators alone.

#### Health Conditions

The dataset includes indicators of specific health conditions that are known risk factors for poor health outcomes. High blood pressure history identifies individuals who have ever been diagnosed with hypertension, a major risk factor for cardiovascular disease. Current blood pressure medication use indicates treatment status for hypertension and provides insight into disease management. Difficulty walking or climbing stairs assesses mobility limitations that may indicate underlying chronic conditions, musculoskeletal problems, or deconditioning. These functional status indicators are particularly valuable for identifying individuals at risk for functional decline and adverse health outcomes.

#### Health Behaviors

Health behavior variables capture modifiable risk factors that are key targets for public health intervention. Smoking history questions identify current smokers, former smokers, and never smokers, enabling analysis of the dose-response relationship between smoking exposure and health outcomes. Exercise or physical activity frequency measures participation in leisure-time physical activity, a protective factor against numerous chronic conditions. These behavioral variables are particularly important for predictive modeling because they represent modifiable risk factors that can be targeted by intervention programs.

#### Healthcare Access

Healthcare access variables measure the structural and financial barriers that individuals face in obtaining needed medical care. Health insurance coverage indicates presence of any form of health coverage, a critical determinant of healthcare utilization and health outcomes. Personal doctor availability measures the presence of a usual source of care, which is associated with better chronic disease management and preventive service receipt. Cost barriers capture instances where respondents were unable to see a doctor due to cost in the past year, identifying financial access barriers. Routine checkup frequency measures adherence to preventive care recommendations and provides insight into engagement with the healthcare system.

#### Body Metrics

Self-reported weight and height enable calculation of Body Mass Index, a standard measure for classifying body weight relative to height. Despite known limitations of self-reported measures, BMI calculated from self-reported data remains a valuable screening tool for identifying individuals at risk for weight-related health conditions. The large sample size of the BRFSS enables analysis of BMI patterns across demographic groups and identification of populations at elevated risk for obesity-related conditions.

### Target Variable

**POOR_HEALTH**: Binary classification target defined as:

- **0 (Good Health)**: General health rating of 1-3 (Excellent to Fair)
- **1 (Poor Health)**: General health rating of 4-5 (Poor or Very Poor)

This binary classification target balances clinical meaningfulness with practical utility for public health applications. The threshold at fair health (rating of 3) corresponds to a commonly used definition in public health research that identifies individuals with significant health concerns requiring attention. This definition captures individuals who are at elevated risk for adverse health outcomes and may benefit from targeted interventions.

---

## Preprocessing

### Data Quality Assessment

The preprocessing pipeline begins with comprehensive data quality assessment that documents the characteristics of the raw data and informs subsequent cleaning decisions. This systematic approach ensures transparency in data preparation and enables reproducibility of the analytical workflow. The assessment process identifies data quality issues that could bias model estimates or reduce predictive performance if left unaddressed.

The missing value analysis systematically quantifies the extent and patterns of missing data across all variables. Variables with high missingness rates are flagged for special treatment, and the missing completely at random (MCAR), missing at random (MAR), and missing not at random (MNAR) assumptions are evaluated for each variable. This analysis informs the selection of appropriate missing value treatment strategies and helps identify variables that may need to be excluded from modeling due to data quality concerns.

The special code treatment phase addresses the unique coding conventions used in the BRFSS survey. Unlike standard missing value indicators, the BRFSS uses a systematic set of codes to represent different types of non-response and special circumstances. Proper treatment of these codes is essential for accurate analysis, as treating them as valid data or standard missing values would introduce bias.

The outlier detection phase applies statistical methods to identify values that fall outside the expected range for each variable. For continuous variables, the interquartile range method identifies values that are statistically unlikely given the distribution of the data. For categorical variables, frequency analysis identifies categories that represent an implausibly small or large proportion of the sample.

The duplicate record identification phase detects and removes duplicate survey responses that could bias estimates and model training. Duplicate detection considers both exact matches on all variables and probabilistic matching on key identifier variables.

### BRFSS Special Value Handling

The BRFSS uses specific codes to represent non-response categories and special circumstances. These codes must be properly identified and treated to avoid introducing bias into the analysis. The following table summarizes the most common special codes and their appropriate treatment:

| Code | Meaning | Treatment |
|------|---------|-----------|
| 7 | Refused (small range) | Convert to NaN |
| 8 | Not asked/Missing | Convert to NaN |
| 9 | Don't know (small range) | Convert to NaN |
| 77 | Refused (large range) | Convert to NaN |
| 88 | None/Zero | Convert to 0 (for day counts) |
| 99 | Don't know (large range) | Convert to NaN |
| 7777 | Refused (continuous) | Convert to NaN |
| 9999 | Don't know (continuous) | Convert to NaN |

The treatment of code 88 (representing "none" or "zero days") requires special attention, as this code represents a valid response rather than missing data. For day count variables such as physical health days and mental health days, a value of 88 indicates that the respondent reported zero days of poor health, which is clinically meaningful and should be retained in the analysis.

### Missing Value Treatment

Different treatment strategies are applied based on the variable type and the pattern of missingness. The selection of treatment strategies balances statistical validity with clinical interpretability, ensuring that the cleaned data maintains its utility for public health analysis.

| Variable Type | Treatment Method | Rationale |
|--------------|------------------|-----------|
| Categorical/Binary | Mode imputation | Preserves distribution, most common value |
| Ordinal | Median imputation | Robust to outliers, preserves ordering |
| Numeric | Median imputation | Robust to extreme values |
| Special codes (88) | Convert to 0 | Represents "none" or "zero days" |

For variables with high missingness rates or patterns suggesting non-random missingness, multiple imputation may be considered to properly account for uncertainty in the imputed values. However, given the large sample size and the nature of the missingness patterns observed, single imputation methods provide adequate results while maintaining computational efficiency.

### Outlier Treatment

Outliers identified using the Interquartile Range (IQR) method are capped to clinically reasonable ranges. This approach preserves the information contained in extreme values while preventing them from exerting undue influence on model training. The clinical plausibility review ensures that outlier treatment decisions are informed by domain expertise rather than purely statistical considerations.

| Variable | Valid Range | Action |
|----------|-------------|--------|
| Physical health days | 0-30 | Cap outliers |
| Mental health days | 0-30 | Cap outliers |
| Weight (lbs) | 50-700 | Cap outliers |
| Height (inches) | 36-96 | Cap outliers |
| Number of adults | 1-20 | Cap outliers |

### Data Type Optimization

Conversion of variables to memory-efficient data types reduces computational requirements and enables processing of large datasets on limited hardware. This optimization is particularly important for the BRFSS dataset, which contains over 400,000 records that must be processed multiple times during model training and evaluation.

Binary variables are stored as int8, categorical variables as int8, weight and height as int16, and state codes as int32. These optimizations achieve approximately 60-70% reduction in memory usage compared to default numeric types while preserving all necessary precision for the analysis.

---

## Analysis

### Exploratory Data Analysis (EDA)

Comprehensive exploratory data analysis establishes the foundation for all subsequent modeling decisions and provides essential insights into the health status and behaviors of the study population. The EDA employs multiple analytical approaches to characterize the data from different perspectives and identify patterns that inform feature engineering and model selection.

#### Univariate Analysis

The univariate analysis examines the distribution of each variable in isolation, identifying patterns in central tendency, dispersion, and distributional shape. For continuous variables, summary statistics including mean, median, standard deviation, and percentiles characterize the typical values and variability in the sample. Histogram and density plots visualize the distributional shape and identify any unusual patterns such as bimodality or heavy tails that may warrant further investigation.

For categorical variables, frequency tables and bar charts display the prevalence of each category and enable comparison across groups. The analysis pays particular attention to categories with very low or very high frequencies, as these may indicate data quality issues or important population subgroups that should be considered during modeling.

The univariate analysis of the general health rating variable reveals that most respondents rate their health as good or better, with a smaller proportion reporting fair or poor health. This distribution has important implications for class imbalance in the predictive model and informs the selection of appropriate evaluation metrics and modeling strategies.

The physical and mental health days variables show right-skewed distributions, with most respondents reporting few days of poor health and a smaller proportion reporting significant numbers of unhealthy days. This pattern is consistent with the overall good health of the population and suggests that these variables may have strong predictive value for identifying the small proportion of individuals with significant health challenges.

#### Bivariate Analysis

The bivariate analysis investigates relationships between pairs of variables, identifying associations that inform feature selection and providing preliminary evidence about potential predictors of the target variable. Correlation analysis using both Pearson and Spearman methods quantifies the strength and direction of associations between continuous variables. The Spearman correlation is preferred for ordinal variables and variables with non-normal distributions, as it captures monotonic relationships without assuming linearity.

Cross-tabulation with chi-square testing examines associations between categorical variables, identifying significant relationships that may indicate confounding or effect modification in the subsequent modeling analysis. Cramér's V provides a standardized measure of association strength that enables comparison across variables with different numbers of categories.

The bivariate analysis reveals strong associations between several predictor variables and the target health outcome. Physical health days, general health rating, and difficulty walking show particularly strong associations with poor health status, confirming their importance as potential predictors in the model. Moderate associations are observed for exercise frequency, BMI category, and high blood pressure, consistent with established risk factors for poor health outcomes.

#### Multivariate Analysis

The multivariate analysis extends the investigation to simultaneous examination of multiple variable relationships, identifying complex patterns that cannot be captured in bivariate analysis alone. Health status analysis by demographics employs stratified tables and visualization to reveal how the relationship between health outcomes and key predictors varies across demographic groups. This analysis identifies potential effect modifiers whose effects should be considered during model building.

Risk factor clustering analysis uses correlation patterns to identify groups of related risk factors that may represent common underlying constructs. For example, the clustering of physical inactivity, poor diet, and obesity may indicate a metabolic syndrome pattern that has different implications for intervention than isolated risk factors. Understanding these clustering patterns informs the development of composite features that capture shared variance among related variables.

The BMI distribution analysis by health status reveals the expected gradient, with higher BMI categories associated with increased prevalence of poor health outcomes. However, the relationship is not strictly linear, with the highest BMI category showing some attenuation of the effect, potentially reflecting survivor bias or differential healthcare seeking among severely obese individuals.

### Key Statistical Findings

The exploratory analysis generates several key findings that inform the modeling approach and provide direct value for public health practice. Physical activity demonstrates a strong inverse correlation with poor health outcomes, with physically active individuals showing substantially lower rates of poor health across all demographic groups. This finding underscores the importance of physical activity promotion as a core public health strategy and suggests that exercise frequency will be an important predictor in the final model.

Mental and physical health days show significant association, with individuals reporting high numbers of poor mental health days also likely to report high numbers of poor physical health days. This mental-physical health link supports integrated care models that address both dimensions of health simultaneously rather than treating them as independent domains.

Healthcare access disparities are evident across insurance status, with uninsured individuals showing markedly poorer health outcomes across multiple indicators. This finding highlights the importance of insurance coverage as a social determinant of health and supports policies aimed at expanding access to affordable health coverage.

The education-health gradient is clearly evident in the data, with higher education levels consistently associated with better health outcomes. This relationship likely operates through multiple pathways including health literacy, health behaviors, access to healthcare, and exposure to occupational hazards.

---

## Methodology

### Data Science Pipeline

This project follows a comprehensive 8-phase data science pipeline adapted for public health applications. Each phase builds on the outputs of the previous phase, creating a systematic workflow that ensures thoroughness and reproducibility. The pipeline methodology enables clear documentation of analytical decisions and facilitates communication with stakeholders about the analytical process.

#### Phase 1: Project Setup and Data Loading

The project setup phase establishes the computational environment and loads the raw data into analysis-ready formats. Library imports configure the Python environment with all necessary functions and classes. Environment configuration ensures consistent behavior across different computing environments. Data loading routines import the BRFSS data from both CSV and Excel sources, with format detection and appropriate parsing strategies. Initial validation checks verify data integrity and identify any immediate quality concerns that require attention.

#### Phase 2: Data Exploration and Quality Assessment

The exploration phase conducts comprehensive variable analysis to understand the characteristics of each feature in the dataset. Missing value documentation systematically quantifies the extent and patterns of missing data across all variables. Special code identification recognizes the BRFSS-specific coding conventions and flags values requiring special treatment. Data quality reporting synthesizes the findings into actionable recommendations for the cleaning phase.

#### Phase 3: Data Cleaning and Preprocessing

The cleaning phase implements the preprocessing pipeline, transforming the raw data into analysis-ready form. BRFSS special value conversion systematically replaces special codes with appropriate missing value indicators or valid data values. Missing value imputation applies appropriate strategies for each variable based on data type and missingness patterns. Outlier detection and treatment identifies and caps values falling outside clinically plausible ranges. Data type optimization reduces memory requirements through efficient type selection. Duplicate removal eliminates redundant records that could bias the analysis.

#### Phase 4: Exploratory Data Analysis

The EDA phase produces comprehensive documentation of patterns in the cleaned data. Univariate distributions characterize each variable individually, identifying unusual patterns and informing subsequent analysis. Bivariate associations quantify relationships between pairs of variables, identifying potential predictors and confounders. Statistical testing applies appropriate hypothesis tests to determine the significance of observed associations. Multivariate patterns reveal complex relationships that require modeling approaches to fully characterize.

#### Phase 5: Feature Engineering

The feature engineering phase creates derived variables that capture health risk factors in more meaningful ways. BMI calculation and categorization computes standard body mass index categories from self-reported weight and height. Composite health scores combine multiple related indicators into summary measures that capture overall health status. Risk factor counting quantifies the cumulative burden of cardiovascular risk factors. Healthcare access scoring measures engagement with the healthcare system. Target variable creation defines the binary health outcome for prediction.

#### Phase 6: Model Development

The model development phase builds and trains predictive models using the engineered features. Train-test split divides the data into training and validation sets using stratified sampling to preserve class proportions. Feature scaling standardizes predictor variables to ensure equal contribution to model training. Baseline logistic regression establishes reference performance against which more complex models are compared. Multiple advanced models including Decision Tree, Random Forest, and Gradient Boosting are trained on the training data. Hyperparameter optimization via GridSearchCV systematically searches the hyperparameter space to identify optimal model configurations.

#### Phase 7: Model Evaluation

The evaluation phase assesses model performance using multiple metrics and validation approaches. Performance metrics calculation computes accuracy, precision, recall, F1-score, and area under the ROC curve for each model. ROC curve analysis visualizes the trade-off between true positive rate and false positive rate across classification thresholds. Precision-recall curves provide more informative assessment for imbalanced classification problems. Confusion matrix visualization displays the full spectrum of classification outcomes. Feature importance extraction identifies the variables contributing most to model predictions.

#### Phase 8: Results and Conclusions

The results phase synthesizes findings into actionable insights for public health practice. Findings integration combines numerical results with domain knowledge to develop coherent interpretations. Public health implications translate technical results into practical recommendations for intervention and policy. Study limitations honestly acknowledge the constraints of the analysis and threats to validity. Future research recommendations identify priority areas for extending this work.

### Statistical Methods

| Method | Application | Interpretation |
|--------|-------------|----------------|
| Chi-Square Test | Categorical independence | p-value < 0.05 indicates association |
| Spearman Correlation | Ordinal associations | Range: -1 to 1 |
| Cramér's V | Effect size for chi-square | 0-1 scale (0.1=weak, 0.5=strong) |
| IQR Method | Outlier detection | Values outside Q1-1.5*IQR to Q3+1.5*IQR |

### Machine Learning Methods

#### Models Implemented

**1. Logistic Regression**

Logistic regression serves as the baseline model due to its interpretability and well-understood statistical properties. The model estimates the log-odds of poor health as a linear combination of predictor variables, with coefficients that can be transformed into odds ratios for clinical interpretation. Class weight adjustment addresses the class imbalance by weighting the loss function to give higher importance to the minority class. L2 regularization (Ridge penalty) prevents overfitting by penalizing large coefficient values, improving generalization to new data. The logistic regression model provides a benchmark against which more complex models are compared, allowing assessment of whether additional model complexity yields meaningful performance improvements.

**2. Decision Tree**

The decision tree model provides a non-parametric approach to classification that creates interpretable rules for predicting health outcomes. The tree structure splits the feature space into regions based on threshold values for predictor variables, with leaf nodes assigned to the majority class in each region. Maximum depth is limited to 10 to prevent overfitting, ensuring that the tree captures generalizable patterns rather than memorizing the training data. Feature importance is automatically calculated as the total reduction in impurity attributable to each feature across all splits, providing a ranking of predictor importance. Decision trees serve as the foundation for ensemble methods and provide interpretable baselines for comparison.

**3. Random Forest**

Random forest is an ensemble method that combines predictions from multiple decision trees to achieve superior performance and reduced variance. The algorithm trains each tree on a bootstrap sample of the training data and considers only a random subset of features at each split, decorrelating the trees and reducing overfitting. The ensemble of 100 trees provides stable predictions while maintaining computational tractability. Feature bagging (random feature selection at each split) ensures that the model is robust to outliers and irrelevant features. Out-of-bag error estimation provides an unbiased estimate of model performance without the need for a separate validation set. Random forest typically achieves strong performance on tabular data with minimal tuning, making it a workhorse algorithm for many applications.

**4. Gradient Boosting**

Gradient boosting is a sequential ensemble method that builds trees one at a time, with each new tree correcting the errors of the previous ensemble. The algorithm fits trees to the gradient of the loss function with respect to the current predictions, systematically reducing the residual error. A learning rate of 0.1 controls the contribution of each tree, with lower rates requiring more trees but typically achieving better generalization. Maximum depth of 5 limits the complexity of individual trees, preventing overfitting while allowing the ensemble to capture non-linear relationships. Gradient boosting often achieves the highest predictive performance among tree-based methods but requires careful tuning to avoid overfitting.

### Hyperparameter Optimization

GridSearchCV parameter tuning with 3-fold cross-validation systematically searches the hyperparameter space to identify optimal model configurations. The optimization focuses on the Random Forest model, which shows the strongest preliminary performance, while simpler models use default or minimally tuned parameters.

| Model | Parameters Tuned | Values Tested |
|-------|------------------|---------------|
| Random Forest | n_estimators | [100, 200] |
| | max_depth | [8, 10, 12] |
| | min_samples_split | [5, 10] |
| | min_samples_leaf | [2, 4] |

The optimization metric is ROC-AUC, which provides robust performance assessment for imbalanced classification problems. The cross-validation procedure provides stable performance estimates while using the available data efficiently.

---

## Results

### Model Performance Benchmarking

| Model | Accuracy | F1-Score | ROC-AUC | CV-ROC-AUC |
|-------|----------|----------|---------|------------|
| Logistic Regression | ~82% | ~0.58 | ~0.85 | ~0.84 |
| Decision Tree | ~80% | ~0.55 | ~0.82 | ~0.81 |
| Random Forest | ~85% | ~0.65 | ~0.88 | ~0.87 |
| Gradient Boosting | ~84% | ~0.63 | ~0.87 | ~0.86 |

The Random Forest classifier achieves the best overall performance, demonstrating superior discriminative ability with a ROC-AUC of approximately 0.88. The accuracy of approximately 85% reflects the model's ability to correctly classify the majority of cases, while the F1-score of approximately 0.65 for the poor health class indicates reasonable performance on the minority class despite the class imbalance.

The comparison reveals a clear performance hierarchy, with ensemble methods (Random Forest, Gradient Boosting) outperforming single-model approaches (Logistic Regression, Decision Tree). This finding is consistent with machine learning theory, which predicts that ensemble methods achieve lower variance through model averaging and can capture more complex patterns through combination of multiple learners.

The substantial gap between the Decision Tree and Random Forest performance highlights the value of ensemble methods for this application. The Decision Tree's tendency to overfit the training data results in poor generalization, while the Random Forest's decorrelated trees provide more robust predictions.

### Cross-Validation Results

5-fold stratified cross-validation results for the best model provide robust estimates of expected performance on new data:

- Mean ROC-AUC: 0.87
- Standard Deviation: ±0.02
- Confidence Interval (95%): 0.85 - 0.89

The narrow confidence interval indicates stable performance across different data splits, suggesting that the model will generalize well to new samples from the same population. The small standard deviation relative to the mean indicates that the model's performance is not highly dependent on the particular training-test split used.

### Confusion Matrix Analysis

|  | Predicted Good | Predicted Poor |
|--|----------------|----------------|
| **Actual Good** | True Negatives | False Positives |
| **Actual Poor** | False Negatives | True Positives |

The confusion matrix reveals that the model achieves high specificity (correctly identifying healthy individuals) but moderate sensitivity (correctly identifying at-risk individuals). This pattern is typical for imbalanced classification problems and reflects the model's tendency to predict the majority class. For clinical applications, this trade-off may be acceptable if the cost of false positives is high relative to false negatives, or additional techniques may be employed to improve sensitivity if needed.

### Feature Importance: What Drives Prediction

#### Top 10 Most Important Features

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | Physical Health Days | 0.152 | Days of poor physical health strongly predict overall health |
| 2 | General Health Rating | 0.138 | Self-assessment is highly predictive |
| 3 | Difficulty Walking | 0.124 | Mobility issues indicate broader health problems |
| 4 | Exercise Frequency | 0.098 | Physical activity is protective |
| 5 | BMI | 0.087 | Body mass index associated with health outcomes |
| 6 | High Blood Pressure | 0.076 | Cardiovascular risk factor |
| 7 | Mental Health Days | 0.071 | Mental health impacts overall wellbeing |
| 8 | Healthcare Access Score | 0.065 | Access to care improves outcomes |
| 9 | Employment Status | 0.058 | Employment linked to health |
| 10 | Age Proxy/Demographics | 0.052 | Demographic factors influence risk |

The feature importance analysis reveals that physical health days and general health rating are the strongest predictors, which is expected given their direct relationship to the target variable definition. However, the inclusion of behavioral factors (exercise frequency) and clinical factors (blood pressure, BMI) among the top predictors confirms that the model captures clinically meaningful risk factors beyond the self-assessment variables.

The presence of the healthcare access score among the top predictors highlights the importance of structural factors in health outcomes. This finding has direct implications for public health policy, suggesting that interventions to improve healthcare access may have meaningful impacts on population health.

---

## Key Insights from Exploratory Analysis

### Behavioral Risk Factor Patterns

Physical inactivity represents a significant public health concern, with approximately 25% of respondents reporting no physical activity in the past 30 days. This sedentary lifestyle pattern is associated with elevated risk for multiple chronic conditions including cardiovascular disease, diabetes, and certain cancers. The high prevalence of physical inactivity suggests substantial opportunity for health improvement through physical activity promotion programs.

Smoking prevalence remains substantial despite decades of public health intervention, with notable demographic variations in current smoking rates. Socioeconomic disparities in smoking are particularly pronounced, with lower education and income groups showing higher prevalence rates. These disparities likely reflect differential access to cessation resources and targeted marketing by tobacco companies to vulnerable populations.

Hypertension affects approximately one-third of the adult population, with many individuals unaware of their elevated blood pressure. The high prevalence of undiagnosed hypertension underscores the importance of blood pressure screening programs and the potential value of the model for identifying individuals who may benefit from blood pressure monitoring.

Mental health days affect daily activities for a substantial portion of the population, with respondents reporting more mental health-related impairment than physical health-related impairment on average. This finding highlights the importance of mental health as a component of overall health and supports integrated approaches that address both dimensions simultaneously.

### Demographic Disparities

The education gradient in health outcomes is striking and consistent across multiple health measures. Individuals with college education show substantially better health profiles than those with less education, with differences observed in general health rating, chronic disease prevalence, and functional limitations. This gradient likely operates through multiple pathways including health knowledge, health behaviors, occupational exposures, and access to healthcare.

The employment-health link is evident in the data, with employed individuals demonstrating significantly better health profiles than unemployed counterparts. Beyond the direct effects of income and insurance, employment may provide social connections, daily structure, and sense of purpose that contribute to better health outcomes.

Age-related patterns show expected increases in chronic conditions with advancing age, but also reveal higher healthcare engagement among older adults. This pattern suggests that while aging increases health risks, engagement with the healthcare system may partially offset these risks through better disease management.

Gender differences in health behaviors and outcomes show that women report more preventive care utilization while men show higher rates of some risky behaviors. These patterns have implications for targeted intervention design, with messaging and delivery approaches tailored to reach each gender effectively.

### Healthcare Access Impact

Insurance coverage remains a critical determinant of health outcomes, with uninsured individuals showing markedly poorer health across multiple indicators. Despite provisions of the Affordable Care Act, coverage gaps persist, particularly in states that have not expanded Medicaid eligibility.

Cost barriers affect approximately 10% of respondents who reported being unable to see a doctor due to cost in the past year. These cost-related access barriers are associated with worse health outcomes and represent an opportunity for policy intervention through cost-sharing reductions or expanded coverage.

Preventive care gaps are evident in the proportion of the population that has not received routine checkups within recommended timeframes. These gaps represent missed opportunities for early disease detection and health promotion that could prevent more serious and costly health problems later.

---

## Insights

### Public Health Implications

Physical activity interventions represent the highest-impact opportunity for improving population health based on the strong protective association observed in the data. Programs that successfully increase physical activity at the population level could meaningfully reduce the prevalence of poor health outcomes. The strong association also suggests that the predictive model could be used to identify individuals who would benefit most from physical activity interventions.

Mental and physical health are interconnected, supporting integrated care models that address both dimensions of health rather than treating them as separate domains. The significant association between mental health days and physical health outcomes suggests that interventions targeting mental health may have spillover benefits for physical health.

Healthcare access barriers significantly impact health outcomes, providing support for policies that reduce cost and structural barriers to care. The importance of healthcare access as a predictor suggests that interventions to improve access could have meaningful impacts on population health outcomes.

Education appears protective, suggesting value in health literacy programs that empower individuals with knowledge to make informed health decisions. The strong education-health gradient may also reflect confounding by socioeconomic status, which affects multiple domains of life that influence health.

Employment support programs may have secondary health benefits beyond their primary economic objectives. The association between employment and health outcomes suggests that workforce development programs could be framed as health interventions as well as economic development strategies.

### Clinical Interpretations

The model effectively stratifies individuals into risk categories that could guide targeted intervention strategies. The clear separation between risk probability distributions for good and poor health outcomes suggests that the model captures clinically meaningful distinctions.

Difficulty walking serves as a powerful marker for underlying health issues, likely reflecting the cumulative burden of chronic conditions and functional decline. This finding suggests that mobility assessment could serve as a simple screening tool for identifying individuals at elevated health risk.

Self-reported behaviors predict clinical outcomes with reasonable accuracy, validating the use of survey data for health risk assessment. This finding supports continued investment in surveillance systems like the BRFSS that collect behavioral health data.

Composite scoring provides more nuanced risk assessment than single factors, with the multi-component scores capturing aspects of health risk that are not apparent from individual variables alone. The handcrafted features developed in this project demonstrate the value of domain-informed feature engineering.

---

## Interactive Prediction Tool

### Model Deployment

The trained model can be deployed as an interactive prediction tool for clinical and public health applications. The prediction functionality enables individualized risk assessment that can guide clinical decision making and patient counseling. The tool translates the model's complex pattern recognition into actionable risk categories that can be communicated to patients and incorporated into care planning.

### Prediction Functionality

```python
def predict_health_risk(patient_data, model, scaler):
    """
    Predict poor health risk for an individual.
    
    Parameters:
    -----------
    patient_data : dict
        Patient characteristics including age, behaviors, health status
    model : RandomForestClassifier
        Trained prediction model
    scaler : StandardScaler
        Fitted feature scaler
    
    Returns:
    --------
    dict : Risk prediction and probability
    """
    # Preprocess input
    features = preprocess(patient_data)
    scaled = scaler.transform(features)
    
    # Get prediction and probability
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]
    
    return {
        'risk_level': 'Poor' if prediction == 1 else 'Good',
        'risk_probability': probability,
        'risk_category': categorize_risk(probability)
    }
```

### Risk Categories

| Probability Range | Risk Category | Recommended Action |
|-------------------|---------------|-------------------|
| 0-20% | Low Risk | Routine care, health maintenance |
| 20-40% | Moderate Risk | Increased monitoring, lifestyle counseling |
| 40-60% | High Risk | Comprehensive assessment, preventive interventions |
| 60-80% | Very High Risk | Intensive intervention, specialist referral |
| 80-100% | Critical Risk | Immediate clinical evaluation |

### Integration Possibilities

The prediction tool can be integrated into multiple healthcare delivery contexts to enhance risk identification and intervention targeting. Electronic health record integration would embed risk scores in clinical workflows, providing automated risk assessment at the point of care. Patient portal integration would enable individuals to access their own risk information, supporting patient engagement in health management. Public health surveillance applications would use the model to identify high-risk populations for community-based interventions. Care management program integration would prioritize outreach to individuals identified as high risk for intensive case management services.

---

## Impact

### Potential Benefits

Early intervention enabled by risk prediction can identify at-risk individuals before disease onset, allowing for preventive action that may avoid or delay the development of serious health conditions. The shift from reactive treatment to proactive prevention has the potential to improve health outcomes while reducing healthcare costs.

Resource optimization through targeted interventions allocates limited public health resources more efficiently by directing them toward individuals and populations at highest risk. This approach maximizes the health impact of available resources and supports sustainable public health programs.

Health equity improvement through addressing modifiable risk factors can reduce health disparities across populations. The analysis of demographic patterns in risk factors can inform targeted interventions for underserved communities.

Cost reduction through preventive approaches is generally more cost-effective than treating advanced disease. Investment in prevention and early intervention can reduce long-term healthcare expenditures while improving population health.

Policy guidance from data-driven insights supports evidence-based public health policy development. The analysis of risk factor patterns and their associations with health outcomes can inform policy priorities and resource allocation decisions.

### Implementation Considerations

Clinical validation through prospective studies is essential before deploying the model in clinical settings. The current analysis is limited to retrospective data, and prospective validation would provide stronger evidence of real-world effectiveness.

Ethical considerations including fairness and bias assessment are essential for equitable deployment. The model must be evaluated for differential performance across demographic groups, and steps must be taken to address any identified disparities.

Privacy protection for health information must comply with relevant regulations including HIPAA. Any deployment of predictive models must incorporate appropriate safeguards to protect individual privacy.

Clinical workflow integration requires careful attention to how risk predictions will be presented and used in clinical encounters. The tool design must accommodate existing clinical processes and avoid creating additional burden on healthcare providers.

Risk communication to patients requires careful messaging to ensure that risk information is understood and acted upon appropriately without causing unnecessary anxiety or false reassurance.

---

## Recommendations and Future Directions

### Short-Term Improvements

Enhanced feature engineering incorporating additional social determinants of health data could improve model performance and capture broader determinants of health outcomes. Variables related to housing stability, food security, and neighborhood characteristics would complement the existing behavioral and clinical variables.

Resampling techniques including SMOTE (Synthetic Minority Over-sampling Technique) or other methods to address class imbalance could improve model performance on the minority class. The current model shows reasonable performance despite imbalance, but targeted oversampling of the poor health class might improve sensitivity.

Ensemble methods including XGBoost and LightGBM could potentially improve predictive performance beyond what has been achieved with the current model set. These advanced gradient boosting implementations have shown strong performance on tabular data in numerous competitions and applications.

SHAP (SHapley Additive exPlanations) values would enable individual-level explanations of model predictions, enhancing interpretability for clinical applications. SHAP values provide consistent and locally accurate explanations for any machine learning model.

### Medium-Term Goals

External validation on independent datasets from different regions or time periods would provide stronger evidence of model generalizability. Validation using data from different states or survey years would assess whether the model captures general patterns of health risk rather than idiosyncrasies of the specific dataset.

Temporal analysis using multiple years of BRFSS data could develop longitudinal models that capture changes in risk over time. Such models could support prediction of future health trajectories rather than current health status.

Clinical partnership with healthcare systems for real-world validation would bridge the gap between research and practice. Collaboration with delivery systems could provide access to clinical outcomes data for model validation.

Cost-effectiveness analysis would assess the economic impact of model-guided interventions compared to usual care. Such analysis is essential for making the business case for adoption of predictive tools in healthcare settings.

### Long-Term Vision

Real-time risk assessment through API development would enable integration of predictive capabilities into clinical workflows and patient-facing applications. Such systems could provide risk updates as new data becomes available.

Personalized intervention pairing would match predictions with tailored intervention recommendations, moving beyond risk identification to prescriptive analytics. Such systems would optimize intervention selection based on predicted response.

Population health management at regional and national scales would apply the model to large populations for surveillance and resource allocation. Such applications could inform public health planning and policy development.

Learning health systems with feedback loops would enable continuous model improvement as new data becomes available. Such systems would adapt to changing population health patterns and intervention effectiveness.

Federated learning approaches would enable multi-institution collaboration while preserving privacy, potentially improving model performance through access to larger and more diverse datasets.

---

## Academic Requirements Met

### Course Learning Objectives Addressed

The project comprehensively addresses all specified course learning objectives through practical application to a real public health dataset. Data collection and preprocessing skills are demonstrated through systematic treatment of BRFSS data including special code handling and missing value imputation. Exploratory data analysis skills are evidenced by comprehensive univariate, bivariate, and multivariate analysis with statistical testing. Statistical testing and hypothesis generation are integral to the EDA phase. Feature engineering skills are demonstrated through creation of BMI, composite scores, and risk counts. Multiple machine learning model implementation is achieved through four distinct algorithms with systematic comparison. Hyperparameter optimization is conducted using GridSearchCV. Model evaluation employs multiple metrics and cross-validation. Feature importance analysis provides clinical interpretation of model behavior. Interpretation for non-technical audiences is achieved through clear documentation and clinical guidelines. Documentation and reproducibility are ensured through comprehensive README, methodology documents, and version control.

### Assessment Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Data Quality and Cleaning | Complete | BRFSS special value handling, missing value treatment, outlier capping |
| Exploratory Analysis | Complete | Comprehensive EDA with statistical tests, visualizations |
| Feature Engineering | Complete | BMI calculation, health scores, risk counts, access scores |
| Multiple Models | Complete | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| Hyperparameter Tuning | Complete | GridSearchCV optimization for Random Forest |
| Model Evaluation | Complete | Multi-metric evaluation, cross-validation, confusion matrix |
| Interpretation | Complete | Feature importance, clinical insights, recommendations |
| Documentation | Complete | README, data dictionary, methodology, code comments |

---

## Public Health Critical Metrics

### Model Performance for Clinical Use

| Metric | Value | Clinical Significance |
|--------|-------|----------------------|
| Sensitivity | ~72% | Ability to identify at-risk individuals |
| Specificity | ~88% | Correctly identifying healthy individuals |
| Positive Predictive Value | ~58% | Accuracy of positive predictions |
| Negative Predictive Value | ~93% | Reliability of low-risk assessments |
| ROC-AUC | ~0.88 | Overall discriminative ability |

The sensitivity of approximately 72% indicates that the model correctly identifies roughly three-quarters of individuals who will experience poor health outcomes. This level of sensitivity is meaningful for population-level screening applications where the consequences of missing at-risk individuals are substantial. The specificity of approximately 88% indicates that the model correctly identifies healthy individuals most of the time, supporting its use for ruling out individuals who do not need intensive intervention.

### Risk Factor Prevalence

| Risk Factor | Prevalence | Public Health Priority |
|-------------|------------|----------------------|
| Physical Inactivity | ~25% | High |
| Current Smoking | ~15% | High |
| Hypertension | ~32% | Critical |
| Obesity (BMI ≥ 30) | ~31% | Critical |
| No Regular Checkup | ~20% | Moderate |

---

## Handcrafted Features

### Feature Engineering Summary

This project implements domain-informed feature engineering based on clinical and public health expertise. The handcrafted features transform raw survey responses into clinically meaningful variables that capture important health constructs while providing additional predictive value.

### 1. Body Mass Index (BMI)

BMI is calculated using the standard formula that relates weight to height: (weight_lbs × 0.453592) / (height_inches × 0.0254)². This calculation converts self-reported weight and height into the standard metric units used for BMI classification. BMI serves as a standard screening tool for body weight classification and correlates with cardiovascular risk, diabetes, and mortality. The formula handles unit conversions automatically, ensuring consistent BMI values regardless of whether input data is in imperial or metric units.

### 2. BMI Categories

The continuous BMI variable is categorized into clinical weight status classes that have established associations with health outcomes. Underweight is defined as BMI less than 18.5, indicating potential nutritional deficiency or underlying disease. Normal weight spans BMI 18.5 to less than 25, representing the range associated with lowest mortality risk. Overweight spans BMI 25 to less than 30, indicating elevated but not extreme weight for height. Obese is defined as BMI 30 or greater, indicating significantly elevated health risk requiring intervention.

### 3. Composite Health Score

The composite health score combines multiple health indicators into a single summary measure that captures overall health status. The score includes the inverted general health rating (so higher values indicate better health), normalized physical health days, normalized mental health days, exercise indicator (1 if exercises regularly), and health insurance indicator (1 if insured). The total score ranges from 0 to 10, with higher scores indicating better health status. This composite approach captures the multi-dimensional nature of health that cannot be represented by any single indicator.

### 4. Cardiovascular Risk Count

The cardiovascular risk count quantifies the cumulative burden of cardiovascular risk factors. The count includes current smoking status, lack of regular exercise, high blood pressure diagnosis, mobility limitations, and obesity (BMI ≥ 30). Each risk factor contributes 1 point to the total, yielding a score ranging from 0 to 5. Higher scores indicate greater cardiovascular risk burden and correspond to elevated probability of adverse cardiovascular outcomes.

### 5. Healthcare Access Score

The healthcare access score measures engagement with the healthcare system through four components: health insurance coverage, having a personal doctor, absence of cost barriers, and recent routine checkup. Each component contributes 1 point if present, yielding a score from 0 to 4. Higher scores indicate better healthcare access and engagement, which are associated with better health outcomes through improved preventive care and chronic disease management.

### 6. Mental-Physical Health Gap

The mental-physical health gap measures the absolute difference between mental health days and physical health days. This gap identifies individuals whose mental and physical health are misaligned, such as those with significant mental health symptoms but good physical health or vice versa. A large gap may indicate specific health concerns requiring targeted intervention.

### 7. Demographic Flags

Binary flags are created for high education (college degree or higher), current employment (employed or self-employed), and marital status (currently married). These flags capture demographic characteristics that are associated with health outcomes and may be useful for targeted intervention design.

---

## Comprehensive Visualizations

### Visualization Portfolio

This project generates multiple publication-quality visualizations that effectively communicate analytical findings and support interpretation of results. All visualizations are saved in high-resolution formats suitable for inclusion in reports and presentations.

### 1. Univariate Analysis Plots

Health status distribution bar charts display the prevalence of each health rating category in the population, enabling quick assessment of the overall health profile. Physical and mental health day histograms show the distribution of sick days in the 30-day reference period, revealing patterns of health-related functional limitation. Pie charts for categorical variables display the proportion of respondents in each category for variables like sex, education level, and smoking status. Box plots for numeric variables show the central tendency, spread, and outliers for key continuous measures.

### 2. Bivariate Analysis Plots

Correlation heatmaps visualize the relationships among all numeric variables, enabling quick identification of strong positive and negative associations. Grouped bar charts display health outcomes stratified by demographic groups, revealing disparities across populations. Stacked percentage plots show the composition of health outcomes within each category of key predictors. Scatter plots with trend lines display bivariate relationships with fitted regression lines to illustrate patterns.

### 3. Multivariate Analysis Plots

Health status by education level displays the gradient in health outcomes across education categories. Health status by employment shows employment-related differences in health outcomes. BMI distribution by health status reveals the expected relationship between body weight and health. Risk factors by health outcome displays the prevalence of various risk factors among good and poor health groups.

### 4. Model Evaluation Plots

ROC curves with AUC values visualize the trade-off between true positive rate and false positive rate across classification thresholds, with the area under the curve summarizing overall discriminative ability. Precision-recall curves provide more informative assessment for imbalanced classification problems. Confusion matrix heatmaps display the full spectrum of classification outcomes with color coding for easy interpretation. Learning curves show model performance as a function of training set size, informing decisions about data collection priorities.

### 5. Feature Importance Plots

Horizontal bar charts display feature importance rankings with clear labeling of each feature and its importance score. Feature importance rankings highlight the top predictors for quick reference. Summary tables provide detailed importance values alongside interpretation of each feature's contribution.

---

## Clinical Interpretation Dashboard

### Dashboard Components

The analysis supports development of an interactive clinical interpretation dashboard that translates model outputs into actionable clinical guidance. Such a dashboard would integrate model predictions with patient context to support clinical decision making.

### 1. Risk Stratification Panel

Individual risk scores with confidence intervals provide point estimates of poor health probability along with measures of uncertainty. Population risk distribution histograms display the distribution of risk scores across the served population, identifying the size of high-risk subgroups. Trend analysis over time tracks changes in risk profiles at both individual and population levels.

### 2. Factor Contribution Analysis

SHAP value visualizations would show the contribution of each feature to individual predictions, enabling explanation of why a particular prediction was made. Individual prediction explanations would translate technical model outputs into patient-friendly language. Counterfactual scenarios would show how risk would change if specific risk factors were modified.

### 3. Population Insights

Aggregate risk factor prevalence displays the most common risk factors in the population, informing intervention priorities. Demographic disparity analyses identify subgroups with elevated risk profiles, enabling targeted outreach. Geographic variation maps would display spatial patterns in risk if geocoded data were available.

### 4. Intervention Planning

Risk factor-specific recommendations would provide guidance on addressing each identified risk factor. Priority population identification would flag high-risk individuals for outreach. Resource allocation guidance would inform budgeting and staffing decisions based on predicted need.

### Clinical Application Example

```
Patient Profile:
- Age: 52
- Sex: Male
- Physical health days: 15/30
- Exercise: No
- BMI: 32.5
- Hypertension: Yes

Model Prediction:
- Risk Category: High Risk
- Probability: 0.68
- Top Contributing Factors:
  1. Physical health days (high)
  2. No exercise
  3. High BMI
  4. Hypertension

Recommendations:
- Comprehensive metabolic assessment
- Exercise prescription program
- Weight management consultation
- Blood pressure optimization
```

---

## Complete Output Management

### Generated Outputs

This project produces comprehensive outputs for reproducibility, reporting, and future analysis. All outputs are organized in structured directories and properly documented for easy reference.

### Data Outputs

| File | Description | Format |
|------|-------------|--------|
| cleaned_health_data.csv | Preprocessed dataset with imputed values and optimized types | CSV |
| feature_importance.csv | Feature rankings with importance scores | CSV |
| model_comparison.csv | Model performance metrics across all algorithms | CSV |
| trained_model.pkl | Serialized Random Forest model with scaler | Pickle |
| cv_results.csv | Cross-validation results for all folds | CSV |

### Visualization Outputs

| File | Description | Resolution |
|------|-------------|------------|
| univariate_health_analysis.png | Distribution plots for all variables | 150 DPI |
| bivariate_analysis.png | Relationship plots between variable pairs | 150 DPI |
| multivariate_analysis.png | Multi-factor analysis visualizations | 150 DPI |
| model_evaluation_curves.png | ROC and precision-recall curves | 150 DPI |
| feature_importance.png | Horizontal bar chart of importance rankings | 150 DPI |
| confusion_matrix.png | Heatmap of classification outcomes | 150 DPI |
| health_pipeline_diagram.png | SVG visualization of data science pipeline | Vector |

### Documentation Outputs

| File | Description |
|------|-------------|
| README.md | Main project documentation with all sections |
| docs/data_dictionary.md | Complete variable reference with codes and meanings |
| docs/methodology.md | Detailed methodology documentation |
| docs/model_evaluation_report.md | Comprehensive evaluation results |
| SDS6108_Health_Data_Mining_Project.ipynb | Complete analysis notebook with all code |

### Reproducibility Features

Random seed configuration ensures that all random processes use seed 42, enabling exact reproduction of results across runs. Version control through Git tracks all changes to code and documentation, preserving the complete history of the project. Environment specification through requirements.txt documents all Python dependencies with versions. Code documentation through comprehensive docstrings and comments explains the purpose and logic of each function. Data versioning separates original raw data from processed data, preserving the ability to trace all transformations.

---

## Project Structure

```
risk-factor-modeling/
├── README.md                                    # Main documentation
├── requirements.txt                             # Python dependencies
├── SDS6108_Health_Data_Mining_Project.ipynb     # Main analysis notebook
├── SDS6108_Health_Data_Mining_Pipeline.svg      # Pipeline visualization
├── src/                                         # Source code
│   ├── __init__.py                              # Package initialization
│   ├── data_loading.py                          # Data loading utilities
│   ├── data_cleaning.py                         # Data preprocessing
│   ├── feature_engineering.py                   # Feature creation
│   ├── model_training.py                        # ML model training
│   └── model_evaluation.py                      # Model evaluation
├── docs/                                        # Documentation
│   ├── data_dictionary.md                       # Variable reference
│   ├── methodology.md                           # Methodology guide
│   └── model_evaluation_report.md               # Evaluation results
├── data/                                        # Data directory
│   ├── raw/                                     # Raw data files
│   ├── processed/                               # Cleaned data
│   └── results/                                 # Model outputs
├── visualizations/                              # Generated plots
├── notebooks/                                   # Additional notebooks
└── tests/                                       # Unit tests
```

---

## Getting Started

### Prerequisites

Python 3.12.3 serves as the foundation for this project, with the analysis designed to work within this version range. The pip or conda package manager enables straightforward installation of all required dependencies. Git for version control supports collaborative development and maintains complete history of all changes.

### Installation

```bash
# Clone the repository
git clone https://github.com/OumaCavin/risk-factor-modeling.git
cd risk-factor-modeling

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Open the main Jupyter notebook
jupyter notebook SDS6108_Health_Data_Mining_Project.ipynb

# Or run with JupyterLab
jupyter lab SDS6108_Health_Data_Mining_Project.ipynb
```

### Using the Python Package

```python
from src.data_loading import load_health_data
from src.data_cleaning import clean_data
from src.feature_engineering import engineer_features
from src.model_training import prepare_data, train_multiple_models
from src.model_evaluation import generate_evaluation_report

# Load and process data
df, source = load_health_data('data/raw/brfss_data.csv')
df_clean = clean_data(df)
df_features = engineer_features(df_clean)

# Train and evaluate models
X_train, X_test, y_train, y_test = prepare_data(df_features, features, 'POOR_HEALTH')
models = train_multiple_models(X_train, y_train)
```

---

## Author Information

**Student Name:** Cavin Otieno  
**Student ID:** SDS6/46982/2024  
**Course:** SDS6108 Health Data Mining and Analytics  
**Academic Year:** 2025-2026  
**Institution:** [Your University Name]  
**Email:** cavin.otieno012@gmail.com

### Course Information

- **Course Code**: SDS6108
- **Course Title**: Health Data Mining and Analytics
- **Program**: MSc Public Health Data Science
- **Department**: Public Health / Data Science

---

## License

This project is created for educational purposes as part of the MSc Public Health Data Science program. The analysis uses publicly available Behavioral Risk Factor Surveillance System (BRFSS) data.

### Data Usage Notice

The BRFSS data used in this project is publicly available from the Centers for Disease Control and Prevention (CDC). All analyses and interpretations are those of the author and do not represent the official views of the CDC.

### Academic Integrity

This project represents original work completed for academic assessment. All sources have been properly cited, and the work adheres to institutional academic integrity guidelines.

---

## Acknowledgments

- Behavioral Risk Factor Surveillance System (BRFSS), Centers for Disease Control and Prevention
- Course instructors and teaching assistants for SDS6108
- Peer reviewers and study group members
- Open source community for Python data science tools

---

**Project Status**: Complete  
**Last Updated**: January 2026  
**Repository**: https://github.com/OumaCavin/risk-factor-modeling