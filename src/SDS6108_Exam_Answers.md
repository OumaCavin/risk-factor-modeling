# MSc Public Health Data Science
## SDS6108: HEALTH DATA MINING AND ANALYTICS - Exam Answers

**Name:** Cavin Otieno Ouma  
**Registration Number:** SDS6/46982/2024  
**Institution:** University of Nairobi, Department of Mathematics

---

## QUESTION 1 (20 Marks)

### A) How a public health officer can use data analytics to mitigate malaria impacts [8 marks]

1. **Surveillance and Early Warning Systems**: Analyze historical malaria data to predict outbreaks based on seasonal patterns, rainfall, and temperature data.

2. **Resource Allocation**: Use data to identify high-burden areas and allocate insecticide-treated nets, antimalarial drugs, and healthcare workers optimally.

3. **Vector Control Planning**: Analyze mosquito breeding patterns and population density data to target indoor residual spraying and larviciding efforts.

4. **Treatment Effectiveness Monitoring**: Track drug resistance patterns by analyzing treatment outcome data across regions.

5. **Risk Mapping**: Create geospatial models identifying vulnerable populations and high-transmission zones.

6. **Campaign Effectiveness**: Evaluate the impact of intervention programs (bed net distribution, awareness campaigns) using before-and-after data analysis.

7. **Supply Chain Optimization**: Predict demand for antimalarial medications and diagnostic kits to prevent stockouts.

8. **Real-time Monitoring**: Dashboard analytics for tracking case counts, mortality rates, and intervention coverage in real-time.

### B) Healthcare analytics transforming care from reactive to proactive [6 marks]

**Reactive care** responds to illness after it occurs, while **proactive care** prevents illness before it happens.

**Examples:**
1. **Predictive Risk Scoring**: Instead of treating diabetic complications, analytics can identify pre-diabetic patients from lab values and lifestyle data, enabling early intervention.

2. **Readmission Prevention**: Rather than repeatedly admitting heart failure patients, predictive models identify high-risk patients for intensive outpatient follow-up.

3. **Population Health Management**: Instead of treating individual flu cases, analyzing vaccination rates and demographic data helps target immunization campaigns before flu season.

### C) Main types of data analyzed in healthcare [6 marks]

1. **Clinical Data**: Electronic Health Records (EHR), diagnoses, lab results, vital signs, medications, treatment outcomes.

2. **Administrative/Claims Data**: Billing codes, insurance claims, hospital admissions, length of stay, costs.

3. **Patient-Generated Data**: Wearable device data, patient surveys, self-reported symptoms, lifestyle information.

4. **Genomic Data**: DNA sequences, genetic markers, pharmacogenomic profiles for personalized medicine.

5. **Imaging Data**: X-rays, CT scans, MRIs, pathology slides for diagnostic analysis.

6. **Public Health Data**: Disease registries, vital statistics, epidemiological surveillance data, environmental health data.

---

## QUESTION 2 (15 Marks)

### A) Five reasons for using machine learning for clinical diagnosis [5 marks]

1. **Pattern Recognition**: ML can identify subtle patterns in complex medical data that humans might miss.
2. **Consistency**: Provides standardized, unbiased diagnostic suggestions without fatigue or emotional influence.
3. **Speed**: Can analyze large volumes of patient data rapidly for faster diagnosis.
4. **Integration of Multiple Data Sources**: Can combine lab results, imaging, genetics, and symptoms simultaneously.
5. **Continuous Improvement**: Models improve with more data, leading to increasingly accurate diagnoses over time.

### B) Python Instructions [10 marks]

**i. Open a CSV file using pandas [2 marks]**
```python
import pandas as pd
df = pd.read_csv('data.csv')
```

**ii. Count null entries in age column [2 marks]**
```python
df['age'].isnull().sum()
```

**iii. Get number of records in the dataset [2 marks]**
```python
len(df)
# OR
df.shape[0]
```

**iv. Summary of categorical data [2 marks]**
```python
df.describe(include='object')
```

**v. Plot histogram of glucose column [2 marks]**
```python
df['glucose'].hist()
# OR
import matplotlib.pyplot as plt
plt.hist(df['glucose'])
plt.show()
```

---

## QUESTION 3 (15 Marks)

### A) Data analytics for monitoring eateries compliance [5 marks]

1. **Risk-Based Inspection Scheduling**: Analyze historical violation data to prioritize inspections for high-risk establishments.
2. **Trend Analysis**: Track compliance patterns over time to identify deteriorating establishments.
3. **Geospatial Analysis**: Map violation hotspots to allocate inspector resources efficiently.
4. **Predictive Modeling**: Predict which eateries are likely to have violations based on past performance, type, and location.
5. **Automated Reporting**: Generate compliance reports and dashboards for stakeholders and public transparency.

### B) Three ways to handle null values [3 marks]

```python
# Method 1: Drop rows with null values
df_clean = df.dropna()
# Removes all rows containing any null values

# Method 2: Fill with mean/median
df['noChildren'].fillna(df['noChildren'].mean(), inplace=True)
# Replaces nulls with the column's average value

# Method 3: Fill with a specific value (e.g., 0 or mode)
df['noChildren'].fillna(0, inplace=True)
# Replaces nulls with zero (useful when null implies absence)
```

### C) Repeatable train-test split with stratification [3 marks]

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    dataValues, dataY, 
    test_size=0.2, 
    random_state=42,  # Makes split repeatable
    stratify=dataY    # Ensures class distribution is maintained
)
```

### D) Performance Metrics [4 marks]

**i. Accuracy**: Proportion of correct predictions (TP+TN) out of all predictions. Good for balanced datasets.

**ii. Recall (Sensitivity)**: Proportion of actual positives correctly identified (TP/(TP+FN)). Important when missing positive cases is costly.

**iii. Balanced Accuracy**: Average of recall for each class. Useful for imbalanced datasets: (Sensitivity + Specificity)/2.

**iv. Precision**: Proportion of predicted positives that are actually positive (TP/(TP+FP)). Important when false positives are costly.

---

## QUESTION 4 (15 Marks)

### A) CategoryEncodingLayer [4 marks]

CategoryEncodingLayer is a Keras preprocessing layer that encodes categorical features into numerical representations. It supports:
- **One-hot encoding**: Creates binary vectors for each category
- **Multi-hot encoding**: For multiple categories per sample
- **Count encoding**: Frequency-based encoding

It automatically adapts to the vocabulary in your data during the `adapt()` call and is useful for integrating categorical preprocessing directly into the model pipeline.

### B) Four image preprocessing tasks [4 marks]

1. **Resizing**: Standardizing all images to a uniform dimension (e.g., 224×224) required by the model architecture.

2. **Normalization**: Scaling pixel values from 0-255 to 0-1 or -1 to 1 for faster convergence during training.

3. **Data Augmentation**: Applying random transformations (rotation, flip, zoom, brightness) to increase training data diversity and reduce overfitting.

4. **Grayscale Conversion**: Converting color images to single-channel grayscale when color information is not relevant.

### C) Conv2D layer parameters [3 marks]

```python
tf.keras.layers.Conv2D(64, (5,5), activation='relu')
```

- **64**: Number of filters (output channels). The layer will learn 64 different feature detectors.
- **(5,5)**: Kernel/filter size. Each filter is a 5×5 pixel window that slides across the input image.
- **activation='relu'**: Activation function applied after convolution. ReLU (Rectified Linear Unit) introduces non-linearity by outputting max(0, x).

### D) Steps to train COVID chest X-ray classifier [4 marks]

1. **Data Loading**: Use `ImageDataGenerator` or `tf.keras.utils.image_dataset_from_directory` to load images from the two directories (covid_positive, covid_negative) with automatic labeling.

2. **Preprocessing**: Resize images to uniform size, normalize pixel values, apply data augmentation for the training set.

3. **Model Architecture**: Build a CNN with Conv2D layers, MaxPooling layers, Flatten layer, Dense layers, and a final sigmoid output for binary classification.

4. **Training**: Compile the model with binary_crossentropy loss and adam optimizer, then fit on training data with validation split. Use callbacks for early stopping and model checkpointing.

---

## QUESTION 5 (15 Marks)

### A) Text processing definitions [5 marks]

**i. Corpus**: A collection of text documents used for training or analysis. Example: all patient clinical notes in a hospital.

**ii. Dictionary**: A mapping of unique words to integer indices. Each word in the vocabulary gets a unique ID.

**iii. OOV Token**: Out-Of-Vocabulary token. A placeholder for words not found in the dictionary, typically represented as `<OOV>` or `<UNK>`.

**iv. Token**: A single unit of text, typically a word or subword, obtained after splitting text. "Hello world" → ["Hello", "world"].

**v. Stemming**: Reducing words to their root form by removing suffixes. Example: "running", "runs", "ran" → "run".

### B) Dictionary and Count Vector [6 marks]

**Documents:**
1. Malaria is a common disease caused by a parasite of the Plasmodium genus
2. Common symptoms of malaria include fever, headache and fatigue
3. Other symptoms include rapid breathing and rapid heart rate

**i. Dictionary (alphabetically ordered) [3 marks]**

| Word | Index |
|------|-------|
| a | 1 |
| and | 2 |
| breathing | 3 |
| by | 4 |
| caused | 5 |
| common | 6 |
| disease | 7 |
| fatigue | 8 |
| fever | 9 |
| genus | 10 |
| headache | 11 |
| heart | 12 |
| include | 13 |
| is | 14 |
| malaria | 15 |
| of | 16 |
| other | 17 |
| parasite | 18 |
| plasmodium | 19 |
| rapid | 20 |
| rate | 21 |
| symptoms | 22 |
| the | 23 |

**ii. Count vector of third document [3 marks]**

Third document: "Other symptoms include rapid breathing and rapid heart rate"

Words: other(1), symptoms(1), include(1), rapid(2), breathing(1), and(1), heart(1), rate(1)

Count Vector (using dictionary indices):
| Index | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|-------|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|----|----|----|----|----|----|----|---|
| Count | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0  | 0  | 1  | 1  | 0  | 0  | 0  | 1  | 0  | 0  | 2  | 1  | 1  | 0  |

### C) Keras Text Tokenizer [4 marks]

The Keras Tokenizer is a text preprocessing utility that:
- **Tokenizes text**: Splits sentences into words (tokens)
- **Builds vocabulary**: Creates a word-to-index dictionary automatically
- **Handles OOV**: Assigns a special token for unknown words
- **Provides methods**: `fit_on_texts()` to build vocabulary, `texts_to_sequences()` to convert text to integer sequences, `texts_to_matrix()` for bag-of-words representation

```python
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
tokenizer.fit_on_texts(documents)
sequences = tokenizer.texts_to_sequences(documents)
```

---

## QUESTION 6 (30 Marks)

### a) Distinguish between terminologies [16 marks]

**a) Big data vs Noisy data [2 marks]**
- **Big Data**: Extremely large datasets characterized by Volume, Velocity, Variety, and Veracity that require specialized tools for processing.
- **Noisy Data**: Data containing errors, outliers, or irrelevant information that can affect analysis accuracy.

**b) Data mining vs Data science [2 marks]**
- **Data Mining**: The process of discovering patterns and extracting useful information from large datasets using specific algorithms.
- **Data Science**: A broader field encompassing data mining, statistics, machine learning, visualization, and domain expertise to derive insights.

**c) Information vs Knowledge [2 marks]**
- **Information**: Processed data that has context and meaning (e.g., "Sales increased 20% in Q4").
- **Knowledge**: Information combined with experience, interpretation, and understanding that enables decision-making.

**d) Model vs Pattern [2 marks]**
- **Model**: A mathematical representation that describes relationships in data and can make predictions (e.g., regression equation).
- **Pattern**: A recurring structure or regularity found in data (e.g., customers who buy bread often buy butter).

**e) Dataset vs Data warehouse [2 marks]**
- **Dataset**: A collection of related data organized in rows and columns for a specific analysis purpose.
- **Data Warehouse**: A centralized repository that stores historical data from multiple sources, optimized for querying and reporting.

**f) Tacit knowledge vs Explicit (In-tacit) knowledge [2 marks]**
- **Tacit Knowledge**: Personal, experiential knowledge that is difficult to articulate or transfer (e.g., clinical intuition).
- **Explicit Knowledge**: Documented, codified knowledge that can be easily shared (e.g., clinical guidelines, protocols).

**g) Algorithm vs Process [2 marks]**
- **Algorithm**: A specific set of step-by-step instructions to solve a computational problem (e.g., decision tree algorithm).
- **Process**: A broader sequence of activities or phases to achieve an objective (e.g., the KDD process).

**h) Trends vs Deviation [2 marks]**
- **Trends**: General direction or pattern of change over time (e.g., increasing hospital admissions during flu season).
- **Deviation**: Departure from normal patterns or expected values; anomalies or outliers.

### b) Significance in data mining [2 marks]

**a) Data source [1 mark]**: Determines data quality, reliability, and the types of patterns that can be discovered. Poor sources lead to poor insights.

**b) Analysis and interpretation [1 mark]**: Transforms raw patterns into actionable knowledge. Without proper interpretation, discovered patterns have no business value.

### c) Data mining vs Knowledge discovery debate [4 marks]

**a) Output [2 marks]**
- **Data Mining Output**: Patterns, models, associations, clusters, classifications
- **Knowledge Discovery Output**: Actionable knowledge, validated insights, decision support information

**b) Techniques [2 marks]**
- **Data Mining Techniques**: Classification, clustering, association rules, regression algorithms
- **Knowledge Discovery Techniques**: Includes data mining plus data cleaning, preprocessing, transformation, evaluation, and visualization

### e) Four aims for knowledge discovery in business [4 marks]

1. **Competitive Advantage**: Identify market trends and customer behavior patterns before competitors.
2. **Decision Support**: Provide evidence-based insights for strategic planning and operational decisions.
3. **Risk Management**: Detect fraud, predict failures, and identify potential threats early.
4. **Operational Efficiency**: Optimize processes, reduce costs, and improve resource allocation through data-driven insights.

### f) Six challenges in data mining and knowledge discovery [6 marks]

1. **Data Quality Issues**: Incomplete, inconsistent, or noisy data requires extensive cleaning and preprocessing.
2. **Scalability**: Handling massive datasets requires specialized algorithms and infrastructure.
3. **High Dimensionality**: Many features lead to the "curse of dimensionality," making pattern detection difficult.
4. **Privacy and Security**: Sensitive data (especially in healthcare) requires strict compliance with regulations.
5. **Interpretability**: Complex models (deep learning) produce results that are difficult to explain to stakeholders.
6. **Dynamic Data**: Patterns change over time, requiring continuous model updating and monitoring.

### g) Importance of pre-processing and curation phase [2 marks]

Pre-processing is critical because:
- **Data quality directly impacts model accuracy**: "Garbage in, garbage out" - flawed data produces unreliable results.
- **Proper preparation enables pattern discovery**: Clean, transformed data allows algorithms to identify meaningful patterns rather than noise.

---

## QUESTION 7 (20 Marks)

### a) Supervised Learning Algorithms [12 marks]

**a) Naive Bayes [2 marks]**
- **Characteristics**: Probabilistic classifier based on Bayes' theorem with independence assumption between features.
- **Strengths**: Fast, works well with small datasets, handles high-dimensional data, good for text classification.
- **Weaknesses**: Assumes feature independence (rarely true), poor with correlated features.

**b) Neural Networks [2 marks]**
- **Characteristics**: Interconnected nodes (neurons) organized in layers that learn complex non-linear relationships.
- **Strengths**: Handles complex patterns, excellent for image/text/speech, automatic feature extraction.
- **Weaknesses**: Requires large data, computationally expensive, "black box" interpretability issue.

**c) Logistic Regression [2 marks]**
- **Characteristics**: Linear model using sigmoid function for binary classification, outputs probabilities.
- **Strengths**: Simple, interpretable coefficients, fast training, good baseline model.
- **Weaknesses**: Assumes linear decision boundary, poor with complex non-linear relationships.

**d) Support Vector Machine (SVM) [2 marks]**
- **Characteristics**: Finds optimal hyperplane that maximizes margin between classes, uses kernel tricks for non-linear data.
- **Strengths**: Effective in high-dimensional spaces, memory efficient, works well with clear margins.
- **Weaknesses**: Not suitable for large datasets, sensitive to noise, difficult to interpret.

**e) K-Nearest Neighbour (KNN) [2 marks]**
- **Characteristics**: Instance-based learning that classifies based on majority vote of k nearest neighbors.
- **Strengths**: Simple, no training phase, naturally handles multi-class problems.
- **Weaknesses**: Slow prediction (computes all distances), sensitive to irrelevant features and k value.

**f) Random Forest [2 marks]**
- **Characteristics**: Ensemble of decision trees using bagging and random feature selection.
- **Strengths**: Handles overfitting well, robust to outliers, provides feature importance, works with missing data.
- **Weaknesses**: Less interpretable than single tree, can be slow with many trees.

### b) Knowledge Discovery Process Diagram [6 marks]

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE DISCOVERY PROCESS                       │
└─────────────────────────────────────────────────────────────────────┘

    ┌──────────┐    ┌──────────────┐    ┌────────────────┐
    │  DATA    │───►│   DATA       │───►│     DATA       │
    │ SELECTION│    │  CLEANING    │    │ TRANSFORMATION │
    └──────────┘    └──────────────┘    └────────────────┘
         │                │                     │
         ▼                ▼                     ▼
    Identify         Remove noise,         Normalize,
    relevant         handle missing        reduce dimensions,
    data sources     values, outliers      aggregate

                           │
                           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                      DATA MINING                              │
    │    Apply algorithms: Classification, Clustering,              │
    │    Association Rules, Regression                              │
    └──────────────────────────────────────────────────────────────┘
                           │
                           ▼
    ┌──────────────┐    ┌────────────────────┐
    │  PATTERN     │───►│   KNOWLEDGE        │
    │  EVALUATION  │    │   PRESENTATION     │
    └──────────────┘    └────────────────────┘
         │                      │
         ▼                      ▼
    Interpret,              Visualize,
    validate patterns       report insights
```

### c) Supervised vs Unsupervised Learning [2 marks]

| Aspect | Supervised Learning | Unsupervised Learning |
|--------|--------------------|-----------------------|
| Labels | Uses labeled training data | Uses unlabeled data |
| Goal | Predict outcomes | Discover hidden patterns |
| Examples | Classification, Regression | Clustering, Association |

### d) Regression vs Classification [4 marks]

**Regression Supervised Learning [2 marks]**
- Predicts continuous numerical values
- Output is a real number (e.g., price, temperature, blood pressure)
- Examples: Linear regression, polynomial regression
- Use case: Predicting patient hospital stay duration

**Classification Supervised Learning [2 marks]**
- Predicts categorical class labels
- Output is a discrete category (e.g., disease/no disease, malignant/benign)
- Examples: Logistic regression, decision trees, SVM
- Use case: Diagnosing whether a tumor is malignant or benign

---

## QUESTION 8 (20 Marks)

**Selected Application Area: Healthcare**

### a) Concept of data mining in healthcare [2 marks]

Healthcare data mining involves extracting meaningful patterns from medical databases, electronic health records, clinical trials, and administrative data to improve patient outcomes, reduce costs, and enhance clinical decision-making. It transforms raw healthcare data into actionable medical knowledge.

### b) Types of data sources in healthcare [2 marks]

1. **Electronic Health Records (EHR)**: Patient demographics, diagnoses, medications, lab results
2. **Medical Imaging**: X-rays, CT scans, MRI images
3. **Claims/Administrative Data**: Insurance claims, billing codes, hospital admissions
4. **Genomic Data**: DNA sequences, genetic markers
5. **Wearable/IoT Devices**: Heart rate monitors, glucose monitors, fitness trackers

### c) Data mining approach applicable to healthcare [2 marks]

**Predictive and Descriptive approaches** are both applicable:
- **Predictive**: For disease risk prediction, readmission forecasting, treatment outcome prediction
- **Descriptive**: For patient segmentation, disease pattern discovery, treatment pathway analysis

### d) Two relevant data mining techniques [4 marks]

**1. Classification [2 marks]**
- Used for disease diagnosis (diabetic/non-diabetic), risk stratification (high/low risk), treatment response prediction
- Algorithms: Decision trees, Random Forest, Neural Networks
- Example: Classifying chest X-rays as COVID-positive or negative

**2. Clustering [2 marks]**
- Used for patient segmentation, identifying disease subtypes, grouping similar treatment responses
- Algorithms: K-means, Hierarchical clustering
- Example: Identifying subgroups of diabetes patients with different characteristics

### e) Data patterns, models, and trends [3 marks]

- **Patterns**: Association between symptoms and diseases; medication interactions; comorbidity relationships
- **Models**: Predictive models for disease onset; risk scoring models; treatment response models
- **Trends**: Seasonal disease patterns; increasing antibiotic resistance; rising chronic disease prevalence

### f) Expected knowledge after interpretation [2 marks]

- Disease risk factors and their relative importance
- Optimal treatment protocols for specific patient profiles
- Early warning indicators for disease progression
- Cost-effective care pathways
- Population health insights for resource planning

### g) Influence on current/future operations [3 marks]

1. **Clinical Decision Support**: Models integrated into EHR systems provide real-time recommendations to clinicians
2. **Resource Optimization**: Predictive models help forecast patient volumes, staffing needs, and supply requirements
3. **Personalized Medicine**: Patient-specific risk profiles enable tailored prevention and treatment strategies

### h) Two software applications [2 marks]

**1. Python with scikit-learn/TensorFlow**
- Services: Data preprocessing, machine learning algorithms, deep learning for medical imaging, model evaluation

**2. RapidMiner/KNIME**
- Services: Visual workflow design, data integration, automated machine learning, model deployment, healthcare-specific templates

---

## QUESTION 9 (20 Marks)

### a) Four Architectural Models for Data Mining Systems [6 marks]

**1. No Coupling Architecture**
- Data mining system operates independently from the database
- Data is extracted and processed separately
- Simple but limited by data transfer overhead

**2. Loose Coupling Architecture**
- Data mining system fetches data from database via standard queries
- Some integration but limited optimization
- Database provides data; mining system processes independently

**3. Semi-tight Coupling Architecture**
- Some data mining primitives implemented in the database
- Improved performance through partial integration
- Balance between flexibility and efficiency

**4. Tight Coupling Architecture**
- Data mining fully integrated into the database system
- Mining operations optimized by query processor
- Best performance but most complex to implement

### b) Knowledge Discovery Process Diagram [10 marks]

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE DISCOVERY IN DATABASES (KDD)                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────┐
│   RAW DATA    │  Various sources: databases, files, web, sensors
└───────┬───────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  1. DATA SELECTION                                            │
│  - Identify relevant data sources                             │
│  - Select target data for analysis                            │
│  - Define scope and objectives                                │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  2. DATA PREPROCESSING                                        │
│  - Handle missing values                                      │
│  - Remove duplicates and noise                                │
│  - Detect and treat outliers                                  │
│  - Ensure data consistency                                    │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  3. DATA TRANSFORMATION                                       │
│  - Normalization and scaling                                  │
│  - Feature engineering                                        │
│  - Dimensionality reduction (PCA)                             │
│  - Data aggregation                                           │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  4. DATA MINING                                               │
│  - Apply mining algorithms                                    │
│  - Classification, Clustering, Association Rules              │
│  - Pattern extraction                                         │
│  - Model building                                             │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  5. INTERPRETATION/EVALUATION                                 │
│  - Validate discovered patterns                               │
│  - Assess model performance                                   │
│  - Remove redundant/trivial patterns                          │
│  - Statistical significance testing                           │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  6. KNOWLEDGE PRESENTATION                                    │
│  - Visualization of results                                   │
│  - Report generation                                          │
│  - Integration into decision support systems                  │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────┐
│   KNOWLEDGE   │  Actionable insights for decision-making
└───────────────┘
```

---

## QUESTION 10 (20 Marks)

### a) Algorithmic Factors [4 marks]

**1. Scalability [2 marks]**
- Determines if the algorithm can handle increasing data volumes efficiently
- Important for big data applications where datasets grow continuously
- Example: K-means scales better than hierarchical clustering for large datasets

**2. Interpretability [2 marks]**
- Determines if results can be understood by domain experts
- Critical in healthcare where decisions must be explainable
- Example: Decision trees are interpretable; neural networks are not

### b) Data Mining Approaches and Tasks [10 marks]

**a) Machine Discovery Approach [2 marks]**
- **Tasks**: Pattern discovery, anomaly detection, knowledge extraction
- **Application**: Discovers hidden patterns without predefined hypotheses; useful for exploratory analysis in unknown domains

**b) Probabilistic Approach [2 marks]**
- **Tasks**: Uncertainty modeling, risk assessment, missing data handling
- **Application**: Handles uncertain and incomplete data; useful for medical diagnosis where symptoms are probabilistic indicators

**c) Bayesian Approach [2 marks]**
- **Tasks**: Classification with prior knowledge, spam filtering, medical diagnosis
- **Application**: Incorporates prior domain knowledge into analysis; updates predictions as new evidence arrives

**d) Classification Approach [2 marks]**
- **Tasks**: Disease diagnosis, customer segmentation, fraud detection
- **Application**: Assigns data instances to predefined categories; useful when outcome categories are known

**e) Hybrid Approach [2 marks]**
- **Tasks**: Complex problems requiring multiple techniques
- **Application**: Combines multiple approaches (e.g., neural networks + rule extraction) to leverage strengths of each method

### c) Predictive vs Descriptive Data Mining [6 marks]

**Predictive Data Mining [3 marks]**
- **Operation**: Uses historical data to build models that predict future outcomes
- **Results**: Predictions, forecasts, probability scores, risk assessments
- **Techniques**: Classification (decision trees, SVM), Regression (linear, logistic), Time series forecasting, Neural networks

**Descriptive Data Mining [3 marks]**
- **Operation**: Summarizes and describes patterns in existing data without making predictions
- **Results**: Clusters, associations, summaries, data profiles
- **Techniques**: Clustering (K-means, hierarchical), Association rules (Apriori), Summarization, Sequence discovery

---

## QUESTION ONE (Compulsory) [30 Marks]

### a) Abbreviations [2 marks]

**i) ROLAP [1 mark]**: Relational Online Analytical Processing - OLAP implementation that stores data in relational databases and uses SQL for queries.

**ii) ETL [1 mark]**: Extract, Transform, Load - Process of extracting data from sources, transforming it for analysis, and loading it into a data warehouse.

### b) Types of Data Mining Activities [3 marks]

**i) Text Mining [1 mark]**
- Extracts patterns and knowledge from unstructured text documents
- **Yields**: Sentiment analysis, topic modeling, entity extraction, document classification
- Example: Analyzing patient clinical notes to identify adverse drug reactions

**ii) Multimedia Mining [1 mark]**
- Discovers patterns from multimedia data (images, audio, video)
- **Yields**: Object recognition, content-based retrieval, pattern detection in medical imaging
- Example: Detecting tumors in radiology images

**iii) Web Mining [1 mark]**
- Extracts knowledge from web content, structure, and usage logs
- **Yields**: User behavior patterns, website structure analysis, content recommendations
- Example: Analyzing health website usage to improve patient education content

### c) Regression vs Classification Learning [4 marks]

| Aspect | Regression | Classification |
|--------|------------|----------------|
| **Output** | Continuous numerical value | Discrete categorical label |
| **Goal** | Predict quantity | Predict category |
| **Loss Function** | Mean Squared Error, MAE | Cross-entropy, accuracy |
| **Examples** | Predicting blood pressure, length of hospital stay | Diagnosing disease presence, classifying tumor type |
| **Algorithms** | Linear regression, polynomial regression | Logistic regression, decision trees, SVM |

---

# MSc Public Health Data Science - SDS6108 Examination Answers

**Name:** Cavin Otieno Ouma
**Registration Number:** SDS6/46982/2024
**Course Unit:** SDS6108: Health Data Mining and Analytics
**Institution:** University of Nairobi, Department of Mathematics

---

## QUESTION 12

### i. Terminologies in Data Mining and Warehousing (5 Marks)

**a. Data Management**

Data Management refers to the comprehensive process of collecting, storing, organizing, maintaining, and utilizing data efficiently and securely within an organization. It encompasses policies, procedures, and technologies that ensure data quality, accessibility, integrity, and security throughout its lifecycle.

**b. Metadata**

Metadata is "data about data" – descriptive information that provides context about other data. It includes details such as data source, creation date, author, format, size, and structure. For example, in a patient database, metadata would describe field names, data types, and relationships between tables.

**c. Big Data**

Big Data refers to extremely large and complex datasets characterized by the 5 V's: **Volume** (massive scale), **Velocity** (high-speed generation), **Variety** (diverse formats), **Veracity** (data quality/accuracy), and **Value** (meaningful insights). Examples include genomic sequences, social media health data, and real-time patient monitoring streams.

**d. Knowledge Discovery**

Knowledge Discovery in Databases (KDD) is the overall process of extracting useful, valid, and actionable knowledge from large datasets. It encompasses the entire workflow from data selection, preprocessing, transformation, data mining, to interpretation and evaluation of patterns discovered.

**e. Data Mining**

Data Mining is a core step within KDD that involves applying algorithms and statistical techniques to discover hidden patterns, correlations, anomalies, and relationships within large datasets. Techniques include classification, clustering, association rule mining, and regression analysis.

---

### ii. Importance of Data Management to an Organization (2 Marks)

1. **Improved Decision Making:** Proper data management ensures accurate, timely, and reliable data is available for evidence-based strategic and operational decisions.

2. **Regulatory Compliance:** Helps organizations meet legal requirements (e.g., GDPR, HIPAA) by maintaining data security, privacy, and audit trails.

3. **Operational Efficiency:** Reduces data redundancy, eliminates silos, and streamlines workflows, leading to cost savings and productivity gains.

4. **Data Quality Assurance:** Ensures consistency, accuracy, and completeness of organizational data assets.

---

### iii. Key Steps in Data Analytics (3 Marks)

1. **Data Collection:** Gathering relevant data from multiple sources (databases, sensors, surveys).

2. **Data Cleaning and Preprocessing:** Handling missing values, removing duplicates, correcting errors, and standardizing formats.

3. **Data Exploration and Analysis:** Applying statistical methods, visualization, and analytical techniques to understand patterns.

4. **Modeling and Interpretation:** Building predictive/descriptive models and interpreting results.

5. **Communication and Action:** Presenting findings through reports/dashboards and implementing data-driven decisions.

---

## QUESTION 13

### i. Data Warehousing Technologies (4 Marks)

**a. ETL Process (Extract, Transform, Load)**

ETL is the process of extracting data from heterogeneous source systems, transforming it into a consistent format, and loading it into a data warehouse.

- **Example:** Extracting patient records from hospital EMR systems, transforming date formats and coding schemes to standards (ICD-10), and loading into a central health data warehouse.

**b. Metadata Management**

The administration of metadata to ensure data assets are properly documented, searchable, and understood across the organization.

- **Example:** Maintaining a data dictionary that defines each variable in a public health surveillance system, including field descriptions, valid values, and data lineage.

**c. Data Mart Management**

Data marts are subset repositories of data warehouses focused on specific business functions or departments.

- **Example:** A hospital creating separate data marts for Oncology, Cardiology, and Pharmacy departments, each containing only relevant subset data for departmental analytics.

**d. Data Mining Algorithm Efficiency**

Refers to optimizing computational performance and accuracy of data mining algorithms to handle large-scale data.

- **Example:** Using parallel processing to run Apriori algorithm for finding drug interaction patterns across millions of prescription records, reducing processing time from hours to minutes.

---

### ii. Six Types of Knowledge Discovery in Data Mining (6 Marks)

**1. Classification**

Assigns data items to predefined categories based on attributes.

*Example:* Classifying patients as high-risk or low-risk for diabetes based on clinical parameters.

**2. Clustering**

Groups similar data objects without predefined labels based on inherent similarities.

*Example:* Grouping patients with similar disease progression patterns for personalized treatment planning.

**3. Association Rule Mining**

Discovers interesting relationships and co-occurrence patterns among variables.

*Example:* Finding that patients prescribed Drug A and Drug B together frequently develop a specific side effect.

**4. Regression**

Predicts continuous numerical values based on relationships between variables.

*Example:* Predicting hospital length of stay based on patient demographics and diagnosis.

**5. Anomaly/Outlier Detection**

Identifies unusual patterns that deviate significantly from expected behavior.

*Example:* Detecting fraudulent insurance claims or unusual disease outbreaks in surveillance data.

**6. Sequential Pattern Mining**

Discovers patterns in ordered/time-series data.

*Example:* Identifying typical disease progression sequences (e.g., symptoms → diagnosis → treatment → outcomes).

---

## QUESTION 14

### Big Data Infrastructure and Analytics Technologies (10 Marks)

**a. Remote Procedure Call (RPC)**

RPC is a protocol that allows a program to execute procedures/functions on a remote server as if they were local calls, abstracting network communication complexity.

- **Mechanism:** Client sends request with procedure name and parameters → Server executes → Results returned to client.
- **Example:** A health analytics application on a local machine calling a remote server's function to perform complex genomic analysis, receiving results without managing network protocols directly.
- **Healthcare Application:** Telehealth systems using RPC to fetch patient records from centralized databases.

**b. Role of Middleware in Distributed Systems**

Middleware is software that acts as an intermediary layer connecting distributed applications, databases, and services, enabling communication and data management.

**Key Roles:**
- **Integration:** Connects heterogeneous systems (EMR, lab systems, imaging)
- **Communication:** Handles message passing and protocol translation
- **Security:** Manages authentication and access control
- **Transaction Management:** Ensures data consistency across systems

- **Example:** Health Information Exchange (HIE) middleware connecting multiple hospital systems to share patient records seamlessly while maintaining security protocols.

**c. Hadoop Framework**

Hadoop is an open-source distributed computing framework for storing and processing large datasets across clusters of computers.

**Core Components:**
- **HDFS (Hadoop Distributed File System):** Distributed storage across nodes
- **MapReduce:** Parallel processing programming model
- **YARN:** Resource management and job scheduling
- **Hadoop Common:** Shared utilities and libraries

- **Example:** A public health agency using Hadoop to process and analyze billions of health insurance claims to identify disease trends and healthcare utilization patterns.

**d. Web Services Technology**

Web services enable machine-to-machine communication over networks using standardized protocols (HTTP, XML, JSON).

**Types:**
- **SOAP (Simple Object Access Protocol):** XML-based, highly structured
- **REST (Representational State Transfer):** Lightweight, uses HTTP methods

- **Example:** A mobile health app using REST APIs to retrieve patient appointment data from hospital servers, or public health dashboards pulling COVID-19 statistics from WHO web services.

**e. Choreography and Orchestration in Web Services**

| Aspect | Orchestration | Choreography |
|--------|---------------|--------------|
| **Control** | Centralized controller manages workflow | Decentralized, peer-to-peer interaction |
| **Coordination** | One service directs others | Services know their role and collaborate |
| **Flexibility** | Easier to modify central logic | More scalable but complex to manage |

- **Orchestration Example:** A central hospital system coordinating patient admission workflow: verifying insurance → assigning bed → notifying pharmacy → scheduling tests.

- **Choreography Example:** Multiple healthcare providers independently responding to a disease outbreak notification, each executing predefined protocols without central coordination.

---

## QUESTION 15

### Healthcare Big Data Infrastructure Design

#### i. Big Data Infrastructure Solution Design

**Architecture Overview:**

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCES LAYER                        │
├─────────────┬─────────────────┬─────────────────────────────┤
│ Structured  │  Semi-Structured │      Unstructured           │
│ (EHR, Labs) │  (HL7, JSON)     │  (Genomic, Imaging)         │
└──────┬──────┴────────┬────────┴──────────────┬──────────────┘
       │               │                       │
       ▼               ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│              DATA INGESTION & ETL LAYER                      │
│  Apache Kafka (streaming) | Apache NiFi (batch) | Spark     │
└─────────────────────────────┬───────────────────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
┌─────────────┐      ┌───────────────┐      ┌──────────────┐
│   Data Lake │      │ Data Warehouse│      │  NoSQL Store │
│ (HDFS/S3)   │      │ (Structured)  │      │ (MongoDB)    │
│ Raw Storage │      │ Curated Data  │      │ Genomic Data │
└─────────────┘      └───────────────┘      └──────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              ANALYTICS & ML LAYER                            │
│    Apache Spark MLlib | TensorFlow | R Statistical Tools    │
└─────────────────────────────────────────────────────────────┘
```

**Addressing Key Requirements:**

**1. Data Heterogeneity:**
- **Data Lake Architecture:** Store raw data in native formats (HDFS/cloud object storage)
- **Schema-on-Read:** Apply structure during analysis, not ingestion
- **Unified Data Catalog:** Metadata management using Apache Atlas
- **Polyglot Persistence:** Different databases for different data types (RDBMS for EHR, MongoDB for genomic, DICOM servers for imaging)

**2. Storage Optimization:**
- **Tiered Storage:** Hot (SSD for active analytics), Warm (HDD for recent data), Cold (archive for historical)
- **Data Compression:** Parquet/ORC columnar formats for analytical workloads
- **Deduplication:** Identify and eliminate redundant records
- **Partitioning:** Organize data by date, patient cohort, or data type

**3. Regulatory Compliance (HIPAA, GDPR):**
- **Data Classification:** Automated tagging of PHI/PII data
- **Encryption:** At-rest (AES-256) and in-transit (TLS 1.3)
- **Audit Logging:** Comprehensive logging of all data access
- **Data Retention Policies:** Automated lifecycle management
- **Consent Management:** Track patient consent for data use

---

#### ii. Data Accessibility with Security and Privacy

**Accessibility Measures:**

1. **Role-Based Access Control (RBAC):**
   - Researchers access de-identified datasets
   - Clinicians access full patient records for care
   - Analysts access aggregated data for reporting

2. **Data Virtualization Layer:**
   - Unified query interface across heterogeneous sources
   - Users access data without knowing physical location

3. **Self-Service Analytics Platform:**
   - Secure sandbox environments for researchers
   - Pre-approved analytical tools and notebooks

4. **API Gateway:**
   - Standardized, secure APIs for programmatic access
   - Rate limiting and authentication

**Security and Privacy Standards:**

| Measure | Implementation |
|---------|----------------|
| **Authentication** | Multi-factor authentication, SSO integration |
| **Authorization** | Fine-grained permissions, attribute-based access |
| **De-identification** | Safe Harbor/Expert Determination methods |
| **Anonymization** | K-anonymity, differential privacy for analytics |
| **Monitoring** | Real-time threat detection, anomaly alerts |
| **Data Masking** | Dynamic masking for non-production environments |
| **Breach Response** | Incident response plan, notification procedures |

---

## QUESTION 16

### i. Cloud and Edge Computing Technologies Example (4 Marks)

**Example: Smart Hospital Patient Monitoring System**

**Edge Computing Implementation:**
- **IoT Sensors:** Wearable devices continuously monitor patient vitals (heart rate, SpO2, temperature)
- **Edge Gateways:** Bedside computing units process data locally in real-time
- **Local Analytics:** Immediate alert generation for critical threshold breaches (e.g., cardiac arrhythmia detection)
- **Latency:** <10ms response time for critical alerts

**Cloud Computing Implementation:**
- **Data Aggregation:** Processed summaries uploaded to cloud platform (AWS/Azure)
- **Long-term Storage:** Historical patient data stored in cloud data lakes
- **Advanced Analytics:** Machine learning models for predictive analytics (patient deterioration prediction)
- **Dashboard Services:** Hospital-wide monitoring dashboards accessible via web

**Hybrid Integration:**
- Edge handles time-critical processing; cloud handles resource-intensive analytics and storage
- Reduces bandwidth costs while enabling comprehensive analytics

---

### ii. Cloud and Edge Computing Technologies (6 Marks)

**a. Inter-cloud**

Communication and interaction between different cloud service providers to enable resource sharing, data portability, and service interoperability.

- **Purpose:** Avoid vendor lock-in, enable cloud bursting, disaster recovery
- **Example:** A health organization's primary data on AWS automatically failing over to Azure during outages

**b. Multi-cloud Service**

Strategy of using multiple cloud providers simultaneously to leverage best-of-breed services and reduce dependency on single vendors.

- **Benefits:** Redundancy, cost optimization, specialized services
- **Example:** Using Google Cloud for AI/ML analytics, AWS for storage, and Microsoft Azure for enterprise integration

**c. Peer-to-Peer Federation**

Decentralized cloud architecture where cloud providers interact as equals without central coordination, sharing resources directly.

- **Characteristics:** No central broker, autonomous negotiation
- **Example:** Regional health clouds in different countries directly sharing anonymized epidemic data during pandemic response

**d. Centralized Inter-cloud**

Architecture with a central broker/mediator that coordinates interactions between multiple cloud providers.

- **Characteristics:** Single point of coordination, unified management
- **Example:** A national health data platform acting as central broker, coordinating data exchange between various provincial cloud systems

**e. Multi-Cloud Libraries**

Software frameworks and APIs that provide abstraction layers for developing applications that work across multiple cloud platforms.

- **Examples:** Apache Libcloud, Apache jclouds, Terraform
- **Purpose:** Write once, deploy anywhere; unified interface for multi-cloud management
- **Healthcare Use:** Developing portable health analytics applications deployable on any cloud infrastructure

---

## QUESTION 17

### Financial Services Infrastructure for High-Frequency Trading

#### i. Comparison of Computing Paradigms

| Criteria | Cloud Computing | Edge/Fog Computing | Grid Computing |
|----------|-----------------|---------------------|----------------|
| **Architecture** | Centralized data centers | Distributed at network edge | Distributed heterogeneous resources |
| **Latency** | Higher (10-100ms) | Very Low (<10ms) | Variable (depends on resources) |
| **Scalability** | Excellent, on-demand | Limited by edge capacity | Good, federated resources |
| **Control** | Provider-managed | User-managed locally | Shared governance |
| **Cost Model** | Pay-per-use (OpEx) | Higher CapEx, lower OpEx | Resource sharing |
| **Data Sovereignty** | Data in provider's region | Data stays local | Distributed across sites |
| **Availability** | High (99.9%+ SLAs) | Depends on local infrastructure | Variable |
| **Use Case Fit** | Batch analytics, storage | Real-time processing | Scientific computation |

**For High-Frequency Trading (HFT):**
- **Cloud:** Suitable for back-office analytics, historical analysis, but latency too high for trading execution
- **Edge/Fog:** Ideal for trading execution requiring microsecond latency at exchange co-location
- **Grid:** Less suitable; designed for batch scientific workloads, not real-time trading

---

#### ii. Recommended Architecture: Hybrid Edge-Cloud Approach

**Recommended Solution: Edge Computing for Trading Execution + Cloud for Analytics**

```
┌─────────────────────────────────────────────────────────────┐
│                    EDGE LAYER (Trading)                      │
│  Co-located servers at stock exchanges                       │
│  • Ultra-low latency execution (<1ms)                       │
│  • Real-time market data processing                         │
│  • Algorithmic trading engines                              │
│  • Local redundancy for fault tolerance                     │
└────────────────────────┬────────────────────────────────────┘
                         │ Secured High-Speed Links
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    CLOUD LAYER (Analytics)                   │
│  • Historical data storage and analysis                     │
│  • Risk management and compliance reporting                 │
│  • Machine learning model training                          │
│  • Disaster recovery and backup                             │
│  • Global scalability for expansion                         │
└─────────────────────────────────────────────────────────────┘
```

**Justification:**

| Factor | Hybrid Edge-Cloud Justification |
|--------|--------------------------------|
| **Latency** | Edge co-location at exchanges delivers microsecond latency for order execution—critical for HFT competitive advantage. Cloud latency acceptable for non-time-critical functions. |
| **Fault Tolerance** | Edge: Local redundant systems with automatic failover. Cloud: Geographic redundancy, 99.99% availability for analytics and backup. |
| **Data Sovereignty** | Edge keeps trading data in local jurisdiction. Cloud regions selected to comply with financial regulations (GDPR, MiFID II, local securities laws). |
| **Cost** | Optimized TCO: High CapEx for critical edge infrastructure, OpEx model for scalable cloud analytics. Avoids over-provisioning. |
| **Global Scalability** | Cloud enables rapid expansion to new markets. Edge nodes deployed at new exchanges as needed. |

**Implementation Recommendations:**
1. Deploy FPGA-based edge servers at major exchange co-locations for nanosecond-level execution
2. Use dedicated fiber connections between edge and cloud layers
3. Implement multi-region cloud deployment for global disaster recovery
4. Establish real-time data replication from edge to cloud for analytics
5. Use cloud for ML model training, deploy optimized models to edge for inference

---

## Role of Health Data Mining and Analytics in Public Health

Health data mining and analytics plays a transformative role in modern public health by:

1. **Disease Surveillance & Outbreak Detection:** Real-time analysis of health data to identify disease patterns and emerging outbreaks early.

2. **Predictive Modeling:** Forecasting disease trends, hospital admissions, and resource needs for proactive planning.

3. **Population Health Management:** Identifying at-risk populations and targeting interventions effectively.

4. **Evidence-Based Policy:** Providing data-driven insights to inform public health policies and resource allocation.

5. **Healthcare Quality Improvement:** Analyzing clinical outcomes to identify best practices and reduce variations in care.

6. **Precision Public Health:** Tailoring interventions to specific communities based on their unique health profiles and social determinants.

---


