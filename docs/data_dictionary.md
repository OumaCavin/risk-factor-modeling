# Data Dictionary - Behavioral Risk Factor Analysis

## Overview

This document provides a comprehensive reference for all variables used in the risk factor modeling project. The data is sourced from the Behavioral Risk Factor Surveillance System (BRFSS).

## Variable List

### Demographic Variables

#### `_STATE`
- **Description**: State FIPS code
- **Type**: Categorical
- **Range**: 1-72
- **Notes**: Identifies the US state where the survey was conducted

#### `NUMADULT`
- **Description**: Number of adults in household
- **Type**: Numeric
- **Range**: 1-76
- **Notes**: Excludes respondents themselves from count

#### `SEX`
- **Description**: Respondent sex
- **Type**: Binary
- **Values**: 
  - 1 = Male
  - 2 = Female

#### `MARITAL`
- **Description**: Marital status
- **Type**: Categorical
- **Values**:
  - 1 = Married
  - 2 = Divorced
  - 3 = Widowed
  - 4 = Separated
  - 5 = Never married
  - 6 = Unmarried couple

#### `EDUCA`
- **Description**: Education level
- **Type**: Ordinal
- **Values**:
  - 1 = Never attended school or only kindergarten
  - 2 = Grades 1-8 (Elementary)
  - 3 = Grades 9-11 (Some high school)
  - 4 = Grade 12 or GED (High school graduate)
  - 5 = College 1-3 years (Some college or technical school)
  - 6 = College 4 years or more (College graduate)

#### `EMPLOY1`
- **Description**: Employment status
- **Type**: Categorical
- **Values**:
  - 1 = Employed for wages
  - 2 = Self-employed
  - 3 = Out of work for 1 year or more
  - 4 = Out of work for less than 1 year
  - 5 = Homemaker
  - 6 = Student
  - 7 = Retired
  - 8 = Unable to work

#### `RENTHOM1`
- **Description**: Home ownership status
- **Type**: Categorical
- **Values**:
  - 1 = Own
  - 2 = Rent
  - 3 = Other arrangement

#### `VETERAN3`
- **Description**: Military veteran status
- **Type**: Binary
- **Values**:
  - 1 = Yes
  - 2 = No

### Health Status Variables

#### `GENHLTH`
- **Description**: General health status
- **Type**: Ordinal (5-point scale)
- **Values**:
  - 1 = Excellent
  - 2 = Very good
  - 3 = Good
  - 4 = Fair
  - 5 = Poor
- **Notes**: Primary target variable component

#### `PHYSHLTH`
- **Description**: Days of poor physical health in past 30 days
- **Type**: Numeric
- **Range**: 0-30, 88=None
- **Special Values**:
  - 88 = None (no days of poor physical health)
  - 77 = Refused
  - 99 = Don't know

#### `MENTHLTH`
- **Description**: Days of poor mental health in past 30 days
- **Type**: Numeric
- **Range**: 0-30, 88=None
- **Special Values**:
  - 88 = None (no days of poor mental health)
  - 77 = Refused
  - 99 = Don't know

#### `POORHLTH`
- **Description**: Days when poor physical or mental health kept from usual activities
- **Type**: Numeric
- **Range**: 0-30, 88=None
- **Special Values**:
  - 88 = None
  - 77 = Refused
  - 99 = Don't know

### Health Condition Variables

#### `BPHIGH4`
- **Description**: Ever told blood pressure is high
- **Type**: Ordinal
- **Values**:
  - 1 = Yes
  - 2 = No
  - 3 = Told borderline high or pre-hypertensive
  - 7 = Don't know/Not sure
  - 9 = Refused

#### `BPMEDS`
- **Description**: Taking medicine for high blood pressure
- **Type**: Binary
- **Values**:
  - 1 = Yes
  - 2 = No
- **Notes**: Only asked if BPHIGH4 = 1

#### `DIFFWALK`
- **Description**: Difficulty walking or climbing stairs
- **Type**: Binary
- **Values**:
  - 1 = Yes
  - 2 = No

### Health Behavior Variables

#### `SMOKE100`
- **Description**: Smoked at least 100 cigarettes in entire life
- **Type**: Binary
- **Values**:
  - 1 = Yes
  - 2 = No

#### `SMOKDAY2`
- **Description**: Frequency of smoking now
- **Type**: Ordinal
- **Values**:
  - 1 = Every day
  - 2 = Some days
  - 3 = Not at all
- **Notes**: Only asked if SMOKE100 = 1

#### `EXERANY2`
- **Description**: Exercise or physical activity in past 30 days
- **Type**: Binary
- **Values**:
  - 1 = Yes
  - 2 = No

### Healthcare Access Variables

#### `HLTHPLN1`
- **Description**: Any kind of health care coverage
- **Type**: Binary
- **Values**:
  - 1 = Yes
  - 2 = No

#### `PERSDOC2`
- **Description**: Have a personal doctor or health care provider
- **Type**: Ordinal
- **Values**:
  - 1 = Yes, one
  - 2 = Yes, more than one
  - 3 = No

#### `MEDCOST`
- **Description**: Could not see doctor due to cost in past 12 months
- **Type**: Binary
- **Values**:
  - 1 = Yes
  - 2 = No

#### `CHECKUP1`
- **Description**: Time since last routine checkup
- **Type**: Ordinal
- **Values**:
  - 1 = Within past year
  - 2 = Within past 2 years
  - 3 = Within past 5 years
  - 4 = 5 or more years ago
  - 7 = Don't know
  - 8 = Never

### Body Metrics

#### `WEIGHT2`
- **Description**: Self-reported weight in pounds
- **Type**: Numeric
- **Special Values**:
  - 7777 = Refused
  - 9999 = Don't know

#### `HEIGHT3`
- **Description**: Self-reported height in inches
- **Type**: Numeric
- **Format**: feet*12 + inches (e.g., 510 = 5'10")
- **Special Values**:
  - 7777 = Refused
  - 9999 = Don't know

## Derived Variables

### `BMI`
- **Description**: Body Mass Index
- **Calculation**: (weight_lbs * 0.453592) / (height_inches * 0.0254)^2
- **Type**: Continuous

### `BMI_CAT`
- **Description**: BMI Category
- **Type**: Categorical
- **Values**:
  - 1 = Underweight (BMI < 18.5)
  - 2 = Normal (18.5 <= BMI < 25)
  - 3 = Overweight (25 <= BMI < 30)
  - 4 = Obese (BMI >= 30)

### `health_score`
- **Description**: Composite health score
- **Calculation**: Normalized combination of health indicators
- **Range**: 0-10 (higher = better health)

### `RISK_COUNT`
- **Description**: Number of cardiovascular risk factors
- **Type**: Numeric (0-5)
- **Factors included**:
  - Current smoker
  - No exercise
  - High blood pressure
  - Difficulty walking
  - Obesity

### `ACCESS_SCORE`
- **Description**: Healthcare access composite score
- **Type**: Numeric (0-4)
- **Components**:
  - Has health insurance
  - Has personal doctor
  - No cost barriers
  - Recent checkup

### `POOR_HEALTH` (Target Variable)
- **Description**: Binary indicator of poor health status
- **Type**: Binary
- **Values**:
  - 0 = Good health (GENHLTH 1-3)
  - 1 = Poor health (GENHLTH 4-5)

## BRFSS Special Value Codes

| Code | Meaning | Treatment |
|------|---------|-----------|
| 7 | Refused (small range) | Convert to NaN |
| 8 | Not asked/Missing | Convert to NaN |
| 9 | Don't know (small range) | Convert to NaN |
| 77 | Refused (large range) | Convert to NaN |
| 88 | None/Zero | Convert to 0 |
| 99 | Don't know (large range) | Convert to NaN |
| 7777 | Refused (continuous) | Convert to NaN |
| 9999 | Don't know (continuous) | Convert to NaN |

## References

- BRFSS Questionnaire: https://www.cdc.gov/brfss/questionnaires/index.htm
- BRFSS Codebook: https://www.cdc.gov/brfss/annual_data/annual_data.htm
- BMI Calculation: https://www.cdc.gov/healthyweight/assessing/bmi/index.html
