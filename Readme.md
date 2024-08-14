# Wine Quality Analysis

This repository contains a comprehensive analysis of the Wine Quality dataset. The analysis explores the key factors influencing wine quality, including data exploration, statistical analysis, and predictive modeling. Additionally, a Dash-based interactive dashboard is provided to visualize key insights and allow users to explore the data dynamically.

## Table of Contents

1. [Introduction](#introduction)  
2. [Data Analysis](#data-analysis)  
   - Univariate and Bivariate Analysis  
   - Correlation and Statistical Insights  
3. [Predictive Modeling](#predictive-modeling)  
4. [Interactive Dashboard](#interactive-dashboard)  
5. [Conclusion and Next Steps](#conclusion-and-next-steps)  
6. [How to Run the Project](#how-to-run-the-project)  

## Introduction

The goal of this project is to identify the physicochemical properties of wines that most significantly impact their quality. The analysis is structured to guide both consumers and winemakers in making data-driven decisions.

## Data Analysis

- **Univariate and Bivariate Analysis**: Exploration of individual features and their relationships with wine quality.  
- **Correlation and Statistical Insights**: Identification of key predictors like alcohol content, volatile acidity, and sulphates.  

## Predictive Modeling

Multiple regression models were developed to predict wine quality. These models were evaluated for their predictive power and ability to generalize.

## Interactive Dashboard

An interactive dashboard built using Dash allows users to explore the dataset and key insights visually.

### Features of the Dashboard:

- **Feature Selection and Visualization**: Users can select different features to visualize their distributions and relationships.  
- **Dynamic Graphs**: Includes histograms, scatter plots, and more, updated based on user input.  
- **Insights Report**: A summary of the key findings and recommendations is provided directly in the dashboard.  

## Conclusion and Next Steps

- **Key Insights**: Alcohol, volatile acidity, and sulphates are identified as the most significant features influencing wine quality.  
- **Next Steps**: Further exploration of non-linear models, outlier treatment, and additional data collection are recommended.

## How to Run the Project

1. **Clone the repository**:  
   ```bash
   git clone https://github.com/yourusername/wine-quality-analysis.git
   ```

2. **Navigate to the project directory**:  
   ```bash
   cd wine-quality-analysis
   ```

3. **Set up the environment**:  
   Create the conda environment using the `environment.yml` file:  
   ```bash
   conda env create -f environment.yml
   ```  
   Activate the environment:  
   ```bash
   conda activate tcenv
   ```

4. **Run the Jupyter Notebook for the analysis**:  
   ```bash
   jupyter notebook notebook.ipynb
   ```

5. **Run the Dash Dashboard**:  
   ```bash
   python app.py
   ```  
   Open your web browser and go to `http://127.0.0.1:8050/` to interact with the dashboard.

This README provides a complete overview of the project, including how to interact with the analysis and dashboard.
