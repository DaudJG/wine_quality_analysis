import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import pingouin as pg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


def create_two_distributions(data1: pd.DataFrame, data2: pd.DataFrame) -> None:
    num_features = data1.shape[1]
    plt.figure(figsize=(15, num_features * 4))
    for i, column in enumerate(data1.columns):
        plt.subplot(num_features, 2, 2*i+1)
        sns.histplot(data1[column], kde=False, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {column.replace("_", " ").title()} - Original', weight='bold')
        plt.xlabel(column.replace('_', ' ').title())
        plt.ylabel('Count')

        plt.subplot(num_features, 2, 2*i+2)
        sns.histplot(data2[column], kde=False, color='lightgreen', edgecolor='black')
        plt.title(f'Distribution of {column.replace("_", " ").title()} - Processed', weight='bold')
        plt.xlabel(column.replace('_', ' ').title())
        plt.ylabel('Count')
    
    sns.despine()
    plt.tight_layout()
    plt.show()
    

def create_distributions(data: pd.DataFrame) -> None:
    num_features = data.shape[1]
    plt.figure(figsize=(15, num_features * 4))
    for i, column in enumerate(data.columns):
        plt.subplot(num_features, 2, 2*i+1)
        sns.histplot(data[column], kde=False, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {column.replace("_", " ").title()}', weight="bold")
        plt.xlabel(column.replace('_', ' ').title())
        plt.ylabel('Count')

        plt.subplot(num_features, 2, 2*i+2)
        sns.boxplot(x=data[column], color='lightgrey')
        plt.title(f'{column.replace("_", " ").title()}', weight="bold")
        plt.xlabel(column.replace('_', ' ').title())
        plt.ylabel('')
    
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_custom_correlation_heatmap(corr_matrix: pd.DataFrame, p_value_matrix: pd.DataFrame, significance_level: float = 0.05) -> None:
    mask_significant = p_value_matrix < significance_level
    mask_upper_triangle = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, cbar_kws={"shrink": .5}, mask=mask_upper_triangle)

    for y in range(corr_matrix.shape[0]):
        for x in range(corr_matrix.shape[1]):
            if not mask_upper_triangle[y, x] and mask_significant.iloc[y, x]:
                plt.text(x + 0.2, y + 0.8, '*', ha='center', va='center', color='green', fontsize=20)

    plt.title("Correlation Heatmap (Stars indicate significance)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    sns.despine()
    plt.show()
    

def calculate_pearson_correlation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = df.columns
    corr_matrix = pd.DataFrame(np.nan, index=cols, columns=cols)
    p_value_matrix = pd.DataFrame(np.nan, index=cols, columns=cols)

    for col1, col2 in itertools.combinations(cols, 2):
        corr, p_value = stats.pearsonr(df[col1], df[col2])
        corr_matrix.loc[col1, col2] = corr_matrix.loc[col2, col1] = corr
        p_value_matrix.loc[col1, col2] = p_value_matrix.loc[col2, col1] = p_value

    return corr_matrix, p_value_matrix


def create_two_distributions_with_qqplot(data1: pd.DataFrame, data2: pd.DataFrame) -> None:
    num_features = data1.shape[1]
    plt.figure(figsize=(24, num_features * 6))
    z = stats.norm.ppf(0.975)
    for i, column in enumerate(data1.columns):
        plt.subplot(num_features, 4, 4*i+1)
        sns.histplot(data1[column], kde=True, color='skyblue', edgecolor='black')
        plt.title(f'Histogram of {column.replace("_", " ").title()} - Original')
        plt.xlabel(column.replace('_', ' ').title())
        plt.ylabel('Count')

        plt.subplot(num_features, 4, 4*i+2)
        (osm, osr), (slope, intercept, r) = stats.probplot(data1[column], dist="norm", plot=plt)
        plt.plot(osm, intercept + slope * osm * z, ls='--', color='red')
        plt.plot(osm, intercept - slope * osm * z, ls='--', color='red')
        plt.title(f'QQ Plot of {column.replace("_", " ").title()} - Original')

        plt.subplot(num_features, 4, 4*i+3)
        sns.histplot(data2[column], kde=True, color='lightgreen', edgecolor='black')
        plt.title(f'Histogram of {column.replace("_", " ").title()} - Processed')
        plt.xlabel(column.replace('_', ' ').title())
        plt.ylabel('Count')

        plt.subplot(num_features, 4, 4*i+4)
        (osm, osr), (slope, intercept, r) = stats.probplot(data2[column], dist="norm", plot=plt)
        plt.plot(osm, intercept + slope * osm * z, ls='--', color='red')
        plt.plot(osm, intercept - slope * osm * z, ls='--', color='red')
        plt.title(f'QQ Plot of {column.replace("_", " ").title()} - Processed')

    plt.tight_layout(pad=3.0)
    sns.despine()
    plt.show()


def create_distributions_with_qqplot(data: pd.DataFrame) -> None:
    num_features = data.shape[1]
    plt.figure(figsize=(12, num_features * 6))
    for i, column in enumerate(data.columns):
        plt.subplot(num_features, 2, 2*i+1)
        sns.histplot(data[column], kde=True, color='skyblue', edgecolor='black')
        plt.title(f'Histogram with KDE of {column.replace("_", " ").title()}')
        plt.xlabel(column.replace('_', ' ').title())
        plt.ylabel('Count')

        plt.subplot(num_features, 2, 2*i+2)
        pg.qqplot(data[column], dist='norm', confidence=0.95, ax=plt.gca())
        plt.title(f'QQ Plot of {column.replace("_", " ").title()}')

    plt.tight_layout(pad=2.0)
    sns.despine()
    plt.show()


def plot_heatmap(data: pd.DataFrame, title: str, method: str) -> None:
    plt.figure(figsize=(10, 6))
    mask = np.triu(np.ones_like(data, dtype=bool))
    sns.heatmap(data, annot=True, cmap="coolwarm", mask=mask, fmt=".2f", linewidths=0.5)
    plt.title(title, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    sns.despine()


def create_pairgrid_with_feature(data: pd.DataFrame, feature: str, all_features: list, hue: str, palette: str) -> None:
    pairs = [other for other in all_features if other != feature and other != 'quality']
    n_pairs = len(pairs)
    n_cols = 2
    n_rows = n_pairs // n_cols + (n_pairs % n_cols > 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
    axes = axes.flatten()

    for i, feat_y in enumerate(pairs):
        ax = axes[i]
        sns.scatterplot(x=feature, y=feat_y, hue=hue, data=data, palette=palette, edgecolor='w', alpha=0.6, s=35, ax=ax)
        ax.set_xlabel(feature, fontweight="bold", fontsize=10)
        ax.set_ylabel(feat_y, fontweight="bold", fontsize=10)
        sns.despine(ax=ax)

    for i in range(n_pairs, n_rows * n_cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
    

def outliers_count(df: pd.DataFrame) -> pd.DataFrame:
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers_mask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
    extreme_outliers_mask = (df < (Q1 - 3 * IQR)) | (df > (Q3 + 3 * IQR))
    outliers_count = outliers_mask.sum()
    extreme_outliers_count = extreme_outliers_mask.sum()
    skewness = df.skew()
    
    results_df = pd.DataFrame({
        'Outliers': outliers_count,
        'Extreme Outliers': extreme_outliers_count,
        'Skewness': skewness
    })
    
    return results_df


def create_qq_plot(data: pd.DataFrame, column: str, ax: plt.Axes) -> None:
    stats.probplot(data[column], dist="norm", plot=ax)
    ax.set_title(f'QQ Plot for {column}')


def generate_partial_residual_plots(model: sm.OLS, X_train_scaled: pd.DataFrame) -> None:
    num_predictors = len(X_train_scaled.columns[1:])
    num_rows = (num_predictors + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(20, num_rows * 5))
    axes = axes.flatten()

    for i, predictor in enumerate(X_train_scaled.columns[1:]):
        ax = axes[i]
        sm.graphics.plot_ccpr(model, predictor, ax=ax)
        partial_residuals = model.resid + model.params[predictor] * X_train_scaled[predictor]
        sns.regplot(x=X_train_scaled[predictor], y=partial_residuals, lowess=True, ax=ax, scatter=False, color='red')
        ax.set_title(f'Partial Residual Plot for {predictor}')
        ax.set_xlabel(predictor.capitalize())
        ax.set_ylabel('Component + Residual')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()


def preprocess_data(df: pd.DataFrame, target_variable: str, drop_features: list = [], stratify: bool = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop([target_variable] + drop_features, axis=1)
    y = df[target_variable]

    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    X_train_const = sm.add_constant(X_train_scaled)
    X_test_const = sm.add_constant(X_test_scaled)
    
    return X_train_const, X_test_const, y_train, y_test


def fit_and_evaluate_model(X_train: pd.DataFrame, y_train: pd.Series, robust: bool = False, wls: bool = False) -> tuple[sm.OLS, pd.DataFrame]:
    if wls:
        ols_model = sm.OLS(y_train, X_train).fit()
        residuals_squared = ols_model.resid ** 2
        variance_model = sm.OLS(residuals_squared, X_train).fit()
        predicted_variance = variance_model.fittedvalues
        weights = 1 / predicted_variance
        weights = np.clip(weights, 1e-10, 1e10)
        model = sm.WLS(y_train, X_train, weights=weights).fit()
    else:
        X_train = sm.add_constant(X_train)
        model = sm.OLS(y_train, X_train).fit(cov_type='HC3' if robust else 'nonrobust')
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_train.columns
    vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    
    return model, vif_data


def plot_diagnostic_plots(model: sm.OLS) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))

    sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, ax=ax[0, 0], line_kws={'color': 'red', 'lw': 2})
    ax[0, 0].axhline(0, linestyle='--', color='gray', linewidth=2)
    ax[0, 0].set_xlabel('Fitted values')
    ax[0, 0].set_ylabel('Residuals')
    ax[0, 0].set_title('Residuals vs Fitted')

    sm.qqplot(model.resid, line='45', ax=ax[0, 1])
    ax[0, 1].set_title('Q-Q Plot')

    standardized_residuals = np.sqrt(np.abs(model.get_influence().resid_studentized_internal))
    sns.scatterplot(x=model.fittedvalues, y=standardized_residuals, ax=ax[1, 0])
    sns.regplot(x=model.fittedvalues, y=standardized_residuals, ax=ax[1, 0], lowess=True, scatter=False, line_kws={'color': 'red', 'lw': 2})
    ax[1, 0].axhline(0, linestyle='--', color='gray', linewidth=2)
    ax[1, 0].set_xlabel('Fitted values')
    ax[1, 0].set_ylabel('Sqrt(|Standardized Residuals|)')
    ax[1, 0].setTitle('Scale-Location')

    leverage = model.get_influence().hat_matrix_diag
    ax[1, 1].scatter(leverage, model.get_influence().resid_studentized_internal)
    ax[1, 1].axhline(0, linestyle='--', color='gray', linewidth=2)
    ax[1, 1].set_xlabel('Leverage')
    ax[1, 1].set_ylabel('Standardized Residuals')
    ax[1, 1].setTitle('Residuals vs Leverage')

    leverage_threshold = 2 * model.df_model / len(model.fittedvalues)
    for i in np.where((leverage > leverage_threshold) | (np.abs(model.get_influence().resid_studentized_internal) > 2))[0]:
        ax[1, 1].annotate(i, (leverage[i], model.get_influence().resid_studentized_internal[i]), fontsize=8, color='red')

    plt.tight_layout()
    plt.show()


def plot_cooks_distance(summary_frame: pd.DataFrame, percentile: int = 99) -> None:
    cooks_d = summary_frame['cooks_d']
    indices = cooks_d.index
    threshold = np.percentile(cooks_d, percentile)

    plt.figure(figsize=(14, 7))
    plt.stem(indices, cooks_d, markerfmt=",")
    plt.axhline(y=4 / len(cooks_d), color='r', linestyle='--', linewidth=2, label="Threshold")
    plt.xlabel('Observation Index')
    plt.ylabel("Cook's Distance")
    plt.title("Cook's Distance Plot")
    plt.legend()

    for idx, d in cooks_d.items():
        if d > threshold:
            plt.annotate(idx, (idx, d), fontsize=8, color='red')

    plt.show()


def extract_and_rank_model_metrics(model: sm.OLS, model_name: str, model_description: str, results_df: pd.DataFrame) -> pd.DataFrame:
    adj_r_squared = model.rsquared_adj
    aic = model.aic
    bic = model.bic
    
    new_row = pd.DataFrame({
        'Model_Name': [model_name],
        'Model_Description': [model_description],
        'Adj_R_squared': [adj_r_squared],
        'AIC': [aic],
        'BIC': [bic],
    })
    
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    results_df['Rank_Adj_R_squared'] = results_df['Adj_R_squared'].rank(ascending=False)
    results_df['Rank_AIC'] = results_df['AIC'].rank(ascending=True)
    results_df['Rank_BIC'] = results_df['BIC'].rank(ascending=True)
    
    results_df['Total_Rank'] = results_df[['Rank_Adj_R_squared', 'Rank_AIC', 'Rank_BIC']].sum(axis=1)
    
    results_df['Final_Rank'] = results_df['Total_Rank'].rank(ascending=True)
    
    return results_df
