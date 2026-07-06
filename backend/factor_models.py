import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import statsmodels.api as sm

def extract_pca_factors(returns_df: pd.DataFrame, n_components: int = 3) -> dict:
    """
    Perform PCA on a dataframe of asset returns to extract statistical risk factors.
    Returns the factor returns and the factor loadings for each asset.
    """
    # Drop rows with NaN to ensure complete data for PCA
    data = returns_df.dropna(how='any')
    if len(data) < max(n_components, 30):
        return {"error": "Not enough complete data for PCA"}
        
    # Standardize returns
    means = data.mean()
    stds = data.std()
    data_scaled = (data - means) / stds
    
    pca = PCA(n_components=n_components)
    factor_returns = pca.fit_transform(data_scaled)
    loadings = pca.components_
    
    # factor_returns is shape (n_samples, n_components)
    # loadings is shape (n_components, n_features)
    
    factor_df = pd.DataFrame(factor_returns, index=data.index, columns=[f"Factor_{i+1}" for i in range(n_components)])
    loadings_df = pd.DataFrame(loadings.T, index=data.columns, columns=[f"Factor_{i+1}" for i in range(n_components)])
    
    return {
        "factor_returns": factor_df,
        "loadings": loadings_df,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist()
    }

def rolling_factor_regression(asset_returns: pd.Series, factor_returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Perform rolling OLS regression of asset returns against factor returns to isolate alpha and beta.
    """
    df = pd.concat([asset_returns, factor_returns], axis=1).dropna()
    if len(df) < window:
        return pd.DataFrame()
        
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    X_with_const = sm.add_constant(X)
    
    alphas = []
    betas = []
    dates = []
    
    for i in range(window, len(df)):
        y_window = y.iloc[i-window:i]
        X_window = X_with_const.iloc[i-window:i]
        
        try:
            model = sm.OLS(y_window, X_window).fit()
            alphas.append(model.params.iloc[0])
            betas.append(model.params.iloc[1:].values)
            dates.append(df.index[i])
        except:
            alphas.append(np.nan)
            betas.append(np.full(X.shape[1], np.nan))
            dates.append(df.index[i])
            
    betas_df = pd.DataFrame(betas, index=dates, columns=X.columns)
    betas_df['Alpha'] = alphas
    
    return betas_df
