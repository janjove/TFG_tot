import pyreadstat
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from func_sel import *
from funcions_net import *
from crear_dataset import *
from funcions_net import *
from preprocessing import *


# Import custom preprocessing functions


# Parameters
N_ITER = 200          # Number of iterations to average out randomness
ALPHA_THRESH =  0.01  # Threshold for selecting significant Lasso coefficients
TEST_SIZE = 0.2      # Proportion of data for test split

# Load raw data
print("Loading data...")
df_orig, meta = pyreadstat.read_sav("CEJFEAjut2015Updated.sav")
df_variables = pd.read_csv("variables.csv", sep=';')

# Preprocessing pipeline
dict_vars = create_dict(meta)
df_psico = dataset_psicologia(df_orig, dict_vars, df_variables)
df = drop_all_columns(df_psico, meta, df_variables, dict_vars)
# Cap the target at 1095 days
df['temps_fins_reincidencia1a'] = df_orig['temps_fins_reincidencia1a'].clip(upper=1095)

print("Mirem quants nans tenim a cada columna...")
_, df = neteja_na_columns(df, llindar=0.6)
df = omple_nans(df)
print("Columnes netejades")
# Encode categorical variables
df = label_encoding(df)

# Containers for results
i_features = []  # feature importances per iteration
lasso_coefs = [] # lasso coefficients per iteration
mse_list = []    # MSE on test set per iteration
r2_list = []     # R² on test set per iteration

# Main loop: train, select, record
for i in range(N_ITER):
    # Split data (no fixed random_state to let it vary)
    X = df.drop(columns='temps_fins_reincidencia1a')
    y = df['temps_fins_reincidencia1a']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


    # Lasso for feature selection
    alphas = np.logspace(0, 1, 50)  # De 1e-4 a 1 (50 valors)
    lasso = LassoCV(alphas=alphas, cv=5,max_iter=10000).fit(X_train_scaled, y_train)
    coefs = pd.Series(lasso.coef_, index=X_train.columns)
    lasso_coefs.append(coefs)

    # Select features above threshold
    selected_features = coefs[coefs.abs() > ALPHA_THRESH].index.tolist()
    X_train_sel = X_train_scaled[selected_features]
    X_test_sel = X_test_scaled[selected_features]

    if len(selected_features) == 0:
        print(f"Iteració {i}: Cap feature seleccionada. S'omet.")
        continue

    # Train Random Forest and record feature importances
    rf = RandomForestRegressor(n_estimators=250).fit(X_train_sel, y_train)
    importances = pd.Series(rf.feature_importances_, index=selected_features)
    # Align with all features (zeros for unselected)
    full_importances = importances.reindex(X_train.columns).fillna(0)
    i_features.append(full_importances)

    # Evaluate on test set
    y_pred = rf.predict(X_test_sel)
    mse_list.append(mean_squared_error(y_test, y_pred))
    r2_list.append(r2_score(y_test, y_pred))

# Aggregate results across iterations
mean_coefs = pd.concat(lasso_coefs, axis=1).mean(axis=1)
mean_importances = pd.concat(i_features, axis=1).mean(axis=1)

# Save outputs
mean_coefs.to_csv("lasso_mean_coefficients.csv", header=["mean_coef"] )
mean_importances.to_csv("rf_mean_importances.csv", header=["mean_importance"])
pd.DataFrame({
    'mse': mse_list,
    'r2': r2_list
}).to_csv("model_performance_metrics.csv", index=False)

# Print summary of results

print(f"Mean MSE: {pd.Series(mse_list).mean()}")
print(f"Mean R²: {pd.Series(r2_list).mean()}")

print("Pipeline completed.\n- Lasso mean coefficients: lasso_mean_coefficients.csv\n- RF mean importances: rf_mean_importances.csv\n- Performance metrics: model_performance_metrics.csv")
