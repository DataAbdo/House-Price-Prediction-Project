# house_price_prediction.py
# 🏠 House Price Prediction Project - Professional Version
# Complete ML pipeline for house price prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.inspection import permutation_importance
import scipy.stats as st

# Set random seed for reproducibility
np.random.seed(42)

class HousePricePredictor:
    """
    A comprehensive house price prediction system with multiple ML models
    and advanced feature engineering.
    """
    
    def __init__(self, data_path="data/house_prices.csv", target_col="price"):
        self.data_path = data_path
        self.target_col = target_col
        self.models = {}
        self.best_model = None
        self.preprocessor = None
        self.feature_names = None
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("📊 Loading dataset...")
        
        # For demo purposes - in real scenario, load from CSV
        # df = pd.read_csv(self.data_path)
        
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = {
            'area': np.random.normal(1500, 500, n_samples),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'location': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'year_built': np.random.randint(1950, 2020, n_samples),
            'price': np.random.normal(300000, 100000, n_samples)
        }
        
        df = pd.DataFrame(sample_data)
        df['price'] = df['price'] + df['area'] * 100 + df['bedrooms'] * 50000
        
        print(f"Dataset shape: {df.shape}")
        return df
    
    def prepare_features(self, df):
        """Prepare features and target variable"""
        print("🔧 Preparing features...")
        
        y = df[self.target_col]
        X = df.drop(columns=[self.target_col])
        
        # Identify feature types
        self.num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"Numerical features: {self.num_cols}")
        print(f"Categorical features: {self.cat_cols}")
        
        return X, y
    
    def build_preprocessor(self):
        """Build preprocessing pipeline"""
        print("⚙️ Building preprocessing pipeline...")
        
        numeric_pipeline = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_pipeline, self.num_cols),
            ("cat", categorical_pipeline, self.cat_cols)
        ])
        
        return self.preprocessor
    
    def build_models(self):
        """Initialize multiple ML models"""
        print("🤖 Building machine learning models...")
        
        # Ridge Regression
        ridge_model = Pipeline(steps=[
            ("pre", self.preprocessor),
            ("model", Ridge(alpha=1.0, random_state=42))
        ])

        # HistGradientBoosting
        hgb_model = Pipeline(steps=[
            ("pre", self.preprocessor),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.06, max_depth=6, max_leaf_nodes=31,
                min_samples_leaf=20, random_state=42))
        ])

        # Random Forest
        rf_model = Pipeline(steps=[
            ("pre", self.preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=400, min_samples_leaf=2,
                n_jobs=-1, random_state=42))
        ])

        self.models = {
            "Ridge": ridge_model,
            "HistGradientBoosting": hgb_model,
            "RandomForest": rf_model
        }
        
        return self.models
    
    def evaluate_model(self, name, model, X_test, y_test, log_transform=True):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        # Inverse log transformation if applied
        if log_transform:
            y_true_orig = np.expm1(y_test)
            y_pred_orig = np.expm1(y_pred)
        else:
            y_true_orig = y_test
            y_pred_orig = y_pred
        
        mae = mean_absolute_error(y_true_orig, y_pred_orig)
        
        # حساب RMSE يدوياً لتجنب مشكلة المعلمة squared
        mse = mean_squared_error(y_true_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        
        r2 = r2_score(y_true_orig, y_pred_orig)
        
        print(f"📈 {name}:")
        print(f"   MAE: ${mae:,.0f}")
        print(f"   RMSE: ${rmse:,.0f}")
        print(f"   R²: {r2:.3f}")
        
        return {"MAE": mae, "RMSE": rmse, "R2": r2}
    
    def tune_hyperparameters(self, model, X_train, y_train):
        """Perform hyperparameter tuning"""
        print("🎯 Tuning hyperparameters...")
        
        def mae_on_original(y_true, y_pred):
            return mean_absolute_error(np.expm1(y_true), np.expm1(y_pred))

        mae_scorer = make_scorer(mae_on_original, greater_is_better=False)

        param_distributions = {
            "model__learning_rate": st.loguniform(1e-3, 2e-1),
            "model__max_depth": st.randint(3, 9),
            "model__max_leaf_nodes": st.randint(15, 63),
            "model__min_samples_leaf": st.randint(10, 60),
            "model__l2_regularization": st.loguniform(1e-9, 1e-1)
        }

        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=10,  # قلللت لتسريع التشغيل
            scoring=mae_scorer,
            cv=3,  # قلللت لتسريع التشغيل
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        search.fit(X_train, y_train)
        print(f"✅ Best parameters: {search.best_params_}")
        
        return search.best_estimator_
    
    def plot_feature_importance(self, model, X_test, y_test):
        """Plot feature importance"""
        print("📊 Plotting feature importance...")
        
        try:
            # Get feature names after preprocessing
            ohe = model.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
            cat_feature_names = ohe.get_feature_names_out(self.cat_cols).tolist()
            all_feature_names = self.num_cols + cat_feature_names
            
            # Calculate permutation importance
            result = permutation_importance(
                model, X_test, y_test,
                n_repeats=5, random_state=42, scoring='neg_mean_absolute_error'
            )

            importances = result.importances_mean
            idx = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances[idx], y=np.array(all_feature_names)[idx], orient="h")
            plt.title("Top 10 Features - Permutation Importance")
            plt.xlabel("Importance (MAE decrease)")
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Could not plot feature importance: {e}")
    
    def save_model(self, model, path="models/house_price_model.joblib"):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        print(f"💾 Model saved to {path}")
    
    def run_pipeline(self):
        """Execute complete ML pipeline"""
        print("🚀 Starting House Price Prediction Pipeline...")
        
        # 1. Load data
        df = self.load_data()
        
        # 2. Prepare features
        X, y = self.prepare_features(df)
        
        # 3. Apply log transformation to target
        log_transform = True
        y_trans = np.log1p(y) if log_transform else y
        
        # 4. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_trans, test_size=0.2, random_state=42
        )
        
        # 5. Build preprocessor
        self.build_preprocessor()
        
        # 6. Build and train models
        self.build_models()
        
        print("\n" + "="*50)
        print("🤖 MODEL EVALUATION RESULTS")
        print("="*50)
        
        results = {}
        for name, model in self.models.items():
            print(f"\n--- Training {name} ---")
            model.fit(X_train, y_train)
            results[name] = self.evaluate_model(name, model, X_test, y_test, log_transform)
        
        # 7. Hyperparameter tuning on best model
        print("\n" + "="*50)
        print("🎯 HYPERPARAMETER TUNING")
        print("="*50)
        
        best_model_name = "HistGradientBoosting"
        print(f"\nTuning {best_model_name}...")
        self.best_model = self.tune_hyperparameters(
            self.models[best_model_name], X_train, y_train
        )
        
        # 8. Final evaluation
        print("\n" + "="*50)
        print("🏆 FINAL MODEL PERFORMANCE")
        print("="*50)
        
        final_results = self.evaluate_model(
            f"Tuned {best_model_name}", self.best_model, X_test, y_test, log_transform
        )
        
        # 9. Feature importance
        self.plot_feature_importance(self.best_model, X_test, y_test)
        
        # 10. Save model
        self.save_model(self.best_model)
        
        print("\n✅ Pipeline completed successfully!")
        return self.best_model, final_results

def main():
    """Main execution function"""
    try:
        predictor = HousePricePredictor()
        best_model, results = predictor.run_pipeline()
        
        print("\n🎉 Project ready for GitHub!")
        print("📁 Files created:")
        print("   - house_price_prediction.py (main script)")
        print("   - requirements.txt (dependencies)")
        print("   - README.md (documentation)")
        print("   - LICENSE (MIT license)")
        print("   - .gitignore (git rules)")
        
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        raise

if __name__ == "__main__":
    main()