"""
Configuration module for anomaly detection system.
Loads environment variables and defines default parameters.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Central configuration for the anomaly detection system."""
    
    # API Keys
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    
    # Model Configuration
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE: float = 0.2
    GROQ_MAX_TOKENS: int = 1500
    
    # Detection Weights (ensemble)
    WEIGHT_IFOREST: float = 0.4
    WEIGHT_XGBOOST: float = 0.3
    WEIGHT_RULES: float = 0.3
    
    # Threshold Configuration
    THRESHOLD_K: float = 2.5  # MAD multiplier for robust thresholding
    
    # IsolationForest Parameters
    IFOREST_CONTAMINATION: float = 0.1
    IFOREST_N_ESTIMATORS: int = 200
    IFOREST_MAX_SAMPLES: int = 256
    IFOREST_RANDOM_STATE: int = 42
    
    # XGBoost Parameters
    XGBOOST_N_ESTIMATORS: int = 100
    XGBOOST_MAX_DEPTH: int = 6
    XGBOOST_LEARNING_RATE: float = 0.1
    XGBOOST_RANDOM_STATE: int = 42
    
    # Feature Engineering
    MIN_VENDOR_FREQ: int = 3
    MIN_ACCOUNT_SAMPLES: int = 5
    BUSINESS_HOURS_START: int = 7
    BUSINESS_HOURS_END: int = 19
    
    # SHAP Configuration
    SHAP_TOP_K: int = 3
    SHAP_MAX_DISPLAY: int = 10
    
    # UI Configuration
    MAX_ROWS_DISPLAY: int = 100
    DEFAULT_CONFIDENCE: int = 4  # Mid-point on 1-7 scale
    
    # Persistence
    RESULTS_DIR: str = "results"
    USER_STUDY_FILE: str = "user_study_results.csv"
    
    # Feature Flags
    ENABLE_XGBOOST: bool = True
    ENABLE_SHAP: bool = True
    ENABLE_LLM: bool = True
    
    @classmethod
    def has_groq_key(cls) -> bool:
        """Check if Groq API key is configured."""
        return cls.GROQ_API_KEY is not None and len(cls.GROQ_API_KEY) > 0
    
    @classmethod
    def get_results_path(cls) -> str:
        """Get full path to results file."""
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        return os.path.join(cls.RESULTS_DIR, cls.USER_STUDY_FILE)


# Singleton instance
config = Config()

