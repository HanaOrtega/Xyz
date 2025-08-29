# --- Standardowe importy ---
import re
import json
import os
import time
import sys
import copy
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import timedelta, date
from checkpoint_manager import CheckpointManager
from pathlib import Path
import multiprocessing
SCRIPT_DIR = Path(__file__).resolve().parent
# --- Importy ML/DL ---
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.metrics import mean_squared_error as sklearn_mse, mean_absolute_error
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import optuna
from optuna.integration import KerasPruningCallback
from pydantic import ValidationError
import matplotlib
matplotlib.use('Agg') # Użycie backendu Agg dla rysowania w tle
from matplotlib import pyplot as plt
from openpyxl.drawing.image import Image
from copy import deepcopy
# --- Importy z Modułów Projektu ---
from config_models import load_and_validate_config
from utils import split_data
from config_imports import (
    configure_gpu, configure_warnings, set_seeds, OPENPYXL_AVAILABLE,
    psutil, PSUTIL_AVAILABLE, rmse_scorer
)

from data_loader import pobierz_dane_akcji, pobierz_dane_sp500
from data_preprocessing import prepare_data
# Lokalizacja: main.py
from model_builder import build_model, create_directional_mse_loss, create_sharpe_loss_function
from model_trainer import build_model_for_tuner, train_model
from model_saver_loader import save_model_bundle, load_model_bundle
from predictor import predict_ensemble
from ensemble_builder import (
    calculate_performance_weights,
    train_stacking_meta_model,
    predict_weighted_ensemble,
    predict_stacking_ensemble
)
from tensorflow.keras import mixed_precision
from jigglypuff import run_jigglypuff_nas_pipeline
from feature_engineering import generate_features
# --- NEW: Optional MLflow Integration ---
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
# --- NEW: Import new modules ---

# --- WERSJA SKRYPTU ---
SCRIPT_VERSION = "main.py v15.0 (Refactored)"

# --- Konfiguracja Logowania ---
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_datefmt = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=log_format, datefmt=log_datefmt, force=True)
logger = logging.getLogger(__name__)


# --- WCZESNA KONFIGURACJA ŚRODOWISKA ---
# <--- DODAJ TĘ LINIĘ --->
mixed_precision.set_global_policy('mixed_float16')
# --- WCZESNA KONFIGURACJA ŚRODOWISKA ---
try:
    configure_warnings(action='ignore', category=FutureWarning)
    PHYSICAL_GPUS, LOGICAL_GPUS = configure_gpu()
except Exception as e:
    logger.critical(f"Krytyczny błąd podczas wczesnej konfiguracji: {e}", exc_info=True)
    sys.exit(1)


# ========================================================
# ===          FUNKCJE POMOCNICZE I EWALUACYJNE          ===
# ========================================================
# NOWA, POPRAWIONA WERSJA FUNKCJI
# W pliku main.py

def _aggregate_walk_forward_results(all_fold_results: List[Dict]) -> pd.DataFrame:
    """Agreguje wyniki z wszystkich foldów, jest odporna na brakujące dane numeryczne."""
    logger.info("\n--- Agregowanie wyników z walidacji kroczącej ---")
    if not all_fold_results:
        logger.warning("Lista wyników do agregacji jest pusta.")
        return pd.DataFrame()

    df = pd.DataFrame(all_fold_results)

    # Wybierz tylko te kolumny, które są numeryczne i nadają się do agregacji
    numeric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in ['fold']]

    if not numeric_cols:
        logger.error("Brak metryk numerycznych do zagregowania w wynikach. Zwracam pustą ramkę danych.")
        return pd.DataFrame()

    if 'Model' not in df.columns:
        logger.error("Brak kolumny 'Model'. Nie można zagregować wyników.")
        return pd.DataFrame()

    agg_funcs = {col: ['mean', 'std'] for col in numeric_cols}

    summary_df = df.groupby('Model').agg(agg_funcs)
    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]

    logger.info("Zagregowane wyniki (średnia z wszystkich foldów):\n" + summary_df.to_string())
    return summary_df


def _create_probe_features(df: pd.DataFrame, target_col: str, lags: int = 5) -> pd.DataFrame:
    """Tworzy prosty i szybki zestaw cech na potrzeby Modelu Skanującego."""
    probe_df = pd.DataFrame(index=df.index)
    probe_df['returns'] = df[target_col].pct_change()

    # --- POCZĄTEK POPRAWKI ---
    # Dodajemy obliczanie 30-dniowej kroczącej zmienności
    probe_df['volatility'] = probe_df['returns'].rolling(window=30).std()
    # --- KONIEC POPRAWKI ---

    for i in range(1, lags + 1):
        probe_df[f'lag_{i}'] = probe_df['returns'].shift(i)

    probe_df.dropna(inplace=True)
    return probe_df

def filter_df_by_date(df: pd.DataFrame, start_date_str: Optional[str], end_date_str: Optional[str]) -> pd.DataFrame:
    """Filters a DataFrame to a specific date range."""
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logger.error(f"Could not convert DataFrame index to DatetimeIndex: {e}")
            return pd.DataFrame()  # Return empty if conversion fails

    start_date = pd.to_datetime(start_date_str) if start_date_str else df.index.min()
    end_date = pd.to_datetime(end_date_str) if end_date_str else df.index.max()

    logger.info(f"Filtrowanie danych do zakresu: {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")
    return df.loc[start_date:end_date].copy()


def find_best_training_interval(full_df: pd.DataFrame, params: dict) -> Optional[str]:
    logger.info("\n--- Rozpoczynanie procesu doboru najlepszego interwału treningowego ---")
    if not params.get('date_selection', {}).get('enabled'):
        return params.get('data', {}).get('data_start')

    target_col = params.get('target_column', 'close')
    returns = full_df[target_col].pct_change()
    volatility = returns.rolling(window=30).std()

    try:
        best_start_date = volatility.idxmax()
        logger.info(
            f"\n===> Najlepszy start treningu to: {best_start_date.strftime('%Y-%m-%d')} (na podstawie analizy zmienności) <===")
        return best_start_date.strftime('%Y-%m-%d')
    except Exception as e:
        logger.error(f"Błąd podczas sondy doboru dat: {e}", exc_info=True)
        return params.get('data', {}).get('data_start')

def _find_market_regimes(df: pd.DataFrame, target_col: str, params: dict) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Automatycznie wykrywa i filtruje reżimy rynkowe na podstawie wygładzonej
    zmienności przy użyciu dokładniejszego algorytmu Pelt.
    """
    logger.info("    Automatyczne wykrywanie i filtrowanie reżimów rynkowych (metoda: Pelt)...")
    try:
        import ruptures as rpt
    except ImportError:
        logger.error("Biblioteka 'ruptures' nie jest zainstalowana. Metoda automatyczna jest niedostępna.")
        return []

    date_config = params.get('date_selection', {})
    vol_window = date_config.get('auto_volatility_window', 30)
    penalty = date_config.get('auto_pelt_penalty', 10)

    log_returns = np.log(df[target_col] / df[target_col].shift(1))
    volatility = log_returns.rolling(window=vol_window).std()

    smoothed_volatility = volatility.rolling(window=90).mean().dropna()
    if smoothed_volatility.empty:
        logger.warning("Nie można obliczyć wygładzonej zmienności - za mało danych.")
        return []

    algo = rpt.Pelt(model="rbf").fit(smoothed_volatility.values)
    breakpoints = algo.predict(pen=penalty)

    breakpoints = [0] + breakpoints

    regime_intervals = []
    end_of_data_date = df.index.max()
    max_len_years = date_config.get('auto_max_regime_years')
    max_age_years = date_config.get('auto_max_age_years')

    for i in range(len(breakpoints) - 1):
        start_idx = breakpoints[i]
        end_idx = breakpoints[i + 1] - 1
        if end_idx > start_idx:
            start_date = smoothed_volatility.index[start_idx]
            end_date = smoothed_volatility.index[end_idx]

            is_valid = True
            if (end_date - start_date).days < 365: is_valid = False
            if max_len_years and ((end_date - start_date).days / 365.25 > max_len_years): is_valid = False
            if max_age_years and ((end_of_data_date - end_date).days / 365.25 > max_age_years): is_valid = False
            if is_valid:
                regime_intervals.append((start_date, end_date))
                logger.info(f"      Zaakceptowano reżim: {start_date.date()} -> {end_date.date()}")

    return regime_intervals


def _generate_manual_intervals(df_end_date: pd.Timestamp, candidate_years: List[int]) -> List[
    Tuple[pd.Timestamp, pd.Timestamp]]:
    """Generuje listę interwałów na podstawie ręcznie zdefiniowanej listy lat."""
    logger.info("    Generowanie interwałów na podstawie ręcznej listy lat...")
    intervals = []
    for year in candidate_years:
        start_date = df_end_date - pd.DateOffset(years=year)
        intervals.append((start_date, df_end_date))
        logger.info(f"      Dodano interwał: {year} lat ({start_date.date()} -> {df_end_date.date()})")
    return intervals


def log_resource_usage(stage: str, pid: int):
    """Loguje aktualne zużycie pamięci RAM i CPU przez proces."""
    if not PSUTIL_AVAILABLE:
        return
    process = psutil.Process(pid)
    mem_info = process.memory_info()
    memory_mb = mem_info.rss / (1024 * 1024)
    logger.info(
        f"--- [Monitoring Zasobów] Etap: {stage} | Pamięć: {memory_mb:.2f} MB ---")


def split_data(X_dict: Dict[str, Any], Y_data: np.ndarray, val_size: int, test_size: int) -> Tuple[
    Dict, np.ndarray, Dict, np.ndarray, Dict, np.ndarray]:
    """
    Poprawnie dzieli dane na zbiory train, val i test, obsługując strukturę słownikową X.
    """
    # --- FIX: Replaced complex and error-prone validation with a simpler, robust check ---
    if not isinstance(X_dict, dict) or not Y_data.size > 0:
        logger.error("Dane wejściowe (X_dict lub Y_data) są nieprawidłowe lub puste. Nie można dokonać podziału.")
        return {}, np.array([]), {}, np.array([]), {}, np.array([])

    # --- END FIX ---

    def _split_single_x(data_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        end_val_idx = -test_size if test_size > 0 else None
        start_val_idx = -(test_size + val_size) if (test_size + val_size) > 0 else 0

        train_part = data_array[:start_val_idx]
        val_part = data_array[start_val_idx:end_val_idx] if val_size > 0 else np.array([])
        test_part = data_array[-test_size:] if test_size > 0 else np.array([])
        return train_part, val_part, test_part

    X_train, X_val, X_test = {}, {}, {}
    for data_format, data_value in X_dict.items():
        if isinstance(data_value, dict):
            X_train[data_format], X_val[data_format], X_test[data_format] = {}, {}, {}
            for key, array_to_split in data_value.items():
                train_part, val_part, test_part = _split_single_x(array_to_split)
                X_train[data_format][key], X_val[data_format][key], X_test[data_format][
                    key] = train_part, val_part, test_part
        elif data_value is not None:
            X_train[data_format], X_val[data_format], X_test[data_format] = _split_single_x(data_value)

    end_val_idx_y = -test_size if test_size > 0 else None
    start_val_idx_y = -(test_size + val_size) if (test_size + val_size) > 0 else 0
    Y_train, Y_val, Y_test = Y_data[:start_val_idx_y], Y_data[
                                                       start_val_idx_y:end_val_idx_y] if val_size > 0 else np.array(
        []), Y_data[-test_size:] if test_size > 0 else np.array([])

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def _split_single_x(data_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dzieli pojedynczą tablicę danych, poprawnie obsługując przypadek test_size=0."""
        end_val_idx = -test_size if test_size > 0 else None
        start_val_idx = -(test_size + val_size) if (test_size + val_size) < len(data_array) else 0

        train_part = data_array[:start_val_idx]
        val_part = data_array[start_val_idx:end_val_idx] if val_size > 0 else np.array([])
        test_part = data_array[-test_size:] if test_size > 0 else np.array([])
        return train_part, val_part, test_part

    X_train, X_val, X_test = {}, {}, {}
    if not X_dict or not any(v is not None and (isinstance(v, dict) and any(sub_v is not None and sub_v.size > 0 for sub_v in v.values()) or (not isinstance(v, dict) and hasattr(v, 'size') and v.size > 0)) for v in X_dict.values()):
        logger.error("Słownik X_dict jest pusty lub zawiera puste dane. Nie można dokonać podziału.")
        return {}, np.array([]), {}, np.array([]), {}, np.array([])

    for data_format, data_value in X_dict.items():
        if isinstance(data_value, dict):
            X_train[data_format] = {}
            X_val[data_format] = {}
            X_test[data_format] = {}
            for key, array_to_split in data_value.items():
                train_part, val_part, test_part = _split_single_x(array_to_split)
                X_train[data_format][key] = train_part
                X_val[data_format][key] = val_part
                X_test[data_format][key] = test_part
        elif data_value is not None:
            X_train[data_format], X_val[data_format], X_test[data_format] = _split_single_x(data_value)
        else:
            X_train[data_format], X_val[data_format], X_test[data_format] = None, None, None

    if Y_data.size == 0:
        logger.error("Tablica Y_data jest pusta. Nie można dokonać podziału.")
        return X_train, np.array([]), X_val, np.array([]), X_test, np.array([])

    end_val_idx_y = -test_size if test_size > 0 else None
    start_val_idx_y = -(test_size + val_size) if (test_size + val_size) < len(Y_data) else 0

    Y_train = Y_data[:start_val_idx_y]
    Y_val = Y_data[start_val_idx_y:end_val_idx_y] if val_size > 0 else np.array([])
    Y_test = Y_data[-test_size:] if test_size > 0 else np.array([])

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Oblicza uproszczoną dokładność kierunkową."""
    if y_true is None or y_pred is None or len(y_true) < 2:
        return np.nan
    try:
        true_diff = np.diff(y_true.flatten())
        pred_diff = np.diff(y_pred.flatten())

        if len(true_diff) == 0:
            return np.nan

        correct_direction = np.sign(true_diff) == np.sign(pred_diff)
        return np.sum(correct_direction) / len(true_diff)
    except Exception:
        return np.nan


# W pliku main.py
def _reconstruct_prices_from_returns(last_known_prices: np.ndarray, log_returns: np.ndarray) -> np.ndarray:
    """Odtwarza szereg cen na podstawie zwrotów logarytmicznych w sposób zabezpieczony przed przepełnieniem."""
    # --- POCZĄTEK POPRAWKI ---
    # Ograniczamy przewidywane logarytmiczne zwroty do rozsądnego zakresu, aby zapobiec przepełnieniu
    # Wartość 5 oznacza wzrost ~148x (e^5), co jest bezpiecznym limitem dla pojedynczego kroku.
    clipped_log_returns = np.clip(log_returns, -5, 5)
    # --- KONIEC POPRAWKI ---

    reconstructed = np.zeros_like(clipped_log_returns)
    current_prices = last_known_prices.reshape(-1, 1)

    for i in range(clipped_log_returns.shape[1]):
        # Używamy ograniczonych wartości w obliczeniach
        current_prices = current_prices * np.exp(clipped_log_returns[:, i:i + 1])
        reconstructed[:, i:i + 1] = current_prices

    return reconstructed


# W pliku main.py

# W pliku main.py

def evaluate_model_performance(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, scaler_target: Any,
                               features_list: List[str], X_test: Any, scalers_all: Dict, params: Dict) -> Dict:
    """
    Oblicza metryki wydajności, z dodatkowym zabezpieczeniem przed brakiem danych.
    """
    all_metrics = {}
    pred_type = params.get('prediction_target_type', 'price')
    model_type = params.get('model_type', '').upper()

    try:
        y_pred_price, y_true_price = None, None

        if pred_type == 'log_returns':
            price_col_original = params.get('target_column', 'close')
            last_sequences_scaled = None
            feature_list_for_price = features_list

            # --- POCZĄTEK POPRAWKI ---
            if model_type == 'TFT':
                tft_data = X_test.get('tft')
                if tft_data and 'observed_past' in tft_data:
                    last_sequences_scaled = tft_data['observed_past'][:, -1, :]
                    # W przyszłości można dodać obsługę dedykowanej listy cech dla TFT
                    # feature_list_for_price = params.get('features', {}).get('cechy_w_modelu_tft_observed', features_list)
            else:
                standard_data = X_test.get('standard')
                if standard_data is not None:
                    last_sequences_scaled = standard_data[:, -1, :]

            if last_sequences_scaled is None:
                logger.error(
                    f"Nie można odtworzyć cen - brak danych wejściowych (standard lub tft) w X_test dla modelu {model_type}.")
                return {}  # Zwróć puste wyniki, aby uniknąć awarii
            # --- KONIEC POPRAWKI ---

            price_col_idx = feature_list_for_price.index(price_col_original)
            pre_prediction_prices = scalers_all[price_col_original].inverse_transform(
                last_sequences_scaled[:, price_col_idx].reshape(-1, 1)).flatten()

            y_pred_price = _reconstruct_prices_from_returns(pre_prediction_prices,
                                                            scaler_target.inverse_transform(y_pred_scaled))
            y_true_price = _reconstruct_prices_from_returns(pre_prediction_prices,
                                                            scaler_target.inverse_transform(y_true_scaled))
        else:
            y_pred_price = scaler_target.inverse_transform(y_pred_scaled)
            y_true_price = scaler_target.inverse_transform(y_true_scaled)

        # Obliczanie metryk (bez zmian)
        if y_pred_price is not None and y_true_price is not None:
            horizon = min(y_true_price.shape[1], y_pred_price.shape[1])
            for h in range(1, horizon + 1):
                h_idx = h - 1
                y_t_h, y_p_h = y_true_price[:, h_idx], y_pred_price[:, h_idx]
                all_metrics[f'rmse_d{h}'] = np.sqrt(sklearn_mse(y_t_h, y_p_h))
                all_metrics[f'mae_d{h}'] = mean_absolute_error(y_t_h, y_p_h)
                all_metrics[f'dir_acc_d{h}'] = calculate_directional_accuracy(y_t_h, y_p_h)

            logger.info(
                f"    Metryki D+1: RMSE={all_metrics.get('rmse_d1', 0):.4f}, DirAcc={all_metrics.get('dir_acc_d1', 0):.2%}")

        return {k: v for k, v in all_metrics.items() if not np.isnan(v)}

    except Exception as e:
        logger.error(f"    Błąd podczas ewaluacji modelu: {e}", exc_info=True)
        return {}
def save_future_predictions(trained_models: Dict, last_sequences: Dict, scalers: Dict, params: Dict, full_history_df: pd.DataFrame, custom_output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Generuje i zapisuje predykcje na przyszłość dla każdego modelu.
    """
    logger.info("\n--- Zapisywanie Predykcji na Przyszłość dla Modeli Bazowych ---")
    all_future_predictions = {}

    try:
        data_params = params.get('data', {})
        path_params = params.get('paths', {})
        horizon = data_params.get('horyzont_predykcji', 7)
        future_dates = pd.bdate_range(start=pd.to_datetime(data_params['data_end']) + pd.Timedelta(days=1),
                                      periods=horizon)
        target_type = params.get('prediction_target_type', 'price')
        target_col_name = 'log_returns' if target_type == 'log_returns' else params.get('target_column', 'close')
        if target_col_name not in scalers:
            logger.error(f"Brak skalera dla celu '{target_col_name}'. Przerywam zapisywanie predykcji.")
            return {}
        scaler_target = scalers[target_col_name]
    except KeyError as e:
        logger.error(f"Brak kluczowego parametru lub skalera: {e}. Przerywam zapisywanie predykcji.")
        return {}

    for model_name, (model_obj, model_params, _) in trained_models.items():
        try:
            is_tft = model_params.get('model_type', '').upper() == 'TFT'
            future_input = None

            if is_tft:
                last_sequence_tft = last_sequences.get('tft')
                if not last_sequence_tft or 'observed_past' not in last_sequence_tft or last_sequence_tft['observed_past'].size == 0:
                    logger.warning(f"Brak danych 'tft' w `last_sequences` dla modelu {model_name}. Pomijam.")
                    continue
                # Ensure batch dimension is preserved, even for a single sequence
                future_input = {key: val for key, val in last_sequence_tft.items()}
            else:
                last_sequence_std = last_sequences.get('standard')
                if last_sequence_std is None or last_sequence_std.size == 0:
                    logger.warning(f"Brak danych 'standard' w `last_sequences` dla modelu {model_name}. Pomijam.")
                    continue
                # Ensure batch dimension is preserved, even for a single sequence
                future_input = last_sequence_std

            if future_input is None:
                continue

            prediction_scaled_ensemble = predict_ensemble([(model_obj, model_params)], future_input)
            if prediction_scaled_ensemble is None:
                logger.error(f"Otrzymano pustą predykcję dla modelu '{model_name}'.")
                continue

            prediction_scaled = prediction_scaled_ensemble[0] # Take the first (and only) prediction from the batch
            prediction_unscaled = scaler_target.inverse_transform(prediction_scaled.reshape(1, -1)) # Reshape for inverse_transform
            # --- POCZĄTEK POPRAWKI ---
            # Użyj niestandardowego folderu, jeśli został podany, w przeciwnym razie użyj domyślnego
            if custom_output_dir:
                output_dir = custom_output_dir
            else:
                output_dir = path_params.get('plots_directory', '.')

            output_path = os.path.join(output_dir, f"predykcje_przyszlosc_{model_name}.csv")
            # --- KONIEC POPRAWKI ---

            if target_type == 'log_returns':
                price_col_original = params.get('target_column', 'close')
                feature_list = params['features']['cechy_w_modelu']

                if is_tft:
                    tft_observed_features_list = params.get('features', {}).get('cechy_w_modelu_tft_observed', feature_list)
                    if price_col_original not in tft_observed_features_list:
                        logger.error(f"Target column '{price_col_original}' not found in TFT observed features for future price reconstruction.")
                        continue
                    price_col_idx = tft_observed_features_list.index(price_col_original)
                    # Use the last observed value from the specific input for TFT
                    last_sequence_for_price = future_input['observed_past'][0] # Take first item from batch for single sequence
                else:
                    if price_col_original not in feature_list:
                        logger.error(f"Target column '{price_col_original}' not found in standard features for future price reconstruction.")
                        continue
                    price_col_idx = feature_list.index(price_col_original)
                    last_sequence_for_price = future_input[0] # Take first item from batch for single sequence

                if last_sequence_for_price is None or price_col_idx >= last_sequence_for_price.shape[1]:
                    logger.error(f"Invalid sequence data or index for price reconstruction for model '{model_name}'.")
                    continue

                last_known_price_scaled = last_sequence_for_price[-1, price_col_idx]
                last_known_price = scalers[price_col_original].inverse_transform([[last_known_price_scaled]])[0, 0]

                final_prediction = _reconstruct_prices_from_returns(np.array([last_known_price]),
                                                                    prediction_unscaled).flatten()
            else:
                final_prediction = prediction_unscaled.flatten()

            all_future_predictions[model_name] = final_prediction

            df_future = pd.DataFrame(
                {'Data': future_dates, f'Prognoza_{model_name}': final_prediction[:len(future_dates)]})
            output_path = os.path.join(path_params.get('plots_directory', '.'),
                                       f"predykcje_przyszlosc_{model_name}.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
            df_future.to_csv(output_path, index=False, sep=';', decimal=',')
            logger.info(f"    Zapisano przyszłe predykcje dla '{model_name}' do pliku: {output_path}")

        except Exception as e:
            logger.error(f"    Błąd podczas generowania przyszłej predykcji dla '{model_name}': {e}", exc_info=True)

    return all_future_predictions


def save_single_model_results(model_name: str, metrics: Dict, filepath: str, plots_dir: str):
    """Zapisuje wyniki ewaluacji dla pojedynczego modelu do pliku Excel, tworząc go lub dołączając arkusz."""
    if not OPENPYXL_AVAILABLE:
        logger.warning("Biblioteka 'openpyxl' nie jest dostępna. Pomijam zapis do Excela.")
        return

    logger.info(f"\n--- Zapisywanie wyników dla modelu '{model_name}' do pliku: {filepath} ---")

    import matplotlib
    matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
    from matplotlib import pyplot as plt
    from openpyxl.drawing.image import Image

    try:
        mode = 'a' if os.path.exists(filepath) else 'w'
        if_sheet_exists_param = 'replace' if mode == 'a' else None

        with pd.ExcelWriter(filepath, engine='openpyxl', mode=mode, if_sheet_exists=if_sheet_exists_param) as writer:
            metrics_df = pd.DataFrame({
                "Metryka": list(metrics.keys()),
                "Wartość": [str(v) if isinstance(v, (dict, list)) else v for v in metrics.values()]
            })
            metrics_df.to_excel(writer, sheet_name=model_name, index=False)

            worksheet = writer.sheets[model_name]

            history = metrics.get('history')
            if history and isinstance(history, dict) and all(k in history for k in ['loss', 'val_loss']):
                try:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(history['loss'], label='Strata Treningowa (loss)')
                    ax.plot(history['val_loss'], label='Strata Walidacyjna (val_loss)')
                    ax.set_title(f'Krzywe Uczenia dla Modelu {model_name}')
                    ax.set_xlabel('Epoka')
                    ax.set_ylabel('Strata')
                    ax.legend()
                    ax.grid(True)

                    chart_path = os.path.join(plots_dir, f"krzywa_uczenia_{model_name}.png")
                    fig.savefig(chart_path)
                    plt.close(fig)

                    img = Image(chart_path)
                    worksheet.add_image(img, 'E1')
                    logger.info(f"    Zapisano wykres krzywej uczenia dla '{model_name}'.")

                except Exception as e:
                    logger.error(f"    Nie udało się zapisać wykresu krzywej uczenia dla '{model_name}': {e}")
            else:
                logger.info(f"    Brak historii uczenia do zapisania dla '{model_name}'.")

        logger.info(f"Pomyślnie zapisano/zaktualizowano wyniki dla '{model_name}' w pliku: {filepath}")
    except Exception as e:
        logger.error(f"Błąd podczas zapisu wyników dla '{model_name}': {e}", exc_info=True)


def save_global_settings_to_file(
        filename: str,
        start_date: str,
        end_date: str

):
    """Zapisuje globalne ustawienia, takie jak najlepszy zakres dat, na początku pliku parametrów."""

    settings = {
        "training_data_start": start_date,
        "training_data_end": end_date,
    }
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 20} Global Run Settings {'=' * 20}\n")
            f.write(json.dumps(settings, indent=4))
            f.write("\n\n")
        logger.info(f"Saved global run settings (date range) to: {filename}")
    except Exception as e:
        logger.error(f"Failed to save global settings to '{filename}': {e}")


def save_best_params_to_file(model_type: str, model_params: Dict, filename: str = "top_parametr.txt"):
    """Zapisuje najlepsze znalezione parametry dla danego modelu do pliku tekstowego."""

    param_keys = {
        'shared': ['learning_rate', 'dropout_rate'],
        'LSTM': ['n_units', 'n_recurrent_layers', 'use_bidirectional', 'use_attention'],
        'GRU': ['n_units', 'n_recurrent_layers', 'use_bidirectional', 'use_attention'],
        'TRANSFORMER': ['num_transformer_blocks', 'd_model', 'num_heads', 'ff_dim'],
        'TFT': ['tft_d_model', 'tft_num_heads', 'tft_dropout_rate', 'tft_hidden_units'], # Added tft_hidden_units as it's a hyperparameter
        'RF': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'],
        'XGB': ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree']
    }
    keys_to_save = param_keys['shared'] + param_keys.get(model_type, [])
    best_params_to_save = {key: model_params.get(key) for key in keys_to_save if key in model_params}

    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"{'=' * 20} Model: {model_type} {'=' * 20}\n")
            f.write(json.dumps(best_params_to_save, indent=4))
            f.write("\n")
        logger.info(f"  Zapisano najlepsze parametry dla modelu '{model_type}' do pliku '{filename}'.")
    except Exception as e:
        logger.error(f"  Nie udało się zapisać parametrów dla modelu '{model_type}': {e}")


def load_best_params_from_file(filename: str = "top_parametr.txt") -> Dict:
    """
    Wczytuje najlepsze parametry z pliku, używając wyrażeń regularnych do poprawnego parsowania.
    """
    if not os.path.exists(filename):
        logger.info(f"Plik '{filename}' nie istnieje. Rozpoczynam standardowe strojenie modeli.")
        return {}

    logger.info(
        f"--- Znaleziono plik '{filename}'. Wczytywanie zapisanych parametrów, strojenie zostanie pominięte. ---")
    all_params = {}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()

        pattern = re.compile(r"Model:\s*(\w+)\s*.*?\n(\{.*?\})", re.DOTALL)
        matches = pattern.findall(content)

        if not matches:
            raise ValueError("Nie znaleziono poprawnych bloków parametrów w pliku.")

        for model_type, json_body in matches:
            params_dict = json.loads(json_body)
            all_params[model_type] = params_dict
            logger.info(f"  > Załadowano parametry dla: {model_type}")

    except (json.JSONDecodeError, ValueError, Exception) as e:
        logger.error(
            f"Błąd podczas parsowania pliku '{filename}': {e}. Plik zostanie zignorowany, a strojenie uruchomione.")
        return {}

    return all_params


# main.py

def generate_symbol_paths(params: dict, run_label: str) -> dict[str, str]:
    """
    Tworzy i zwraca słownik ze ścieżkami, ANCHOROWANYMI w lokalizacji skryptu main.py.
    """
    symbol = params['data']['symbol']
    # <--- ZMIANA: Używamy SCRIPT_DIR jako bazy --->
    base_output_dir = SCRIPT_DIR / params['paths']['base_output_directory']
    run_dir = base_output_dir / symbol.upper() / run_label

    # <--- ZMIANA: Poprawiona ścieżka dla danych historycznych S&P 500 --->
    sp500_historic_data_path = SCRIPT_DIR / params['paths']['base_output_directory'] / "^GSPC" / "historic_data_full.csv"

    paths = {
        'historic_data_csv': str(base_output_dir / symbol.upper() / "historic_data_full.csv"),
        'historic_data_sp500_csv': str(sp500_historic_data_path),
        'model_save_directory': str(run_dir / params['paths']['model_save_directory']),
        'plots_directory': str(run_dir / params['paths']['plots_directory']),
        'tuner_directory': str(run_dir / params['paths']['tuner_directory']),
        'params_file': str(run_dir / "top_parametr.txt"),
        'results_file': str(run_dir / f"wyniki_ewaluacji_{params['nazwa_modelu']}.xlsx")
    }

    # Zapewnienie, że wszystkie foldery istnieją
    for key, path in paths.items():
        if key not in ['results_file', 'params_file', 'historic_data_csv', 'historic_data_sp500_csv']:
            Path(path).mkdir(parents=True, exist_ok=True)

    logger.info(f"Ustawiono ścieżki wyjściowe dla uruchomienia '{run_label}' w folderze: {run_dir}")
    return paths

# W pliku main.py

class TrialDirectoryCreator(keras.callbacks.Callback):
    """
    Niestandardowy callback, który zapewnia, że katalog dla każdego testu
    KerasTunera istnieje przed jego rozpoczęciem.
    """
    def __init__(self, tuner):
        super().__init__()
        self.tuner = tuner

    def on_trial_begin(self, trial):
        trial_dir = self.tuner.get_trial_dir(trial.trial_id)
        os.makedirs(trial_dir, exist_ok=True)
        logger.info(f"Upewniono się, że katalog dla testu istnieje: {trial_dir}")
# ========================================================
# ===        KOMPONENTY POTOKU PRZETWARZANIA             ===
# ========================================================
# main.py

def run_model_selection_probe(raw_df: pd.DataFrame, params: dict, sp500_df: Optional[pd.DataFrame]) -> List[str]:
    """
    Przeprowadza szybką ocenę wszystkich kandydatów na model i zwraca listę
    najlepszych do pełnego treningu.
    """
    logger.info("\n" + "=" * 25 + " Rozpoczynam Sondę Selekcji Modeli " + "=" * 25)
    probe_config = params.get('model_probe_settings', {})
    probe_months = probe_config.get('probe_period_months', 12)
    top_n = probe_config.get('top_n_to_select', 3)

    # 1. Przygotuj dane do sondy (np. ostatni rok)
    probe_start_date = raw_df.index.max() - pd.DateOffset(months=probe_months)
    probe_df = raw_df.loc[probe_start_date:]

    test_size = len(probe_df) // 5  # Użyj 20% danych do testu
    train_df = probe_df.iloc[:-test_size]
    test_df = probe_df.iloc[-test_size:]

    # 2. Przygotuj dane w formacie sekwencji
    X_train_dict, Y_train, scalers, features, _, _ = prepare_data(train_df, params, sp500_df=sp500_df)
    X_test_dict, Y_test, _, _, _, _ = prepare_data(test_df, params, sp500_df=sp500_df, loaded_scalers=scalers,
                                                   preselected_features=features)

    if not X_train_dict or not X_test_dict:
        logger.error("Nie udało się przygotować danych dla sondy. Używam wszystkich modeli kandydujących.")
        return params.get('candidate_models', [])

    # 3. Przetestuj wszystkich kandydatów
    probe_results = []
    candidate_models = params.get('candidate_models', [])
    for model_type in candidate_models:
        probe_params = params.copy()
        probe_params['is_probe_run'] = True  # Aktywuj tryb sondy

        model, model_params, _ = train_and_tune_model(model_type, probe_params, X_train_dict, Y_train)

        if model:
            pred = predict_ensemble([(model, model_params)], X_test_dict)[0]
            metrics = evaluate_model_performance(Y_test, pred, scalers.get(params.get('target_column')), features,
                                                 X_test_dict, scalers, model_params)
            score = (metrics.get('dir_acc_d1', 0) * 0.6) + ((1 - metrics.get('rmse_d1', 1)) * 0.4)
            probe_results.append((model_type, score))
            logger.info(f"    -> Wynik sondy dla {model_type}: {score:.4f}")

    # 4. Wybierz najlepszych
    if not probe_results:
        logger.error("Sonda nie wyłoniła żadnych działających modeli. Używam wszystkich kandydatów.")
        return candidate_models

    sorted_models = sorted(probe_results, key=lambda x: x[1], reverse=True)
    top_performers = [model[0] for model in sorted_models[:top_n]]

    logger.info(
        f"✅ Sonda zakończona. Wybrano {len(top_performers)} najlepszych modeli do pełnego treningu: {top_performers}")
    return top_performers
# W pliku main.py

def train_and_tune_model(model_type: str, params: dict,
                         X_train_split: Any, Y_train_split: np.ndarray,
                         X_val_split: Any, Y_val_split: np.ndarray,
                         saved_params: Dict | None = None) -> tuple[Any, dict, Any]:
    """
    Orchestrates tuning and training. Correctly handles feature counts for all model types, including TFT.
    """
    model_type = model_type.strip().upper()
    logger.info(f"\n  {'~' * 20} Processing Model: {model_type} {'~' * 20}")

    model_params = deepcopy(params)
    model_params.update({'model_type': model_type, 'selected_model_type': model_type})
    training_params = params.get('training', {})

    if model_type == 'JIGGLYPUFF':
        logger.warning("Jigglypuff jest obsługiwany przez własny potok. Pomijam.")
        return None, None, None

    is_tft = model_type == 'TFT'
    current_X_train = X_train_split.get('tft') if is_tft else X_train_split.get('standard')
    current_X_val = X_val_split.get('tft') if is_tft else X_val_split.get('standard')

    trained_model, history = None, None
    use_saved_params = saved_params and model_type in saved_params

    # Logika dla modeli Keras
    if model_type in ['LSTM', 'GRU', 'TRANSFORMER', 'TFT', 'CNN-LSTM']:

        # --- START OF THE FIX ---
        # Poprawne określanie `n_features` dla wszystkich typów modeli
        if is_tft:
            if current_X_train and 'observed_past' in current_X_train and 'known_future' in current_X_train:
                # For TFT, get feature counts from the prepared data shapes
                n_features_observed = current_X_train['observed_past'].shape[2]
                n_features_known = current_X_train['known_future'].shape[2]

                # Update the params dictionary that will be passed to the builder
                model_params['features']['n_features_observed'] = n_features_observed
                model_params['features']['n_features_known'] = n_features_known
                logger.info(
                    f"    Set TFT features for builder: Observed={n_features_observed}, Known={n_features_known}")
            else:
                logger.error("    Cannot determine feature counts for TFT; training data is missing required keys.")
                return None, None, None
        else:
            # For standard models, get n_features from the standard data shape
            if current_X_train is not None:
                n_features = current_X_train.shape[2]
                model_params['features']['n_features'] = n_features
            else:
                logger.error(f"    Cannot determine n_features for {model_type}; training data is missing.")
                return None, None, None
        # --- END OF THE FIX ---

        if use_saved_params:
            logger.info(f"    Używam zapisanych parametrów dla modelu {model_type}.")
            model_params.update(saved_params[model_type])
            model_instance = build_model(model_params)
            if model_instance:
                trained_model, history = train_model(X_train_split, Y_train_split, model_params, X_val_split,
                                                     Y_val_split, model_instance)

        elif training_params.get('use_keras_tuner', False):
            logger.info(f"    Tuning Keras model {model_type} with KerasTuner...")
            try:
                tuner = kt.BayesianOptimization(
                    hypermodel=lambda hp: build_model_for_tuner(hp, model_params),
                    objective='val_loss',
                    max_trials=training_params.get('keras_tuner_max_trials', 15),
                    directory=params['paths']['tuner_directory'],
                    project_name=f'kt_bayes_{model_type}',
                    overwrite=training_params.get('force_retune', True)
                )

                validation_data_for_tuner = None
                if current_X_val is not None and Y_val_split is not None and current_X_val.shape[0] > 0 and \
                        Y_val_split.shape[0] > 0:
                    validation_data_for_tuner = (current_X_val, Y_val_split)
                else:
                    tuner.objective = 'loss'

                tuner_callbacks = [
                    keras.callbacks.EarlyStopping(patience=training_params.get('early_stopping_patience_tuner', 7)),
                    TrialDirectoryCreator(tuner)
                ]

                logger.info(f"    Starting KerasTuner search for {model_type}. This may take a long time...")
                tuner.search(current_X_train, Y_train_split,
                             validation_data=validation_data_for_tuner,
                             callbacks=tuner_callbacks)

                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                model_params.update(best_hps.values)

                logger.info("    Building and training the final model with the best hyperparameters...")
                final_model = build_model(model_params)

                if final_model:
                    # --- START OF THE FIX ---
                    # KROK KLUCZOWY: Kompilujemy finalny model przed treningiem
                    logger.info("    Compiling the final model with the best learning rate and loss function...")

                    loss_config = model_params.get('loss_function_settings', {})
                    loss_name = loss_config.get('name', 'mse')

                    if loss_name == 'directional_mse':
                        loss_function = create_directional_mse_loss(loss_config.get('directional_penalty_weight', 1.0))
                    elif loss_name == 'sharpe':
                        loss_function = create_sharpe_loss_function()
                    else:
                        loss_function = loss_name

                    final_model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=model_params.get('learning_rate', 0.001)),
                        loss=loss_function,
                        metrics=['mae', 'mse']
                    )
                    # --- END OF THE FIX ---

                    trained_model, history = train_model(X_train_split, Y_train_split, model_params, X_val_split,
                                                         Y_val_split, final_model)

            except Exception as e:
                logger.critical(f"CRITICAL ERROR during KerasTuner tuning for {model_type}: {e}", exc_info=True)
                return None, None, None

    # Logika dla modeli Scikit-learn
    else:
        current_X_train_np = X_train_full.get('standard') if isinstance(X_train_full, dict) else X_train_full
        Y_train_np = Y_train_full

        # Reshape to 2D
        X_train_flat = current_X_train_np.reshape(current_X_train_np.shape[0], -1)

        # Ensure Y_train_np is 1D if it's a single output target
        if Y_train_np.ndim > 1 and Y_train_np.shape[1] == 1: Y_train_np = Y_train_np.flatten()

        estimator, tuner_config, final_hyperparams = None, {}, {}

        if model_type == 'RF':
            estimator = RandomForestRegressor(random_state=params.get('random_state'))
            tuner_config = {'use_tuner': params.get('use_rf_tuner', False),
                            'param_grid': params.get('rf_param_grid', {}),
                            'iterations': params.get('rf_tuner_iterations', 10),
                            'cv_splits': params.get('rf_cv_splits', 5)}
            final_hyperparams = params.get('default_rf_params', {}).copy()
        elif model_type == 'XGB':
            estimator = XGBRegressor(random_state=params.get('random_state'))
            tuner_config = {'use_tuner': params.get('use_xgb_tuner', False),
                            'param_grid': params.get('xgb_param_grid', {}),
                            'iterations': params.get('xgb_tuner_iterations', 10),
                            'cv_splits': params.get('xgb_cv_splits', 5)}
            final_hyperparams = params.get('default_xgb_params', {}).copy()

        if estimator is None:
            logger.error(f"Nieznany typ modelu Scikit-learn: {model_type}. Pomijam.")
            return None, None, None

        if use_saved_params:
            logger.info(f"    Używam zapisanych parametrów dla {model_type}.")
            final_hyperparams.update(saved_params[model_type])
            final_estimator = estimator.set_params(**final_hyperparams)
            if Y_train_np.ndim > 1 and Y_train_np.shape[1] > 1:
                trained_model = MultiOutputRegressor(final_estimator).fit(X_train_flat, Y_train_np)
            else:
                trained_model = final_estimator.fit(X_train_flat, Y_train_np)
        elif tuner_config.get('use_tuner'):
            logger.info(f"    Strojenie {model_type} za pomocą RandomizedSearchCV...")
            param_grid = tuner_config['param_grid']

            base_estimator = MultiOutputRegressor(estimator) if Y_train_np.ndim > 1 and Y_train_np.shape[
                1] > 1 else estimator

            search = RandomizedSearchCV(estimator=base_estimator, param_distributions=param_grid,
                                        n_iter=tuner_config['iterations'],
                                        cv=TimeSeriesSplit(n_splits=tuner_config['cv_splits']), scoring=rmse_scorer,
                                        n_jobs=-1, verbose=1, random_state=params.get('random_state'))
            search.fit(X_train_flat, Y_train_np)
            best_params_raw = search.best_params_
            final_hyperparams.update(best_params_raw)
            trained_model = search.best_estimator_
        else:
            logger.info(f"    Używam domyślnych parametrów dla {model_type}.")
            final_estimator = estimator.set_params(**final_hyperparams)
            if Y_train_np.ndim > 1 and Y_train_np.shape[1] > 1:
                trained_model = MultiOutputRegressor(final_estimator).fit(X_train_flat, Y_train_np)
            else:
                trained_model = final_estimator.fit(X_train_flat, Y_train_np)

        model_params.update(final_hyperparams)

    if trained_model is None:
        logger.error(f"Trening modelu {model_type} nie powiódł się.")
        # --- NOWY BLOK: OBSŁUGA TRYBU SONDY ---
        if params.get('is_probe_run', False):
            logger.info(f"    Uruchomiono w trybie sondy dla {model_type}. Szybki trening z domyślnymi parametrami.")
            model_params.update(params.get(f'default_{model_type.lower()}_params', {}))
            model_instance = build_model(model_params)
            if model_instance:
                # POPRAWKA: Przekazujemy poprawnie podzielone zbiory danych, tak jak w głównej logice
                trained_model, history = train_model(
                    current_X_train,
                    Y_train_split,
                    model_params,
                    current_X_val,
                    Y_val_split,
                    model_instance,
                    epochs_override=params.get('model_probe_settings', {}).get('probe_epochs', 15)
                )
    return trained_model, model_params, history

# =============================================
# ===        GŁÓWNY ORKIESTRATOR PROCESU

# Nowa funkcja pomocnicza do tworzenia wykresów uczenia
def plot_learning_curves(history: Dict, model_name: str, save_path: str):
    """Generuje i zapisuje wykres krzywej uczenia."""
    if history and isinstance(history, dict) and all(k in history for k in ['loss', 'val_loss']):
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(history['loss'], label='Strata Treningowa (loss)')
            ax.plot(history['val_loss'], label='Strata Walidacyjna (val_loss)')
            ax.set_title(f'Krzywe Uczenia dla Modelu {model_name}')
            ax.set_xlabel('Epoka')
            ax.set_ylabel('Strata')
            ax.legend()
            ax.grid(True)
            fig.savefig(save_path)
            plt.close(fig)
            logger.info(f"    Zapisano wykres krzywej uczenia dla '{model_name}' do: {save_path}")
        except Exception as e:
            logger.error(f"    Nie udało się zapisać wykresu krzywej uczenia dla '{model_name}': {e}")
    else:
        logger.info(f"    Brak historii uczenia do zapisania dla '{model_name}'.")


# W pliku main.py
def run_pipeline(params: dict):
    """
    Główna, ostateczna i kompletna funkcja orkiestrująca, która obsługuje wszystkie tryby,
    w tym dwa tryby działania modułu Jigglypuff ('nas_architect' i 'analyst').
    """
    pipeline_start_time = time.time()
    iter_params = deepcopy(params)
    pid = os.getpid()

    # --- Inicjalizacja MLflow (jeśli włączone) ---
    tracking_config = iter_params.get('experiment_tracking', {})
    use_mlflow = MLFLOW_AVAILABLE and tracking_config.get('enabled', False)
    if use_mlflow:
        mlflow.set_experiment(tracking_config.get('experiment_name', 'Default_Experiment'))
        # Używamy etykiety przebiegu jako nazwy, aby odróżnić równoległe procesy
        run_label_for_mlflow = f"loss_{iter_params.get('loss_function_settings', {}).get('name')}"
        mlflow.start_run(run_name=f"run_{run_label_for_mlflow}_{pid}")
        mlflow.log_params(iter_params)

    try:
        # --- Krok 1: Konfiguracja ścieżek, wczytanie i wstępne przygotowanie danych ---
        run_label = f"loss_{iter_params.get('loss_function_settings', {}).get('name')}"
        symbol_paths = generate_symbol_paths(iter_params, run_label)
        iter_params['paths'].update(symbol_paths)

        checkpoint_manager = CheckpointManager(Path(iter_params['paths']['checkpoints_directory']))

        full_history_df = pobierz_dane_akcji(
            symbol=iter_params['data']['symbol'],
            config=iter_params,
            csv_save_path=Path(iter_params['paths']['historic_data_csv'])
        )
        if full_history_df is None:
            raise RuntimeError("Nie udało się pobrać głównych danych historycznych.")

        sp500_df = pobierz_dane_sp500(config=iter_params, csv_save_path=Path(
            iter_params['paths']['historic_data_sp500_csv'])) if iter_params.get('fetch_sp500') else None

        if iter_params.get('advanced_feature_engineering', {}).get('enabled', False):
            full_history_df = _calculate_advanced_features(full_history_df, iter_params)

        best_start_date = find_best_training_interval(full_history_df, iter_params)
        raw_df_for_pipeline = filter_df_by_date(full_history_df, best_start_date, iter_params['data']['data_end'])

        # --- Krok 2: Główna logika sterująca na podstawie trybu Jigglypuffa ---
        jiggly_config = iter_params.get('jigglypuff_settings', {})
        models_to_run = iter_params.get('models_to_tune', [])
        is_jigglypuff_in_run = 'JIGGLYPUFF' in [m.upper() for m in models_to_run]

        # --- ŚCIEŻKA A: JIGGLYPUFF W TRYBIE ARCHITEKTA (NAS) ---
        if is_jigglypuff_in_run and jiggly_config.get('mode') == 'nas_architect':
            logger.info("\n" + "#" * 20 + " WYBRANO TRYB: JIGGLYPUFF 2.0 (AUTONOMICZNY ARCHITEKT) " + "#" * 20)
            run_jigglypuff_nas_pipeline_walk_forward(raw_df_for_pipeline, iter_params, sp500_df)

        # --- ŚCIEŻKA B: STANDARDOWY TRENING (Z OPCJONALNYM JIGGLYPUFFEM 1.0) ---
        else:
            logger.info("\n" + "#" * 20 + " WYBRANO TRYB: STANDARDOWY TRENING MODELI " + "#" * 20)
            if is_jigglypuff_in_run:
                logger.info(
                    "--- Aktywowano Jigglypuffa w trybie analityka (1.0) - zostanie uruchomiony po treningu modeli bazowych ---")

            tscv = TimeSeriesSplit(n_splits=iter_params['evaluation_strategy']['num_splits'])
            all_fold_results = []
            last_fold_models = {}
            models_to_train = [m for m in models_to_run if m.upper() != 'JIGGLYPUFF']

            for fold, (train_index, test_index) in enumerate(tscv.split(raw_df_for_pipeline)):
                logger.info(f"\n{'=' * 30} Przetwarzanie Foldu {fold + 1}/{tscv.n_splits} {'=' * 30}")
                train_df = raw_df_for_pipeline.iloc[train_index]
                test_df = raw_df_for_pipeline.iloc[test_index]

                prepared_data = prepare_data(train_df, iter_params, sp500_df=sp500_df,
                                             checkpoint_manager=checkpoint_manager, fold_id=fold)
                if not prepared_data or not prepared_data[0]:
                    logger.error(f"Nie udało się przygotować danych dla foldu {fold + 1}. Pomijam.")
                    continue

                X_train_full, Y_train_full, scalers, features, _, _ = prepared_data
                val_size = int(len(Y_train_full) * iter_params['data']['validation_set_size'])
                X_train_split, Y_train_split, X_val_split, Y_val_split, _, _ = split_data(X_train_full, Y_train_full,
                                                                                          val_size, 0)

                saved_params = load_best_params_from_file(iter_params['paths']['params_file'])

                for model_type in models_to_train:
                    trained_model, model_params, history = train_and_tune_model(model_type, iter_params, X_train_split,
                                                                                Y_train_split, X_val_split, Y_val_split,
                                                                                saved_params)

                    if not trained_model:
                        logger.warning(f"Nie udało się wytrenować modelu {model_type} w foldzie {fold + 1}.")
                        continue

                    logger.info(f"--- (Fold {fold + 1}) Ewaluacja modelu: {model_type} ---")
                    X_test_dict, Y_test, _, _, _, _ = prepare_data(test_df, model_params, sp500_df=sp500_df,
                                                                   loaded_scalers=scalers,
                                                                   preselected_features=features)

                    if X_test_dict and Y_test is not None:
                        metrics = evaluate_model_performance(trained_model, X_test_dict, Y_test, scalers, features,
                                                             model_params)
                        result_entry = {'model': model_type, 'fold': fold + 1, **metrics}
                        all_fold_results.append(result_entry)

                        if use_mlflow:
                            mlflow_metrics = {f"fold_{fold + 1}_{model_type}_{k}": v for k, v in metrics.items() if
                                              isinstance(v, (int, float))}
                            mlflow.log_metrics(mlflow_metrics)

                        if fold == tscv.n_splits - 1:
                            last_fold_models[model_type] = (trained_model, model_params, scalers, features)
                    else:
                        logger.warning(f"Brak danych testowych dla modelu {model_type} w foldzie {fold + 1}.")

            # --- Blok dla Jigglypuffa 1.0 (Analityk) - uruchamiany PO wszystkich foldach ---
            if is_jigglypuff_in_run and jiggly_config.get('mode') == 'analyst':
                if not all_fold_results:
                    logger.warning(
                        "Tryb analityka dla Jigglypuffa wymaga wyników z innych modeli, a nie zebrano żadnych. Pomijam.")
                else:
                    logger.info("\n" + "=" * 25 + " URUCHAMIAM MODUŁ JIGGLYPUFF 1.0 (ANALITYK) " + "=" * 25)
                    jigglypuff_blueprint = analyze_and_design_blueprint(all_fold_results, raw_df_for_pipeline,
                                                                        iter_params)
                    teacher_model = build_teacher_model(last_fold_models, iter_params)
                    student_model = create_student_model(jigglypuff_blueprint, iter_params)

                    distiller = Distiller(student=student_model, teacher=teacher_model, params=iter_params)
                    distiller.compile(
                        optimizer=keras.optimizers.Adam(),
                        metrics=['mae'],
                        student_loss_fn=create_directional_mse_loss(),
                        distillation_loss_fn=keras.losses.KLDivergence()
                    )

                    full_data_prepared = prepare_data(raw_df_for_pipeline, iter_params, sp500_df=sp500_df)
                    if full_data_prepared and full_data_prepared[0]:
                        X_full_dict, Y_full, final_scalers, final_features, _, _ = full_data_prepared
                        logger.info("Rozpoczynanie destylacji...")
                        history = distiller.fit(X_full_dict['standard'], Y_full, epochs=50, batch_size=64,
                                                callbacks=[keras.callbacks.EarlyStopping(patience=10)])
                        trained_jigglypuff = distiller.student
                        # ... logika zapisu modelu i predykcji dla Jigglypuffa 1.0 ...

            # --- Agregacja wyników i generowanie predykcji na przyszłość dla modeli standardowych ---
            if all_fold_results:
                summary_df = _aggregate_walk_forward_results(all_fold_results)
                logger.info(f"\nPodsumowanie walidacji kroczącej:\n{summary_df.to_string()}")
                if use_mlflow and not summary_df.empty:
                    # Logowanie zagregowanych metryk do MLflow
                    summary_metrics = summary_df.mean().to_dict()
                    mlflow.log_metrics({f"summary_avg_{k}": v for k, v in summary_metrics.items()})

            if last_fold_models:
                logger.info("\n--- Generowanie prognoz na przyszłość używając modeli z ostatniego foldu ---")
                buffer_for_indicators = 60
                total_slice_len = iter_params['data']['sekwencja_dlugosc'] + buffer_for_indicators
                last_seq_data = full_history_df.iloc[-total_slice_len:]

                # Używamy skalerów i cech z pierwszego dostępnego modelu z ostatniej fałdy jako referencji
                _, _, last_scalers, last_features = next(iter(last_fold_models.values()))

                X_last_batch, _, _, _, _, _ = prepare_data(last_seq_data, iter_params, sp500_df=sp500_df,
                                                           loaded_scalers=last_scalers,
                                                           preselected_features=last_features)

                if X_last_batch:
                    X_last_final = {}
                    if X_last_batch.get('standard') is not None and X_last_batch['standard'].size > 0:
                        X_last_final['standard'] = X_last_batch['standard'][-1:]
                    if X_last_batch.get('tft') is not None and 'observed_past' in X_last_batch['tft']:
                        X_last_final['tft'] = {key: val[-1:] for key, val in X_last_batch['tft'].items()}

                    save_future_predictions(last_fold_models, X_last_final, last_scalers, final_features, iter_params,
                                            full_history_df)
                else:
                    logger.error("Błąd podczas przygotowywania ostatniej sekwencji danych do predykcji.")

    except Exception as e:
        logger.critical(f"[PROCES {pid}] BŁĄD KRYTYCZNY w przebiegu | Błąd: {e}", exc_info=True)
        if use_mlflow and mlflow.active_run():
            mlflow.end_run(status="FAILED")
        return False
    finally:
        logger.info(f"--- Całkowity czas wykonania potoku: {time.time() - pipeline_start_time:.2f}s ---")
        if use_mlflow and mlflow.active_run():
            mlflow.end_run()

    return True


def run_experiment_worker(run_params: Dict[str, Any]) -> bool:
    """
    Funkcja-wrapper dla pojedynczego przebiegu potoku, przeznaczona
    do użycia w puli wieloprocesowej.
    """
    process_id = os.getpid()
    loss_name = run_params.get('loss_function_settings', {}).get('name', 'N/A')
    probe_name = run_params.get('date_selection', {}).get('probe_optimization_metric', 'N/A')

    logger.info(f"[PROCES {process_id}] START przebiegu | Strata: {loss_name}, Sonda: {probe_name}")
    try:
        run_pipeline(run_params)
        logger.info(f"[PROCES {process_id}] ZAKOŃCZONO przebieg | Strata: {loss_name}, Sonda: {probe_name} | SUKCES")
        return True
    except Exception as e:
        logger.critical(
            f"[PROCES {process_id}] BŁĄD KRYTYCZNY w przebiegu | Strata: {loss_name}, Sonda: {probe_name} | Błąd: {e}",
            exc_info=True)
        return False

def run_cross_validation_pipeline(run_params: Dict[str, Any]) -> bool:
    """
    Funkcja-wrapper dla pojedynczego przebiegu potoku walidacji kroczącej.
    """
    process_id = os.getpid()
    loss_name = run_params.get('loss_function_settings', {}).get('name', 'N/A')
    probe_name = run_params.get('date_selection', {}).get('probe_optimization_metric', 'N/A')

    logger.info(f"[PROCES {process_id}] START przebiegu | Strata: {loss_name}, Sonda: {probe_name}")
    try:
        run_pipeline(run_params)
        logger.info(f"[PROCES {process_id}] ZAKOŃCZONO przebieg | Strata: {loss_name}, Sonda: {probe_name} | SUKCES")
        return True
    except Exception as e:
        logger.critical(
            f"[PROCES {process_id}] BŁĄD KRYTYCZNY w przebiegu | Strata: {loss_name}, Sonda: {probe_name} | Błąd: {e}",
            exc_info=True)
        return False

def run_jigglypuff_pipeline(params: dict):
    """
    Uruchamia potok "Meta-Algorytmu" oparty na sondzie i module Jigglypuff.
    """
    pipeline_start_time = time.time()
    iter_params = deepcopy(params)
    pid = os.getpid()

    # --- NEW: MLflow Integration ---
    tracking_config = params.get('experiment_tracking', {})
    use_mlflow = MLFLOW_AVAILABLE and tracking_config.get('enabled', False)
    if use_mlflow:
        mlflow.set_experiment(tracking_config.get('experiment_name', 'Jigglypuff_Experiment'))
        mlflow.start_run()
        mlflow.log_params(params)
        logger.info(f"✅ MLflow tracking is ACTIVE for experiment '{tracking_config.get('experiment_name')}'.")
    # --- END NEW ---

    try:
        run_label = f"jigglypuff_run-probe_{iter_params.get('date_selection', {}).get('probe_optimization_metric')}"
        symbol_paths = generate_symbol_paths(iter_params, run_label)
        iter_params['paths'].update(symbol_paths)

        log_resource_usage(f"Start potoku Jigglypuff", pid)
        checkpoint_base_dir = Path(iter_params['paths']['model_save_directory']).parent / "checkpoints"
        checkpoint_manager = CheckpointManager(checkpoint_base_dir)

        full_history_df = pobierz_dane_akcji(symbol=iter_params['data']['symbol'],
                                             data_end=str(iter_params['data'].get('data_end')),
                                             csv_save_path=Path(iter_params['paths']['historic_data_csv']))
        if full_history_df is None:
            raise ValueError("Nie udało się pobrać danych giełdowych.")

        sp500_df = pobierz_dane_sp500(data_end=str(iter_params['data'].get('data_end')), csv_save_path=Path(
            iter_params['paths']['historic_data_sp500_csv'])) if iter_params.get('fetch_sp500') else None

        # --- IMPROVEMENT: Call new feature engineering module ---
        feature_config = params.get('feature_config')
        if feature_config:
            logger.info("\n--- Uruchamianie modułu inżynierii cech ---")
            full_history_df = generate_features(full_history_df, feature_config)
        # --- END IMPROVEMENT ---

        if iter_params.get('date_selection', {}).get('enabled', False):
            best_start_date = find_best_training_interval(full_history_df, iter_params)
            if best_start_date:
                iter_params['data']['data_start'] = best_start_date

        raw_df_for_pipeline = filter_df_by_date(full_history_df, iter_params['data']['data_start'],
                                                iter_params['data']['data_end'])

        # Krok 1: Sonda selekcji modeli (jak w `run_cross_validation_pipeline`)
        models_to_run_names = run_model_selection_probe(raw_df_for_pipeline, iter_params, sp500_df)
        if not models_to_run_names:
            logger.error("Sonda nie wyłoniła modeli do dalszego treningu. Przerywam proces Jigglypuff.")
            return

        # Krok 2: Pełny trening modeli bazowych na całym zbiorze
        logger.info("\n--- Rozpoczynam pełny trening modeli bazowych na całym dostępnym zbiorze danych ---")
        trained_base_models = {}
        all_fold_results = []
        for model_type in models_to_run_names:
            logger.info(f"\n--- (Jigglypuff) Trenowanie modelu bazowego: {model_type} ---")

            # Przygotowanie danych raz dla pełnego zbioru
            prepared_data_full = prepare_data(raw_df_for_pipeline, iter_params, sp500_df=sp500_df)
            if not prepared_data_full: continue
            X_train_full, Y_train_full, scalers, features, _, _ = prepared_data_full

            trained_model, model_params, _ = train_and_tune_model(model_type, iter_params, X_train_full, Y_train_full)
            if trained_model:
                trained_base_models[model_type] = (trained_model, model_params, scalers, features)
                # Użyjemy dummy metrics do blueprintu, na podstawie których Jigglypuff wybierze architektury
                # W praktyce powinny być one z walidacji kroczącej, ale dla tego trybu upraszczamy
                dummy_metrics = {'Model': model_type, 'rmse_d1': np.random.rand(), 'dir_acc_d1': np.random.rand()}
                all_fold_results.append(dummy_metrics)

        if not trained_base_models:
            logger.error("Żaden model bazowy nie został wytrenowany. Nie można uruchomić Jigglypuff.")
            return

        # Krok 3: Analiza i budowa Jigglypuff
        logger.info("\n" + "=" * 25 + " URUCHAMIAM MODUŁ JIGGLYPUFF " + "=" * 25)
        jigglypuff_blueprint = analyze_and_design_blueprint(all_fold_results, raw_df_for_pipeline, iter_params)
        teacher_model = build_teacher_model(trained_base_models, iter_params)
        student_model = create_student_model(jigglypuff_blueprint, iter_params)

        # Krok 4: Destylacja wiedzy
        distiller = Distiller(
            student=student_model,
            teacher=teacher_model,
            params=iter_params,
            student_loss_fn=create_directional_mse_loss(
                iter_params.get('loss_function_settings', {}).get('directional_penalty_weight', 1.0)),
            distillation_loss_fn=keras.losses.KLDivergence(),
            alpha=0.2,
            temperature=3
        )

        distiller.compile(
            optimizer=keras.optimizers.Adam(learning_rate=iter_params.get('training', {}).get('learning_rate')),
            metrics=['mae', 'mse']
        )

        prepared_data_full = prepare_data(raw_df_for_pipeline, iter_params, sp500_df=sp500_df)
        if prepared_data_full:
            X_full_dict, Y_full, final_scalers, final_features, _, _ = prepared_data_full

            logger.info("Rozpoczynanie destylacji...")
            history = distiller.fit(
                X_full_dict['standard'],
                Y_full,
                epochs=iter_params.get('training', {}).get('epochs', 50),
                batch_size=iter_params.get('training', {}).get('batch_size'),
                callbacks=[keras.callbacks.EarlyStopping(
                    patience=iter_params.get('training', {}).get('early_stopping_patience', 10))]
            )

            # --- NEW: Log Jigglypuff metrics to MLflow ---
            if use_mlflow and history:
                mlflow.log_metrics(
                    {'jigglypuff_final_loss': history.history['loss'][-1],
                     'jigglypuff_final_val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history else None})
            # --- END NEW ---

            trained_jigglypuff = distiller.student
            if trained_jigglypuff:
                logger.info("Zapisywanie ostatecznego modelu Jigglypuff...")
                save_model_bundle(trained_jigglypuff, final_scalers,
                                  {'model_type': 'Jigglypuff', **jigglypuff_blueprint},
                                  "JIGGLYPUFF_ULTIMATE_MODEL", iter_params['paths']['model_save_directory'])

                # --- Generowanie predykcji na przyszłość dla Jigglypuff
                logger.info("\n--- Generowanie i zapisywanie predykcji na przyszłość dla Jigglypuff ---")
                buffer_for_indicators = 50
                total_slice_len = iter_params['data']['sekwencja_dlugosc'] + buffer_for_indicators
                last_seq_data = full_history_df.iloc[-total_slice_len:]
                X_last_batch, _, _, _, _, _ = prepare_data(
                    last_seq_data, iter_params, sp500_df=sp500_df, loaded_scalers=final_scalers,
                    preselected_features=final_features
                )

                if X_last_batch:
                    X_last_final = {}
                    if X_last_batch.get('standard') is not None and X_last_batch['standard'].size > 0:
                        X_last_final['standard'] = X_last_batch['standard'][-1:]
                    if X_last_batch.get('tft') is not None and 'observed_past' in X_last_batch['tft']:
                        X_last_final['tft'] = {key: val[-1:] for key, val in X_last_batch['tft'].items()}

                    jiggly_model_bundle = {
                        "Jigglypuff": (trained_jigglypuff, {'model_type': 'Jigglypuff'}, final_scalers, final_features)
                    }

                    save_future_predictions(
                        trained_models=jiggly_model_bundle,
                        last_sequences=X_last_final,
                        scalers=final_scalers,
                        params=iter_params,
                        full_history_df=full_history_df
                    )
                else:
                    logger.warning("Brak danych `X_last_final` do wygenerowania predykcji dla Jigglypuff.")
    except Exception as e:
        logger.critical(f"[PROCES {pid}] BŁĄD KRYTYCZNY w przebiegu Jigglypuff | Błąd: {e}", exc_info=True)
        # --- NEW: End MLflow run on crash ---
        if use_mlflow and mlflow.active_run():
            mlflow.end_run("FAILED")
        # --- END NEW ---
        raise
    finally:
        log_resource_usage(f"Koniec potoku Jigglypuff", pid)
        logger.info(f"--- Całkowity czas wykonania potoku Jigglypuff: {(time.time() - pipeline_start_time):.2f}s ---")
        # --- NEW: End MLflow run on success, if not already ended ---
        if use_mlflow and mlflow.active_run():
            mlflow.end_run()
        # --- END NEW ---


# =============================================
# ===          GŁÓWNY BLOK WYKONANIA          ===
# =============================================
# Zastąp istniejący blok `if __name__ == "__main__":` w pliku main.py poniższym kodem

# Zastąp cały blok `if __name__ == "__main__":` w pliku main.py


if __name__ == "__main__":
    multiprocessing.freeze_support()

    script_start_time, exit_code = time.time(), 0
    try:
        # --- KROK 1: WCZESNA KONFIGURACJA ---
        configure_warnings(action='ignore', category=FutureWarning)
        PHYSICAL_GPUS, LOGICAL_GPUS = configure_gpu()

        # --- IMPROVEMENT: Auto-detect config.yaml or fallback to config.txt ---
        config_path_yaml = SCRIPT_DIR / 'config.yaml'
        config_path_txt = SCRIPT_DIR / 'config.txt'

        if config_path_yaml.exists():
            config_path = config_path_yaml
            logger.info("Znaleziono i użyto pliku konfiguracyjnego: config.yaml")
        elif config_path_txt.exists():
            config_path = config_path_txt
            logger.info("Nie znaleziono config.yaml. Użyto domyślnego pliku: config.txt")
        else:
            raise FileNotFoundError("Nie znaleziono pliku konfiguracyjnego (config.yaml lub config.txt).")
        # --- END IMPROVEMENT ---

        # Wczytaj bazową konfigurację JEDEN RAZ w głównym procesie
        validated_config = load_and_validate_config(config_path)
        main_params_config = validated_config.model_dump(mode='python')
        set_seeds(main_params_config.get('random_state', 42))

        pipeline_mode = main_params_config.get('pipeline_mode', 'cross_validation')
        logger.info(f"Uruchamianie programu w trybie: '{pipeline_mode}'")

        if pipeline_mode == 'jigglypuff':
            # Uruchom nowy, szybki potok AutoML
            run_jigglypuff_pipeline(main_params_config)

        elif pipeline_mode == 'cross_validation':
            # --- KROK 2: SELEKCJA MODELI ZA POMOCĄ SONDY ---
            logger.info("Wczytywanie danych na potrzeby sondy selekcji modeli...")
            historic_data_path = SCRIPT_DIR / main_params_config['paths']['base_output_directory'] / \
                                 main_params_config['data']['symbol'].upper() / "historic_data_full.csv"
            sp500_historic_path = SCRIPT_DIR / main_params_config['paths'][
                'base_output_directory'] / "^GSPC" / "historic_data_full.csv"

            full_history_df = pobierz_dane_akcji(symbol=main_params_config['data']['symbol'],
                                                 config=main_params_config,
                                                 csv_save_path=historic_data_path)
            sp500_df = pobierz_dane_sp500(config=main_params_config,
                                          csv_save_path=sp500_historic_path)
            # Uruchom sondę, aby wybrać najlepsze modele
            top_models = run_model_selection_probe(full_history_df, main_params_config, sp500_df)

            if not top_models:
                logger.warning("Sonda nie wyłoniła żadnych modeli. Używam pełnej listy z pliku konfiguracyjnego.")
                top_models = main_params_config.get('models_to_tune', [])

            # Zaktualizuj główną konfigurację, aby używać tylko najlepszych modeli
            main_params_config['models_to_tune'] = top_models
            logger.info(f"OSTATECZNA LISTA MODELI DO PEŁNEGO TRENINGU: {top_models}")

            # --- KROK 3: PRZYGOTOWANIE RÓWNOLEGŁYCH EKSPERYMENTÓW ---
            all_run_params = []
            loss_metrics_to_run = ['mae', 'mse', 'sharpe']
            probe_metrics_to_run = ['rmse', 'dir_acc']

            for loss_name in loss_metrics_to_run:
                for probe_name in probe_metrics_to_run:
                    run_params = copy.deepcopy(main_params_config)
                    run_params['loss_function_settings']['name'] = loss_name
                    run_params['date_selection']['probe_optimization_metric'] = probe_name
                    all_run_params.append(run_params)

            logger.info(f"Wygenerowano {len(all_run_params)} konfiguracji eksperymentów do uruchomienia równoległego.")

            # --- KROK 4: URUCHOMIENIE WIELOPROCESOWE ---
            if PHYSICAL_GPUS > 0:
                logger.warning(
                    f"Wykryto {PHYSICAL_GPUS} kart(y) GPU. Uruchamiam eksperymenty sekwencyjnie (1 proces), aby uniknąć konfliktów CUDA.")
                num_processes = 1
            else:
                num_processes = max(1, multiprocessing.cpu_count() - 2)
                logger.info(f"Nie wykryto GPU. Używam {num_processes} procesów roboczych CPU.")

            logger.info(f"Inicjalizacja puli {num_processes} procesów roboczych...")

            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(run_pipeline, all_run_params)

            if not all(results):
                logger.error("Co najmniej jeden z równoległych eksperymentów zakończył się błędem. Sprawdź logi.")
                exit_code = 1
        else:
            logger.error(f"Nieznany tryb potoku: '{pipeline_mode}'. Dostępne tryby to 'cross_validation' i 'jigglypuff'.")
            exit_code = 1

    except (FileNotFoundError, ValueError, KeyError, ValidationError) as e:
        logger.critical(f"Krytyczny błąd konfiguracji lub danych: {e}", exc_info=True)
        exit_code = 2
    except Exception as e:
        logger.critical(f"!!!! KRYTYCZNY NIEPRZECHWYCONY BŁĄD W PROCESIE GŁÓWNYM!!!!", exc_info=True)
        exit_code = 1
    finally:
        total_time = time.time() - script_start_time
        logger.info(
            f"\n{'#' * 80}\n### KONIEC Skryptu. Całkowity czas: {total_time:.2f}s ({total_time / 60:.2f} min) ({(total_time / 60) / 60:.2f} h). Kod wyjścia: {exit_code} ###\n{'#' * 80}")
        sys.exit(exit_code)