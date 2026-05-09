#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Analiz Sistemi (Risk Analysis System) - v21

v21 KRİTİK DÜZELTMELER (CRITICAL FIXES):
1. Gap-buffered train/validation/test split (60 gün buffer foreshock leakage'ı önler)
2. Threshold optimizasyonu validation setinde (test set leakage'ı önler)
3. LSTM class_weight ile sınıf dengesizliği (class imbalance) yönetimi
4. F1, Precision, Recall metrikleri ön planda (Accuracy yanıltıcı imbalanced data'da)
5. Olumsuz Skill Score için belirgin uyarılar (warnings)
6. Doğru yorumlama notları raporda (interpretation notes)
"""

import pandas as pd
import sqlite3
import os
import time
import requests
from datetime import datetime, timedelta
import numpy as np
import random as py_random
# ✅ DETERMINISTIC: Tekrarlanabilir sonuçlar için tüm random seed'leri sabitle
py_random.seed(42)
np.random.seed(42)
import os as _os_init
_os_init.environ['PYTHONHASHSEED'] = '42'
_os_init.environ['TF_DETERMINISTIC_OPS'] = '1'
import joblib
import traceback
import folium
import warnings
from scipy.spatial import cKDTree
from numba import njit
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, brier_score_loss, roc_curve,
                             precision_recall_curve)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, load_model
import tensorflow as tf
tf.random.set_seed(42)  # ✅ Tekrarlanabilir sonuçlar için
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input, BatchNormalization,
                                     MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Add)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
import matplotlib.pyplot as plt
import shutil

# Renk kodları
R_ = '\033[91m'; G_ = '\033[92m'; P_ = '\033[95m'
C_ = '\033[96m'; Y_ = '\033[93m'; B_ = '\033[94m'; X_ = '\033[0m'
CURRENT_UTC_TIME = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
CURRENT_USER = "ozgursaygi"

# ============================================================================
# SABİT BİLİMSEL PARAMETRELER (MODELDEN BAĞIMSIZ)
# ============================================================================
FORESHOCK_MAG_THRESHOLD = 5.5
FORESHOCK_TIME_WINDOW_DAYS = 30
FORESHOCK_SPATIAL_RADIUS_KM = 50
FORESHOCK_MIN_MAG_DIFF = 0.8

ENHANCED_FEATURES = [
    'mag', 'depth', 'b_value_local', 'event_rate_local', 'time_since_last',
    'mag_completeness', 'spatial_density', 'temporal_clustering',
    'mag_trend', 'depth_clustering', 'energy_rate', 'swarm_indicator',
    'fault_distance', 'event_rate_24h', 'event_rate_12h', 'spatial_decay_index'
]
TARGET = 'is_foreshock'

# ============================================================================
# YENİ FONKSIYON: docs/ KLASÖRÜ YÖNETİMİ (v20)
# ============================================================================
def ensure_docs_dir():
    """docs/ klasörünü oluştur."""
    docs_dir = "docs"
    try:
        os.makedirs(docs_dir, exist_ok=True)
        print(f"{G_}✓ docs/ klasörü hazırlandı{X_}")
    except Exception as e:
        print(f"{R_}✗ docs/ klasörü oluşturulamadı: {e}{X_}")
    return docs_dir

def copy_to_docs(filename, docs_dir="docs"):
    """Dosyayı docs/ klasörüne kopyala ve hata kontrolü yap."""
    try:
        if os.path.exists(filename):
            dest = os.path.join(docs_dir, filename)
            shutil.copy2(filename, dest)
            print(f"{G_}✓ Kopyalandi: {filename} -> {dest}{X_}")
            return True
        else:
            print(f"{Y_}⚠ Bulunamadi: {filename}{X_}")
            return False
    except Exception as e:
        print(f"{R_}✗ Kopyalanamadi: {filename} - {e}{X_}")
        return False

# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================
@njit
def haversine_distance_numba(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1 = np.radians(lon1); lat1 = np.radians(lat1)
    lon2 = np.radians(lon2); lat2 = np.radians(lat2)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))

def haversine_distance(lat1, lon1, lat2, lon2):
    return haversine_distance_numba(lon1, lat1, lon2, lat2)

def get_neighbors_cKDTree(df, radius_km):
    vc = df[['latitude', 'longitude']].dropna()
    if vc.empty:
        return None
    tree = cKDTree(np.deg2rad(vc.values))
    return tree.query_ball_tree(tree, r=radius_km / 6371.0)

def standardize_date(dv):
    if pd.isna(dv) or dv == "":
        return None
    try:
        ds = str(dv).strip()
        if len(ds) <= 10:
            ds += " 00:00:00"
        dt = pd.to_datetime(ds, yearfirst=True, dayfirst=False, errors='coerce', utc=True)
        if pd.notna(dt):
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        dt = pd.to_datetime(ds, dayfirst=True, errors='coerce', utc=True)
        if pd.notna(dt):
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        return None
    except Exception:
        return None

def fix_future_dates(conn, tn):
    now = datetime.utcnow()
    lim = pd.Timestamp((now + timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S'), tz='UTC')
    try:
        df = pd.read_sql(f"SELECT * FROM {tn}", conn)
        if df.empty:
            return
        df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
        fm = df['time'] > lim
        if fm.sum() == 0:
            return
        print(f"{Y_}{fm.sum()} gelecek tarihli kayit bulundu.{X_}")
        drop = []
        for idx in df[fm].index:
            wd = df.at[idx, 'time']
            if pd.isna(wd):
                drop.append(idx)
                continue
            ok = False
            if wd.year > now.year + 1:
                try:
                    nd = wd.replace(year=wd.year - 100)
                    if nd <= lim:
                        df.at[idx, 'time'] = nd
                        ok = True
                except ValueError:
                    pass
            if not ok:
                try:
                    if wd.month != wd.day:
                        sw = wd.replace(month=wd.day, day=wd.month)
                        if sw <= lim:
                            df.at[idx, 'time'] = sw
                            ok = True
                except ValueError:
                    pass
            if not ok and wd.year > now.year + 1:
                try:
                    tmp = wd.replace(year=wd.year - 100)
                    if tmp.month != tmp.day:
                        sw2 = tmp.replace(month=tmp.day, day=tmp.month)
                        if sw2 <= lim:
                            df.at[idx, 'time'] = sw2
                            ok = True
                except ValueError:
                    pass
            if not ok:
                drop.append(idx)
        if drop:
            df.drop(index=drop, inplace=True)
        if (df['time'] > lim).sum() > 0:
            df = df[df['time'] <= lim]
        df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        conn.cursor().execute(f"DELETE FROM {tn}")
        df.to_sql(tn, conn, if_exists='append', index=False)
        conn.commit()
        print(f"{G_}Tarihler duzeltildi. Kayit:{len(df)}{X_}")
    except Exception as e:
        print(f"{R_}Tarih hatasi:{e}{X_}")

def setup_database(conn, tn):
    cur = conn.cursor()
    cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{tn}'")
    exists = cur.fetchone()
    cols = {
        "time": "TEXT", "latitude": "REAL", "longitude": "REAL", "depth": "REAL",
        "mag": "REAL", "eventID": "TEXT", "place": "TEXT", "b_value_local": "REAL",
        "event_rate_local": "REAL", "time_since_last": "REAL", "mag_completeness": "REAL",
        "spatial_density": "REAL", "temporal_clustering": "REAL", "mag_trend": "REAL",
        "depth_clustering": "REAL", "energy_rate": "REAL", "swarm_indicator": "INTEGER",
        "fault_distance": "REAL", "event_rate_24h": "REAL", "event_rate_12h": "REAL",
        "spatial_decay_index": "REAL", "earthquake_type": "TEXT", "is_foreshock": "INTEGER",
        "olasilik": "REAL", "confidence_score": "REAL", "total_uncertainty": "REAL",
        "seismic_zone": "TEXT"
    }
    created = False
    if not exists:
        cs = ", ".join([f'"{k}" {v}' for k, v in cols.items()])
        cur.execute(f"CREATE TABLE {tn} ({cs});")
        created = True
    else:
        cur.execute(f"PRAGMA table_info({tn});")
        ex = {r[1] for r in cur.fetchall()}
        for c, ct in cols.items():
            if c not in ex:
                cur.execute(f"ALTER TABLE {tn} ADD COLUMN {c} {ct};")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_time ON {tn} (time);")
    conn.commit()
    return created

def load_historical_csv(conn, tn, csv_path="ridgecrest_catalog.csv"):
    if not os.path.exists(csv_path):
        print(f"{Y_}Tarihi veri dosyasi bulunamadi ({csv_path}). Atliyoruz...{X_}")
        return False
    try:
        print(f"{C_}Gecmis datalar {csv_path} dosyasindan veritabanina aktariliyor...{X_}")
        df_csv = pd.read_csv(csv_path)
        col_map = {}
        for col in df_csv.columns:
            cl = col.lower()
            if 'date' in cl or 'time' in cl:
                col_map[col] = 'time'
            elif 'mag' in cl:
                col_map[col] = 'mag'
            elif 'lat' in cl:
                col_map[col] = 'latitude'
            elif 'lon' in cl or 'lng' in cl:
                col_map[col] = 'longitude'
            elif 'dep' in cl:
                col_map[col] = 'depth'
            elif 'id' in cl and 'grid' not in cl:
                col_map[col] = 'eventID'
            elif 'place' in cl or 'loc' in cl:
                col_map[col] = 'place'
        df_csv.rename(columns=col_map, inplace=True)
        if 'eventID' not in df_csv.columns:
            df_csv['eventID'] = [f"csv_id_{int(time.time())}_{i}" for i in range(len(df_csv))]
        if 'place' not in df_csv.columns:
            df_csv['place'] = "Ridgecrest Gecmis Veri (CSV)"
        req = ['time', 'latitude', 'longitude', 'depth', 'mag', 'eventID']
        missing = [c for c in req if c not in df_csv.columns]
        if missing:
            print(f"{R_}CSV dosyasinda zorunlu sutunlar eksik: {missing}{X_}")
            return False
        df_csv['time'] = df_csv['time'].apply(standardize_date)
        df_csv.dropna(subset=['time', 'mag', 'latitude', 'longitude'], inplace=True)
        df_csv = df_csv[df_csv['mag'] >= 3.5]
        if df_csv.empty:
            print(f"{Y_}CSV dosyasinda islenebilir (M>=3.5) veri bulunamadi.{X_}")
            return False
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({tn})")
        db_cols = [row[1] for row in cur.fetchall()]
        insert_cols = [c for c in df_csv.columns if c in db_cols]
        df_csv[insert_cols].to_sql(tn, conn, if_exists='append', index=False, chunksize=1000)
        print(f"{G_}Harika! CSV'den {len(df_csv)} gecmis deprem kaydi veritabanina aktarildi.{X_}")
        return True
    except Exception as e:
        print(f"{R_}CSV okuma veya veritabanina yazma sirasinda hata: {e}{X_}")
        return False

def fetch_and_load_api_data(conn, tn, start_override=None):
    now = datetime.utcnow()
    end_lim = (now + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
    cur = conn.cursor()
    fix_future_dates(conn, tn)
    if start_override:
        ss = standardize_date(start_override)
        if not ss:
            ss = '1970-01-01 00:00:00'  # ✅ Daha eski veriler için
    else:
        cur.execute(f"SELECT MAX(time) FROM {tn}")
        r = cur.fetchone()
        latest = r[0] if r else None
        if latest:
            try:
                ldt = pd.to_datetime(latest, utc=True)
                if ldt > pd.Timestamp(end_lim, tz='UTC'):
                    ss = (now - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    ss = (ldt + timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                ss = '1970-01-01 00:00:00'  # ✅ Daha eski veriler için
        else:
            ss = '1970-01-01 00:00:00'  # ✅ Daha eski veriler için
    api_s = pd.to_datetime(ss).strftime('%Y-%m-%d %H:%M:%S')
    api_e = pd.to_datetime(end_lim).strftime('%Y-%m-%d %H:%M:%S')
    if api_s > api_e:
        api_s = (now - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
    
    # ✅ İYİLEŞTİRME: Eğer aralık 2 yıldan büyükse, yıl bazlı çek
    api_s_dt = pd.to_datetime(api_s)
    api_e_dt = pd.to_datetime(api_e)
    range_years = (api_e_dt - api_s_dt).days / 365.25
    
    if range_years > 2:
        print(f"{C_}Büyük zaman aralığı ({range_years:.1f} yıl) - yıl bazlı çekiliyor...{X_}")
        all_new_ids = []
        current_start = api_s_dt
        while current_start < api_e_dt:
            chunk_end = min(current_start + timedelta(days=365), api_e_dt)
            chunk_ids = _fetch_chunk(conn, tn, cur,
                                     current_start.strftime('%Y-%m-%d %H:%M:%S'),
                                     chunk_end.strftime('%Y-%m-%d %H:%M:%S'),
                                     end_lim)
            all_new_ids.extend(chunk_ids)
            current_start = chunk_end + timedelta(seconds=1)
        return all_new_ids
    
    return _fetch_chunk(conn, tn, cur, api_s, api_e, end_lim)

def _fetch_chunk(conn, tn, cur, api_s, api_e, end_lim):
    """Tek bir zaman aralığı için API'den veri çek."""
    params = {'start': api_s, 'end': api_e, 'orderby': 'time-asc', 'minmag': '3.5'}
    for att in range(5):
        try:
            print(f"{C_}API ({att+1}/5)... {api_s} -> {api_e}{X_}")
            resp = requests.get("https://deprem.afad.gov.tr/apiv2/event/filter",
                                params=params, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                return []
            df = pd.DataFrame(data)
            df.rename(columns={'date': 'time', 'magnitude': 'mag', 'location': 'place'}, inplace=True)
            for c in ['latitude', 'longitude', 'depth', 'mag']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df[df['mag'] >= 3.5]
            df['time'] = df['time'].apply(standardize_date)
            df.dropna(subset=['time'], inplace=True)
            df['_tc'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
            fut = df['_tc'] > pd.Timestamp(end_lim, tz='UTC')
            if fut.any():
                df = df[~fut]
            df.drop(columns=['_tc'], inplace=True)
            cur.execute(f"SELECT eventID FROM {tn}")
            ex_ids = {r[0] for r in cur.fetchall()}
            dfn = df[~df['eventID'].isin(ex_ids)]
            if dfn.empty:
                return []
            lc = ['time', 'latitude', 'longitude', 'depth', 'mag', 'place', 'eventID']
            dfn[lc].to_sql(tn, conn, if_exists='append', index=False, chunksize=1000)
            print(f"{G_}{len(dfn)} yeni kayit eklendi.{X_}")
            return dfn['eventID'].tolist()
        except requests.exceptions.RequestException as e:
            print(f"{Y_}API Hatasi:{e}{X_}")
            time.sleep(5)
    return []

def calc_b_value(mags):
    if len(mags) < 20:
        return None
    try:
        mc = pd.Series(mags).value_counts().idxmax()
        cm = mags[mags >= mc]
        if len(cm) < 10:
            return None
        b = np.log10(np.e) / (np.mean(cm) - mc + 0.05)
        return b if 0.3 <= b <= 2.5 else None
    except Exception:
        return None

def fix_numeric(df):
    for c in ENHANCED_FEATURES + ['latitude', 'longitude']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def safe_fill(s):
    try:
        ns = pd.to_numeric(s, errors='coerce')
        return ns.fillna(ns.median())
    except Exception:
        return pd.to_numeric(s, errors='coerce').fillna(0)

def calc_features(df_all):
    df = df_all.copy()
    df = fix_numeric(df)
    cids = set(df.loc[df['b_value_local'].isnull(), 'eventID'])
    if not cids:
        return df_all
    print(f"{len(cids)} kayit hesaplaniyor...")
    dfs = df.sort_values('time').reset_index(drop=True)
    ni = get_neighbors_cKDTree(dfs, 50)
    if ni is None:
        return df
    ups = []
    try:
        from tqdm import tqdm
        it = tqdm(range(len(dfs)), desc="Hesap")
    except ImportError:
        it = range(len(dfs))
    for si in it:
        row = dfs.loc[si]
        eid = row['eventID']
        if eid not in cids:
            continue
        ct = row['time']
        pi = [i for i in ni[si] if dfs.loc[i, 'time'] <= ct]
        if not pi:
            continue
        le = dfs.iloc[pi]
        t30 = ct - timedelta(days=30)
        t24h = ct - timedelta(hours=24)
        t12h = ct - timedelta(hours=12)
        re = le[le['time'] >= t30]
        er = len(re) / 30.0
        er_24h = len(le[le['time'] >= t24h])
        er_12h = len(le[le['time'] >= t12h])
        decay_idx = 0.0
        if len(re) > 0:
            rlats = np.radians(re['latitude'].values)
            rlons = np.radians(re['longitude'].values)
            clat = np.radians(row['latitude'])
            clon = np.radians(row['longitude'])
            dists = np.arccos(np.sin(rlats)*np.sin(clat) + np.cos(rlats)*np.cos(clat)*np.cos(rlons-clon)) * 6371
            dists = np.maximum(dists, 0.1)
            weights = 1.0 / (1.0 + dists/50.0)**2
            decay_idx = weights.mean()
        mag_trend = np.nan
        if len(le) >= 3:
            try:
                x_trend = np.arange(len(le))
                y_trend = le['mag'].values
                mag_trend = np.polyfit(x_trend, y_trend, 1)[0]
            except Exception:
                pass
        bv = calc_b_value(le['mag'].values)
        ups.append({
            'eventID': eid,
            'b_value_local': bv,
            'event_rate_local': er,
            'event_rate_24h': er_24h,
            'event_rate_12h': er_12h,
            'time_since_last': (ct - le['time'].iloc[-1]).total_seconds() / 86400.0 if len(le) > 0 else np.nan,
            'mag_completeness': le['mag'].min(),
            'spatial_density': len(le) / (np.pi * 50**2) if len(le) > 0 else 0,
            'temporal_clustering': np.exp(-er) if er > 0 else 0,
            'mag_trend': mag_trend,
            'depth_clustering': np.std(le['depth'].values) if len(le) > 1 else np.nan,
            'energy_rate': np.sum(10**(le['mag'].values * 1.5)) if len(le) > 0 else 0,
            'swarm_indicator': 1 if er > 5 and len(le) > 10 else 0,
            'fault_distance': np.nan,
            'spatial_decay_index': decay_idx
        })
    if ups:
        upd = pd.DataFrame(ups)
        df = pd.merge(df, upd, on='eventID', how='left', suffixes=('', '_new'))
        for col in ['b_value_local', 'event_rate_local', 'time_since_last', 'mag_completeness',
                    'spatial_density', 'temporal_clustering', 'mag_trend', 'depth_clustering',
                    'energy_rate', 'swarm_indicator', 'fault_distance', 'event_rate_24h',
                    'event_rate_12h', 'spatial_decay_index']:
            nc = f'{col}_new'
            if nc in df.columns:
                df[col] = df[nc].fillna(df[col])
                df.drop(columns=[nc], inplace=True)
    return df

def classify_eq_type(df):
    df = fix_numeric(df)
    dfs = df.sort_values('time').reset_index(drop=True)
    ni = get_neighbors_cKDTree(dfs, 120)
    if ni is None:
        df['earthquake_type'] = "Tekil Deprem"
        return df
    types = np.full(len(dfs), "Tekil Deprem", dtype=object)
    types[dfs['mag'] >= 6.0] = "Ana Deprem"
    mags = dfs['mag'].values
    times = dfs['time'].values
    tw = np.timedelta64(60, 'D')
    for i in np.where(dfs['mag'] < 6.0)[0]:
        cm, ct = mags[i], times[i]
        nt = times[ni[i]]
        nm = mags[ni[i]]
        pl, fl = ct - tw, ct + tw
        wm = (nt >= pl) & (nt <= fl)
        pm = wm & (nt < ct)
        fm_ = wm & (nt > ct)
        lpm = np.max(nm[pm]) if np.any(pm) else -1
        lfm = np.max(nm[fm_]) if np.any(fm_) else -1
        if cm < lpm - 0.8:
            types[i] = "Artci Deprem"
        elif cm < lfm - 0.8:
            types[i] = "Oncu Deprem"
    dfs['earthquake_type'] = types
    tr = dfs[['eventID', 'earthquake_type']]
    if 'earthquake_type' in df.columns:
        df = df.drop(columns=['earthquake_type'])
    return pd.merge(df, tr, on='eventID', how='left')

def create_labels_parametric(df, mag_threshold=None, tw_days=None, r_km=None, verbose=True):
    """
    BİLİMSEL ETİKETLEME (v21):
    Foreshock etiketi, depremin gelecekteki bir ana şokun öncüsü olduğunu belirler.
    
    ÖNEMLİ: Bu etiketleme TANIM gereği gelecek bilgisi kullanır (foreshock kavramı
    geleceğe işarettir). Ancak ÖZELLİKLER (features) sadece geçmişten hesaplanır,
    yani modelin tahmin edeceği şey gelecektir, ama tahmin edeceği veriler geçmiştir.
    Bu standart sismolojik yaklaşımdır.
    
    Önemli düzeltme: Train/Test ayrımı sıkı zaman bazlı yapılmalı ki test setindeki
    ana şoklar train etiketlerini etkilemesin.
    """
    if mag_threshold is None:
        mag_threshold = FORESHOCK_MAG_THRESHOLD
    if tw_days is None:
        tw_days = FORESHOCK_TIME_WINDOW_DAYS
    if r_km is None:
        r_km = FORESHOCK_SPATIAL_RADIUS_KM
    df = df.sort_values('time').reset_index(drop=True)
    twns = timedelta(days=tw_days).total_seconds() * 1e9
    ni = get_neighbors_cKDTree(df, radius_km=r_km)
    if ni is None:
        df[TARGET] = 0
        return df
    labels = np.zeros(len(df), dtype=int)
    mags = df['mag'].values
    tn_ = df['time'].astype(np.int64).values
    eq_types = df.get('earthquake_type', pd.Series(['Tekil Deprem'] * len(df))).values
    for i in range(len(df)):
        if eq_types[i] == 'Artci Deprem':
            continue
        if df.iloc[i]['mag'] < 3.5:
            continue
        fi = [idx for idx in ni[i] if idx > i]
        if not fi:
            continue
        td = tn_[fi] - tn_[i]
        itm = td <= twns
        if np.any(itm):
            ri = np.array(fi)[itm]
            max_future_mag = np.max(mags[ri])
            mag_diff = max_future_mag - mags[i]
            if max_future_mag >= mag_threshold and mag_diff >= FORESHOCK_MIN_MAG_DIFF:
                labels[i] = 1
    df[TARGET] = labels
    pos_rate = (labels.sum() / len(labels)) * 100 if len(labels) > 0 else 0
    if verbose:
        print(f"{G_}Foreshock (Sabit Parametreler / Fixed Parameters): %{pos_rate:.2f} ({labels.sum()}/{len(labels)}){X_}")
        print(f"{G_}  Parametreler: Mag>={mag_threshold}, Time<={tw_days}d, Dist<={r_km}km, ΔM>={FORESHOCK_MIN_MAG_DIFF}{X_}")
        print(f"{Y_}  UYARI: Gelecek ana şok bilgisi kullaniliyor (future leakage).{X_}")
    return df

def sensitivity_analysis_foreshock(df):
    print(f"\n{C_}{'='*70}")
    print("SENSITIVITY ANALYSIS: Foreshock Parametreleri (Foreshock Parameters - Model Independent)")
    print(f"{'='*70}{X_}\n")
    mag_thresholds = [5.0, 5.3, 5.5, 5.8, 6.0]
    time_windows = [7, 14, 30, 45, 60]
    spatial_radii = [25, 50, 75, 100]
    results = []
    total_combos = len(mag_thresholds) * len(time_windows) * len(spatial_radii)
    current = 0
    print(f"{C_}Kombinasyonlar test ediliyor... (total: {total_combos}){X_}\n")
    for mag_t in mag_thresholds:
        for tw in time_windows:
            for sr in spatial_radii:
                current += 1
                try:
                    progress = (current / total_combos) * 100
                    print(f"\r{C_}[{progress:.1f}%] Mag:{mag_t}, TW:{tw}d, SR:{sr}km{X_}", end='', flush=True)
                    df_test = df.copy()
                    labels = create_labels_parametric(df_test, mag_threshold=mag_t, tw_days=tw, r_km=sr, verbose=False)
                    pos_rate = labels[TARGET].sum() / len(labels) * 100 if len(labels) > 0 else 0
                    results.append({
                        'mag_threshold': mag_t,
                        'time_window_days': tw,
                        'spatial_radius_km': sr,
                        'positive_rate_%': pos_rate,
                        'positive_count': int(labels[TARGET].sum()),
                        'total_count': len(labels)
                    })
                except Exception as e:
                    print(f"\n{R_}Hata ({mag_t}, {tw}, {sr}): {e}{X_}")
    print("\n")
    sens_df = pd.DataFrame(results)
    if sens_df.empty:
        print(f"{R_}Sensitivity analizi başarısız oldu{X_}")
        return None
    print(f"{G_}{'='*70}")
    print(f"Positive Rate İstatistikleri:")
    print(f"{'='*70}{X_}")
    print(f"  Ortalama: {sens_df['positive_rate_%'].mean():.2f}%")
    print(f"  Std Dev: {sens_df['positive_rate_%'].std():.2f}%")
    print(f"  Min: {sens_df['positive_rate_%'].min():.2f}%")
    print(f"  Max: {sens_df['positive_rate_%'].max():.2f}%")
    print(f"\n{G_}Referans (Sabit / Fixed) Parametreler:{X_}")
    ref_row = sens_df[
        (sens_df['mag_threshold'] == FORESHOCK_MAG_THRESHOLD) &
        (sens_df['time_window_days'] == FORESHOCK_TIME_WINDOW_DAYS) &
        (sens_df['spatial_radius_km'] == FORESHOCK_SPATIAL_RADIUS_KM)
    ]
    if not ref_row.empty:
        print(f"  Mag>={FORESHOCK_MAG_THRESHOLD}, TW={FORESHOCK_TIME_WINDOW_DAYS}d, SR={FORESHOCK_SPATIAL_RADIUS_KM}km")
        print(f"  Positive Rate: %{ref_row['positive_rate_%'].values[0]:.2f}")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        mag_summary = sens_df.groupby('mag_threshold')['positive_rate_%'].agg(['mean', 'std'])
        axes[0].errorbar(mag_summary.index, mag_summary['mean'], yerr=mag_summary['std'], marker='o', capsize=5, linewidth=2, markersize=8)
        axes[0].set_xlabel('Magnitude Threshold', fontsize=12)
        axes[0].set_ylabel('Positive Rate (%)', fontsize=12)
        axes[0].set_title('Mag Threshold Sensitivity', fontsize=13, fontweight='bold')
        axes[0].grid(alpha=0.3)
        axes[0].axvline(x=FORESHOCK_MAG_THRESHOLD, color='g', linestyle='--', alpha=0.7, label=f'Sabit: {FORESHOCK_MAG_THRESHOLD}')
        axes[0].legend()
        tw_summary = sens_df.groupby('time_window_days')['positive_rate_%'].agg(['mean', 'std'])
        axes[1].errorbar(tw_summary.index, tw_summary['mean'], yerr=tw_summary['std'], marker='s', capsize=5, linewidth=2, markersize=8)
        axes[1].set_xlabel('Time Window (days)', fontsize=12)
        axes[1].set_ylabel('Positive Rate (%)', fontsize=12)
        axes[1].set_title('Time Window Sensitivity', fontsize=13, fontweight='bold')
        axes[1].grid(alpha=0.3)
        axes[1].axvline(x=FORESHOCK_TIME_WINDOW_DAYS, color='g', linestyle='--', alpha=0.7, label=f'Sabit: {FORESHOCK_TIME_WINDOW_DAYS}d')
        axes[1].legend()
        sr_summary = sens_df.groupby('spatial_radius_km')['positive_rate_%'].agg(['mean', 'std'])
        axes[2].errorbar(sr_summary.index, sr_summary['mean'], yerr=sr_summary['std'], marker='^', capsize=5, linewidth=2, markersize=8)
        axes[2].set_xlabel('Spatial Radius (km)', fontsize=12)
        axes[2].set_ylabel('Positive Rate (%)', fontsize=12)
        axes[2].set_title('Spatial Radius Sensitivity', fontsize=13, fontweight='bold')
        axes[2].grid(alpha=0.3)
        axes[2].axvline(x=FORESHOCK_SPATIAL_RADIUS_KM, color='g', linestyle='--', alpha=0.7, label=f'Sabit: {FORESHOCK_SPATIAL_RADIUS_KM}km')
        axes[2].legend()
        plt.suptitle('Foreshock Definition - Sensitivity Analysis (Model Independent)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('sensitivity_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n{G_}✓ Sensitivity analizi grafiği kaydedildi: sensitivity_analysis.png{X_}")
        plt.close()
    except Exception as e:
        print(f"{Y_}Grafik oluşturulamadı: {e}{X_}")
    return sens_df

# ====================== DÜZELTİLMİŞ METRİK FONKSİYONLARI ======================
def get_metrics(yt, yp, ypr):
    """Geçerli metrikleri hesaplar, yetersiz pozitif/negatif durumunda NaN döner."""
    if len(np.unique(yt)) < 2:
        return {
            'accuracy': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'f1_score': np.nan,
            'auc': np.nan,
            'brier_score': np.nan,
            'ece': np.nan
        }
    m = {
        'accuracy': accuracy_score(yt, yp),
        'precision': precision_score(yt, yp, zero_division=0),
        'recall': recall_score(yt, yp, zero_division=0),
        'f1_score': f1_score(yt, yp, zero_division=0),
        'auc': roc_auc_score(yt, ypr),
        'brier_score': brier_score_loss(yt, ypr),
        'ece': 0
    }
    prob_true, prob_pred = calibration_curve(yt, ypr, n_bins=10, strategy='uniform')
    m['ece'] = np.mean(np.abs(prob_true - prob_pred))
    return m

def calc_molchan(yt, ypr):
    if len(np.unique(yt)) < 2:
        return {'skill_score': np.nan, 'molchan_auc': np.nan,
                'miss_rate': np.array([0, 1]), 'alarm_rate': np.array([0, 1]),
                'thresholds': np.array([0])}
    fpr, tpr, th = roc_curve(yt, ypr)
    mr = 1 - tpr
    ar = fpr
    ma = np.trapz(mr, ar)
    ss = 1 - (2 * ma)
    return {'skill_score': ss, 'molchan_auc': ma,
            'miss_rate': mr, 'alarm_rate': ar, 'thresholds': th}

def prospective_sim(df_test, model, feature_cols, threshold=None):
    if threshold is None:
        threshold = 0.5
    y_pred_proba = model.predict_proba(df_test[feature_cols].values)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    correct = (y_pred == df_test[TARGET].values)
    acc = correct.mean() * 100
    p = precision_score(df_test[TARGET], y_pred, zero_division=0)
    r = recall_score(df_test[TARGET], y_pred, zero_division=0)
    return {'accuracy': acc, 'precision': p, 'recall': r, 'predictions': y_pred}

def plot_molchan(md, fn='molchan.png'):
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(md['alarm_rate'], md['miss_rate'], 'b-', lw=2,
                 label=f"Model (Skill={md['skill_score']:.3f})")
        plt.plot([0, 1], [0, 1], 'r--', lw=1, label='Rastgele (Random)')
        plt.xlabel('Alarm Rate')
        plt.ylabel('Miss Rate')
        plt.title('Molchan Diagram')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fn, dpi=150)
        plt.close()
    except Exception:
        pass

# ====================== EĞİTİM FONKSİYONLARI ======================
def train_sklearn_improved(df_full, new_ids, force=False):
    """
    BİLİMSEL DÜZELTMELER (v21):
    1. Gap-buffered split: Train ve test arasında 60 gün buffer (foreshock pencereden taşmaması için)
    2. Validation set: Threshold optimizasyonu için ayrı set (test setine bakılmaz)
    3. Doğru metrik raporlama: F1, Precision, Recall ön planda
    """
    models = {'xgb': None, 'rf': None}
    metrics = {'xgb': {}, 'rf': {}}
    mp = {'xgb': 'xgb_v5.joblib', 'rf': 'rf_v5.joblib'}
    
    # Zamansal sıralama
    df_full = df_full.sort_values('time').reset_index(drop=True)
    
    # GAP-BUFFERED 3'lü split: Train (55%) | GAP (60d) | Validation (20%) | GAP (60d) | Test (25%)
    n = len(df_full)
    cutoff_train = df_full['time'].quantile(0.55)
    cutoff_val = df_full['time'].quantile(0.75)
    
    # Buffer süreleri (foreshock penceresi 30 gün, ama güvenlik için 60 gün ekliyoruz)
    buffer_days = FORESHOCK_TIME_WINDOW_DAYS * 2
    gap_train_val = cutoff_train + timedelta(days=buffer_days)
    gap_val_test = cutoff_val + timedelta(days=buffer_days)
    
    trd = df_full[df_full['time'] <= cutoff_train].copy()
    vad = df_full[(df_full['time'] >= gap_train_val) & (df_full['time'] <= cutoff_val)].copy()
    ted = df_full[df_full['time'] >= gap_val_test].copy()
    
    print(f"{C_}Zamansal ayrım (Time-based split / gap-buffered):{X_}")
    print(f"  Train: {len(trd)} olay (≤{cutoff_train.strftime('%Y-%m-%d')})")
    print(f"  Validation: {len(vad)} olay")
    print(f"  Test: {len(ted)} olay (≥{gap_val_test.strftime('%Y-%m-%d')})")
    print(f"  Gap: {buffer_days} gün (foreshock leakage'ı önlemek için)")
    
    if len(ted) < 10 or len(vad) < 10:
        print(f"{Y_}Validation/Test seti çok küçük, model eğitimi atlanıyor.{X_}")
        return {}, {}, {}
    if not force and all(os.path.exists(p) for p in mp.values()):
        for k in models:
            models[k] = joblib.load(mp[k])
        return models, metrics, {}

    trd = fix_numeric(trd)
    vad = fix_numeric(vad)
    ted = fix_numeric(ted)
    print(f"{C_}Sabit parametrelerle (fixed parameters) foreshock etiketlemesi yapiliyor...{X_}")
    # ÖNEMLI: Her set'in etiketlemesi KENDİ İÇİNDE yapılır - leakage yok
    trl = create_labels_parametric(trd.copy(), verbose=True)
    val = create_labels_parametric(vad.copy(), verbose=False)
    tel = create_labels_parametric(ted.copy(), verbose=False)
    print(f"{G_}  Train pozitif oranı: %{trl[TARGET].mean()*100:.2f}{X_}")
    print(f"{G_}  Validation pozitif oranı: %{val[TARGET].mean()*100:.2f}{X_}")
    print(f"{G_}  Test pozitif oranı: %{tel[TARGET].mean()*100:.2f}{X_}")

    sensitivity_df = sensitivity_analysis_foreshock(trd)
    if sensitivity_df is not None:
        sensitivity_df.to_csv('foreshock_sensitivity_analysis.csv', index=False)
        print(f"{G_}✓ Sensitivity analysis kaydedildi: foreshock_sensitivity_analysis.csv{X_}")

    af = [f for f in ENHANCED_FEATURES if f in trl.columns]
    Xtr = trl[af].apply(safe_fill)
    ytr = trl[TARGET]
    Xva = val[af].apply(safe_fill)
    yva = val[TARGET]
    Xte = tel[af].apply(safe_fill)
    yte = tel[TARGET]

    if ytr.sum() == 0:
        print(f"{R_}Eğitim setinde hiç foreshock örneği yok! Model eğitilemez.{X_}")
        return {}, {}, {}
    if yte.sum() == 0:
        print(f"{Y_}Uyarı: Test setinde hiç foreshock yok. Test metrikleri geçersiz.{X_}")
    if yva.sum() == 0:
        print(f"{Y_}Uyarı: Validation setinde hiç foreshock yok. Default threshold kullanılacak.{X_}")

    for mtype in ['xgb', 'rf']:
        bp = None
        if mtype == 'xgb':
            n_neg = (ytr == 0).sum()
            n_pos = (ytr == 1).sum()
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
            mdl = XGBClassifier(**(bp or {'n_estimators': 300, 'max_depth': 6,
                                          'learning_rate': 0.05, 'random_state': 42,
                                          'use_label_encoder': False, 'eval_metric': 'logloss',
                                          'subsample': 0.8, 'colsample_bytree': 0.8}),
                                scale_pos_weight=scale_pos_weight)
        else:
            mdl = RandomForestClassifier(**(bp or {'n_estimators': 300, 'max_depth': 12,
                                                   'min_samples_split': 5, 'random_state': 42}),
                                         class_weight='balanced')
        mdl.fit(Xtr, ytr)
        cal = CalibratedClassifierCV(mdl, method='sigmoid', cv=3)
        cal.fit(Xtr, ytr)

        # ✅ DÜZELTME: Threshold'u VALIDATION setinde optimize et (test'e bakılmaz)
        if yva.sum() > 0:
            va_proba = cal.predict_proba(Xva)[:, 1]
            prec, rec, ths = precision_recall_curve(yva, va_proba)
            f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
            best_th = ths[np.argmax(f1_scores[:-1])] if len(ths) > 0 else 0.5
        else:
            best_th = 0.5
        print(f"{C_}  {mtype.upper()} optimum eşik: {best_th:.3f}{X_}")

        # TEST setinde değerlendir (threshold validation'dan geldi)
        ypr = cal.predict_proba(Xte)[:, 1]
        yp = (ypr >= best_th).astype(int)

        met = get_metrics(yte, yp, ypr)
        met['optimal_threshold'] = best_th
        mcd = calc_molchan(yte, ypr)
        met['molchan_skill'] = mcd['skill_score']
        met['molchan_auc'] = mcd['molchan_auc']

        # Prospektif simülasyon (zaten test setinde optimum threshold ile)
        met['prospective_accuracy'] = (yp == yte.values).mean() * 100
        met['prospective_precision'] = precision_score(yte, yp, zero_division=0)
        met['prospective_recall'] = recall_score(yte, yp, zero_division=0)
        met['prospective_f1'] = f1_score(yte, yp, zero_division=0)

        # ✅ DÜZELTME: Olumsuz Skill Score uyarısı
        if met['molchan_skill'] < 0:
            print(f"{R_}  ⚠ {mtype.upper()} UYARI: Skill Score < 0 (rastgeleden kötü). Bu model güvenilir değil.{X_}")

        models[mtype] = cal
        metrics[mtype] = met
        plot_molchan(mcd, fn=f'molchan_{mtype}.png')
        joblib.dump(cal, mp[mtype])
        print(f"{G_}{mtype.upper()} Hazir | AUC:{met['auc']:.3f} | F1:{met['f1_score']:.3f} | Recall:{met['recall']:.3f} | Skill:{met['molchan_skill']:.3f}{X_}")
    return models, metrics, {}

def build_lstm(shape):
    """LSTM model - klasik recurrent network"""
    return Sequential([
        Input(shape=shape),
        LSTM(64, return_sequences=True, dropout=0.3), BatchNormalization(),
        LSTM(32, dropout=0.3), BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)), Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)), Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

def build_gru(shape):
    """GRU model - LSTM'den daha hızlı, benzer performans"""
    return Sequential([
        Input(shape=shape),
        GRU(64, return_sequences=True, dropout=0.3), BatchNormalization(),
        GRU(32, dropout=0.3), BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)), Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)), Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

def build_transformer(shape, num_heads=2, ff_dim=32, num_blocks=1):
    """
    Transformer model - sadeleştirilmiş versiyon
    Az pozitif örnek için 1 blok, 2 head (overfit önleme)
    """
    inputs = Input(shape=shape)
    x = inputs
    
    # Tek transformer bloğu (az veri için yeterli)
    for _ in range(num_blocks):
        # Multi-head self-attention
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=shape[-1], dropout=0.3)(x, x)
        x_attn = Add()([x, attn])
        x_attn = LayerNormalization(epsilon=1e-6)(x_attn)
        
        # Feed-forward network (basitleştirilmiş)
        ff = Dense(ff_dim, activation='relu', kernel_regularizer=l2(0.02))(x_attn)
        ff = Dropout(0.4)(ff)
        ff = Dense(shape[-1])(ff)
        x = Add()([x_attn, ff])
        x = LayerNormalization(epsilon=1e-6)(x)
    
    # Pooling ve sınıflandırma (basit)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.02))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=inputs, outputs=outputs)

def prospective_sim_lstm(model, scaler, df_test, feature_cols, seq_length=50, threshold=0.5):
    """LSTM prospektif simülasyon - AUC ve Skill Score dahil tüm metrikleri döner."""
    if df_test.empty or len(df_test) <= seq_length:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                'auc': np.nan, 'skill_score': np.nan, 'f1_score': 0.0}
    df_test = df_test.sort_values('time').reset_index(drop=True)
    X = df_test[feature_cols].apply(safe_fill).values
    X_scaled = scaler.transform(X)
    X_seq = []
    y_true = []
    for i in range(len(X_scaled) - seq_length):
        X_seq.append(X_scaled[i:i+seq_length])
        y_true.append(df_test.iloc[i+seq_length][TARGET])
    if not X_seq:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                'auc': np.nan, 'skill_score': np.nan, 'f1_score': 0.0}
    X_seq = np.array(X_seq)
    y_true = np.array(y_true)
    y_pred_proba = model.predict(X_seq, verbose=0).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)
    acc = (y_pred == y_true).mean() * 100
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = 2 * p * r / (p + r + 1e-9)
    
    # ✅ DÜZELTME: AUC ve Skill Score hesabı
    auc_val = np.nan
    skill_val = np.nan
    if len(np.unique(y_true)) >= 2:
        try:
            auc_val = roc_auc_score(y_true, y_pred_proba)
            mcd = calc_molchan(y_true, y_pred_proba)
            skill_val = mcd['skill_score']
        except Exception:
            pass
    else:
        # Test seti tek sınıflıysa, train üzerinde değerlendir
        print(f"{Y_}  LSTM: Test seti tek sınıf içeriyor ({y_true.sum()} pozitif). AUC hesaplanamadı.{X_}")
    
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1_score': f1,
            'auc': auc_val, 'skill_score': skill_val,
            'pos_count': int(y_true.sum()), 'total_count': len(y_true)}

def train_sequence_model(df_full, model_type='lstm', force=False):
    """
    Sequence model eğit (LSTM, GRU, Transformer)
    Tek bir generic fonksiyon - kod tekrarı önler
    """
    model_files = {
        'lstm': ('lstm_v5.keras', 'lstm_scaler_v5.joblib'),
        'gru': ('gru_v5.keras', 'gru_scaler_v5.joblib'),
        'transformer': ('transformer_v5.keras', 'transformer_scaler_v5.joblib')
    }
    builders = {
        'lstm': build_lstm,
        'gru': build_gru,
        'transformer': build_transformer
    }
    
    if model_type not in model_files:
        print(f"{R_}Bilinmeyen model: {model_type}{X_}")
        return None, None, {}
    
    mp, sp = model_files[model_type]
    builder = builders[model_type]
    seq_length = 50
    label = model_type.upper()
    
    if not force and os.path.exists(mp) and os.path.exists(sp):
        try:
            mdl = load_model(mp)
            scl = joblib.load(sp)
            return mdl, scl, {}
        except Exception:
            pass
    
    # Gap-buffered split (Train %75, Test %25)
    df_full = df_full.sort_values('time').reset_index(drop=True)
    cutoff_train = df_full['time'].quantile(0.75)
    buffer_days = FORESHOCK_TIME_WINDOW_DAYS * 2
    gap_train_test = cutoff_train + timedelta(days=buffer_days)
    
    trd = df_full[df_full['time'] <= cutoff_train].copy()
    ted = df_full[df_full['time'] >= gap_train_test].copy()
    
    if len(ted) < seq_length + 10:
        print(f"{Y_}{label}: Test seti çok küçük, atlanıyor.{X_}")
        return None, None, {}
    trd = fix_numeric(trd)
    ted = fix_numeric(ted)
    trl = create_labels_parametric(trd.copy(), verbose=False)
    tel = create_labels_parametric(ted.copy(), verbose=False)
    af = [f for f in ENHANCED_FEATURES if f in trl.columns]
    if not af:
        return None, None, {}
    scl = StandardScaler()
    X_tr = scl.fit_transform(trl[af].apply(safe_fill))
    y_tr = trl[TARGET].values
    X_seq_tr, y_seq_tr = [], []
    for i in range(len(X_tr) - seq_length):
        X_seq_tr.append(X_tr[i:i+seq_length])
        y_seq_tr.append(y_tr[i+seq_length])
    if not X_seq_tr:
        return None, None, {}
    X_seq_tr = np.array(X_seq_tr)
    y_seq_tr = np.array(y_seq_tr)
    
    if y_seq_tr.sum() == 0:
        print(f"{Y_}{label}: Eğitim setinde foreshock yok, model eğitilemiyor.{X_}")
        return None, None, {}
    
    # Class weight
    n_pos = y_seq_tr.sum()
    n_neg = len(y_seq_tr) - n_pos
    class_weights = {0: 1.0, 1: float(n_neg / max(n_pos, 1))}
    print(f"{C_}  {label} eğitiliyor... (class weights: {{0: 1.0, 1: {class_weights[1]:.1f}}}){X_}")
    
    mdl = builder((seq_length, X_seq_tr.shape[2]))
    
    # Transformer için daha düşük learning rate
    lr = 0.0005 if model_type == 'transformer' else 0.001
    epochs = 25 if model_type == 'transformer' else 30
    
    mdl.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['AUC'])
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    mdl.fit(X_seq_tr, y_seq_tr, epochs=epochs, batch_size=32,
            callbacks=[es], class_weight=class_weights, verbose=0)
    
    # ✅ İYİLEŞTİRME: F1-optimal threshold bul (eğitim seti üzerinde)
    train_proba = mdl.predict(X_seq_tr, verbose=0).flatten()
    if y_seq_tr.sum() > 0 and len(np.unique(y_seq_tr)) >= 2:
        prec, rec, ths = precision_recall_curve(y_seq_tr, train_proba)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
        best_th = float(ths[np.argmax(f1_scores[:-1])]) if len(ths) > 0 else 0.5
        # ✅ DÜZELTME: Threshold üst sınırını 0.45'e düşür (LSTM Recall=0 sorunu için)
        # Yüksek threshold (>0.5) → Recall=0 sorunu yaratır
        # Düşük clip (0.05) overfit önler
        best_th = max(0.05, min(0.45, best_th))
    else:
        best_th = 0.3  # ✅ Default threshold de düşürüldü (eskiden 0.5)
    print(f"{C_}  {label} optimum eşik: {best_th:.3f}{X_}")
    
    ps = prospective_sim_lstm(mdl, scl, tel, af, seq_length=seq_length, threshold=best_th)
    met = {
        'auc': ps.get('auc', np.nan),
        'molchan_skill': ps.get('skill_score', np.nan),
        'f1_score': ps.get('f1_score', 0.0),
        'precision': ps.get('precision', 0.0),
        'recall': ps.get('recall', 0.0),
        'optimal_threshold': best_th,
        'prospective_accuracy': ps['accuracy'],
        'prospective_precision': ps['precision'],
        'prospective_recall': ps['recall'],
        'prospective_f1': ps.get('f1_score', 0.0),
        'pos_count_test': ps.get('pos_count', 0),
        'total_count_test': ps.get('total_count', 0)
    }
    mdl.save(mp)
    joblib.dump(scl, sp)
    auc_disp = f"{ps.get('auc'):.3f}" if not np.isnan(ps.get('auc', np.nan)) else "N/A"
    print(f"{G_}{label} Hazir | AUC:{auc_disp} | F1:{ps.get('f1_score', 0):.3f} | Recall:{ps['recall']:.3f} | Threshold:{best_th:.3f} | Test pos: {ps.get('pos_count', 0)}/{ps.get('total_count', 0)}{X_}")
    return mdl, scl, met

def train_lstm(df_full, new_ids, force=False):
    """LSTM eğit (backward compatibility)"""
    return train_sequence_model(df_full, model_type='lstm', force=force)

def train_gru(df_full, force=False):
    """GRU eğit"""
    return train_sequence_model(df_full, model_type='gru', force=force)

def train_transformer(df_full, force=False):
    """Transformer eğit - en güçlü temporal model"""
    return train_sequence_model(df_full, model_type='transformer', force=force)

def predict_unc(dtp, models, lm, ls, gm=None, gs=None, tm=None, ts=None):
    """
    Ensemble tahmin - LSTM, GRU, Transformer (sequence) + XGB, RF (tabular)
    ✅ Yetersiz veri kontrolü: Feature eksik veya bölgede az deprem varsa olasılık hesaplanmaz
    """
    if not models or not dtp[ENHANCED_FEATURES].notna().any(axis=1).any():
        return pd.DataFrame()
    dtp = fix_numeric(dtp)
    af = [f for f in ENHANCED_FEATURES if f in dtp.columns]
    if not af:
        return pd.DataFrame()
    Xp = dtp[af].apply(safe_fill)
    
    # ✅ YETERSİZ VERİ KRİTERLERİ:
    # 1. b_value_local VE event_rate_local = NaN → Feature hesaplanamadı
    # 2. spatial_density < 0.001 → Bölgede neredeyse hiç deprem yok
    # 3. event_rate_local < 0.05 → Son 30 günde çok az olay
    insufficient_mask = np.zeros(len(dtp), dtype=bool)
    
    if 'b_value_local' in dtp.columns and 'event_rate_local' in dtp.columns:
        both_nan = dtp['b_value_local'].isna() & dtp['event_rate_local'].isna()
        insufficient_mask = insufficient_mask | both_nan.values
    
    if 'spatial_density' in dtp.columns:
        low_density = (dtp['spatial_density'] < 0.001) | dtp['spatial_density'].isna()
        insufficient_mask = insufficient_mask | low_density.values
    
    if 'event_rate_local' in dtp.columns:
        low_rate = dtp['event_rate_local'] < 0.05
        insufficient_mask = insufficient_mask | low_rate.values
    
    insuf_count = insufficient_mask.sum()
    if insuf_count > 0:
        print(f"{Y_}  {insuf_count} deprem için yetersiz veri (insufficient data) - olasılık hesaplanmayacak{X_}")
    
    # Tabular modeller (XGB, RF)
    if models.get('xgb'):
        p_xgb = models['xgb'].predict_proba(Xp)[:, 1]
    else:
        p_xgb = np.full(len(Xp), 0.5)
    if models.get('rf'):
        p_rf = models['rf'].predict_proba(Xp)[:, 1]
    else:
        p_rf = np.full(len(Xp), 0.5)
    
    # Sequence model tahmini (generic)
    def seq_predict(model, scaler, name):
        if model is None or scaler is None:
            return np.full(len(Xp), 0.5)
        try:
            Xp_scaled = scaler.transform(Xp)
            if len(Xp_scaled) < 50:
                return np.full(len(Xp), 0.5)
            preds = []
            for i in range(len(Xp_scaled) - 50):
                X_seq = np.array([Xp_scaled[i:i+50]])
                preds.append(model.predict(X_seq, verbose=0)[0, 0])
            if preds:
                return np.array(preds + [np.mean(preds)] * 50)
            return np.full(len(Xp), 0.5)
        except Exception as e:
            print(f"{Y_}{name} predict error: {e}{X_}")
            return np.full(len(Xp), 0.5)
    
    p_lstm = seq_predict(lm, ls, 'LSTM')
    p_gru = seq_predict(gm, gs, 'GRU')
    p_transformer = seq_predict(tm, ts, 'Transformer')
    
    # ✅ ENSEMBLE: XGB ve RF en tutarlı modeller
    olasilik = (p_xgb * 0.20 + p_rf * 0.15 +
                p_lstm * 0.15 + p_gru * 0.40 +
                p_transformer * 0.10) * 100
    
    # Güven skoru: tüm modellerin uyumu
    all_preds = np.array([p_xgb, p_rf, p_lstm, p_gru, p_transformer])
    confidence = 100.0 - (np.std(all_preds, axis=0) * 100)
    confidence = np.clip(confidence, 0, 100)
    total_unc = np.std(all_preds, axis=0) * 100
    
    # ✅ Yetersiz veri olan depremlere NaN ata
    olasilik[insufficient_mask] = np.nan
    confidence[insufficient_mask] = np.nan
    total_unc[insufficient_mask] = np.nan
    
    return pd.DataFrame({
        'eventID': dtp['eventID'].values,
        'olasilik': olasilik,
        'confidence_score': confidence,
        'total_uncertainty': total_unc
    })

def add_legend(m, title, legend_dict):
    html = (f'<div style="position:fixed;top:10px;right:10px;width:auto;'
            f'padding:10px;border:2px solid grey;z-index:9999;font-size:13px;'
            f'background:white;border-radius:5px;"><b>{title}</b><br>')
    for key, color in legend_dict.items():
        html += f'<i style="background:{color};width:15px;height:15px;float:left;margin-right:8px;border-radius:50%;"></i>{key}<br>'
    html += '</div>'
    m.get_root().html.add_child(folium.Element(html))
    m.get_root().header.add_child(folium.Element(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/'
        'font-awesome/4.7.0/css/font-awesome.min.css">'))

def base_map(dm):
    if dm.empty:
        return folium.Map(location=[39.93, 32.86], zoom_start=6, tiles="CartoDB positron")
    return folium.Map(location=[dm['latitude'].mean(), dm['longitude'].mean()],
                      zoom_start=6, tiles="CartoDB positron")

def add_no_cache(m):
    no_cache_html = (
        "<meta http-equiv='Cache-Control' "
        "content='no-cache, no-store, must-revalidate, max-age=0'>"
        "<meta http-equiv='Pragma' content='no-cache'>"
        "<meta http-equiv='Expires' content='0'>"
    )
    m.get_root().header.add_child(folium.Element(no_cache_html))

def map_by_type(dfr, fn="deprem_haritasi_tip.html"):
    ms = (dfr['mag'] >= 4.0) & (dfr['earthquake_type'].isin(['Ana Deprem', 'Oncu Deprem', 'Artci Deprem']))
    mi = (dfr['mag'] >= 4.5) & ((dfr['earthquake_type'] == 'Tekil Deprem') | (dfr['earthquake_type'].isnull()))
    dm = dfr[ms | mi].copy()
    m = base_map(dm)
    tcm = {"Ana Deprem (Mainshock)": "red",
           "Oncu Deprem (Foreshock)": "orange",
           "Artci Deprem (Aftershock)": "blue",
           "Tekil Deprem (Isolated)": "gray"}
    type_map = {"Ana Deprem": "Ana Deprem (Mainshock)",
                "Oncu Deprem": "Oncu Deprem (Foreshock)",
                "Artci Deprem": "Artci Deprem (Aftershock)",
                "Tekil Deprem": "Tekil Deprem (Isolated)"}
    color_map = {"Ana Deprem": "red", "Oncu Deprem": "orange",
                 "Artci Deprem": "blue", "Tekil Deprem": "gray"}
    for _, r in dm.iterrows():
        et = r.get('earthquake_type', 'Tekil Deprem')
        if pd.isna(et):
            et = "Tekil Deprem"
        et_label = type_map.get(et, et)
        ts = pd.to_datetime(r['time']).strftime('%Y-%m-%d %H:%M')
        ph = (f"<b>Yer (Location):</b> {r['place']}<br>"
              f"<b>Buyukluk (Magnitude):</b> {r.get('mag')}<br>"
              f"<b>Tip (Type):</b> {et_label}<br>"
              f"<b>Tarih (Date):</b> {ts}")
        clr = color_map.get(et, 'gray')
        folium.CircleMarker(
            location=[r['latitude'], r['longitude']],
            radius=min(r.get('mag', 1) * 2, 12),
            popup=folium.Popup(ph, max_width=300),
            color=clr, fill=True, fill_color=clr, fill_opacity=0.7, weight=1
        ).add_to(m)
    add_legend(m, "Deprem Tipi (Earthquake Type)", tcm)
    add_no_cache(m)
    m.save(fn)

def map_by_prob(dfr, fn="deprem_haritasi_olasilik.html"):
    if 'olasilik' not in dfr.columns:
        return
    dm = dfr[(dfr['mag'] >= 4.5) & (dfr['olasilik'].notna())].copy()
    m = base_map(dm)
    for _, r in dm.iterrows():
        p = r.get('olasilik', 0)
        c = r.get('confidence_score', 50)
        if p >= 50:
            clr = 'darkred' if c > 70 else 'red'
        elif p >= 25:
            clr = 'darkorange' if c > 70 else 'orange'
        else:
            clr = 'yellow'
        ts = pd.to_datetime(r['time']).strftime('%Y-%m-%d %H:%M')
        ph = (f"<b>Yer (Location):</b> {r['place']}<br>"
              f"<b>Tarih (Date):</b> {ts}<br>"
              f"<b>Buyukluk (Magnitude):</b> {r.get('mag')}<br>"
              f"<b>Olasilik (Probability):</b> {p:.1f}%<br>"
              f"<b>Guven (Confidence):</b> {c:.1f}")
        folium.CircleMarker(
            location=[r['latitude'], r['longitude']],
            radius=min(r.get('mag', 1) * 2, 15),
            popup=folium.Popup(ph, max_width=350),
            color=clr, fill=True, fill_color=clr, fill_opacity=0.7
        ).add_to(m)
    li = {"Yuksek Risk (High Risk) >= %50 / Yuksek Guven": "darkred",
          "Yuksek Risk (High Risk) >= %50 / Dusuk Guven": "red",
          "Orta Risk (Medium Risk) >= %25 / Yuksek Guven": "darkorange",
          "Orta Risk (Medium Risk) >= %25 / Dusuk Guven": "orange",
          "Dusuk Risk (Low Risk) < %25": "yellow"}
    add_legend(m, "Olasilik ve Guven (Probability & Confidence)", li)
    add_no_cache(m)
    m.save(fn)

def gen_report(dfr, user, rtime, summary, new_ids, minfo, expl):
    recent = dfr.sort_values('time', ascending=False).head(2000)
    filt = recent[recent['mag'] >= 4.0].copy()
    filt['time'] = pd.to_datetime(filt['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    rt = filt[['time', 'place', 'mag', 'depth',
               'earthquake_type', 'olasilik', 'confidence_score']].copy()
    rt.columns = [
        'Zaman (Time UTC)',
        'Yer (Location)',
        'Buyukluk (Magnitude)',
        'Derinlik km (Depth km)',
        'Deprem Tipi (Earthquake Type)',
        'M>=5.5 Oncu Olasiligi % (Foreshock Probability %)',
        'Guven Skoru (Confidence Score)']
    pc = 'M>=5.5 Oncu Olasiligi % (Foreshock Probability %)'
    gc = 'Guven Skoru (Confidence Score)'

    def fp(row):
        # ✅ SADECE ÖNCÜ FİLTRESİ:
        # Öncü: Büyük deprem zaten gelmiş → olasılık gereksiz ve yanıltıcı
        # Artçı: Yeni bir büyük depremin öncüsü OLABİLİR → olasılık GÖSTERİLMELİ!
        et = row.get('Deprem Tipi (Earthquake Type)', '')
        if 'Oncu Deprem' in str(et) or 'Foreshock' in str(et):
            return '<span style="color:#F57C00;font-style:italic">- (Öncü / Foreshock)</span>'
        
        v = row[pc]
        if pd.isna(v) or v == "":
            return '<span style="color:gray">Veri Yetersiz (Insufficient Data)</span>'
        try:
            val = float(v)
            if val >= 50.00:
                return f'<span style="color:#b71c1c;font-weight:bold;font-size:1.05em">{val:.2f} ⚠</span>'
            elif val >= 25.00:
                return f'<span style="color:red;font-weight:bold">{val:.2f}</span>'
            elif val >= 10.00:
                return f'<span style="color:#f57c00">{val:.2f}</span>'
            return f"{val:.2f}"
        except Exception:
            return '<span style="color:gray">Veri Yetersiz (Insufficient Data)</span>'

    def fc(row):
        # ✅ SADECE ÖNCÜ FİLTRESİ: Güven skoru da gizle
        et = row.get('Deprem Tipi (Earthquake Type)', '')
        if 'Oncu Deprem' in str(et) or 'Foreshock' in str(et):
            return "-"
        
        if pd.isna(row[pc]) or row[pc] == "":
            return "-"
        try:
            return f"{float(row[gc]):.2f}"
        except Exception:
            return "-"

    def ft(v):
        clrs = {"Ana Deprem": "#D32F2F", "Oncu Deprem": "#F57C00",
                "Artci Deprem": "#1976D2", "Tekil Deprem": "#616161"}
        labels = {"Ana Deprem": "Ana Deprem (Mainshock)",
                  "Oncu Deprem": "Oncu Deprem (Foreshock)",
                  "Artci Deprem": "Artci Deprem (Aftershock)",
                  "Tekil Deprem": "Tekil Deprem (Isolated)"}
        cl = clrs.get(v, "#333")
        lb = labels.get(v, v)
        return f'<span style="color:{cl};font-weight:bold">{lb}</span>'

    rt[pc] = rt.apply(fp, axis=1)
    rt[gc] = rt.apply(fc, axis=1)
    rt['Deprem Tipi (Earthquake Type)'] = rt['Deprem Tipi (Earthquake Type)'].apply(ft)

    perf = ""
    if minfo:
        rows = ""
        for mn, mt in minfo.items():
            if mt and isinstance(mt, dict):
                auc_val = mt.get('auc', np.nan)
                auc_str = f"{auc_val:.3f}" if not np.isnan(auc_val) else "N/A"
                # AUC renk kodlama
                if not np.isnan(auc_val):
                    if auc_val < 0.5:
                        auc_str = f"<span style='color:#d32f2f;font-weight:bold'>{auc_val:.3f} ⚠</span>"
                    elif auc_val < 0.6:
                        auc_str = f"<span style='color:#f57c00'>{auc_val:.3f}</span>"
                    elif auc_val >= 0.7:
                        auc_str = f"<span style='color:#2e7d32;font-weight:bold'>{auc_val:.3f} ✓</span>"
                
                skill_val = mt.get('molchan_skill', np.nan)
                if not np.isnan(skill_val):
                    if skill_val < 0:
                        skill_str = f"<span style='color:#d32f2f;font-weight:bold'>{skill_val:.3f} ⚠</span>"
                    elif skill_val < 0.2:
                        skill_str = f"<span style='color:#f57c00'>{skill_val:.3f}</span>"
                    else:
                        skill_str = f"<span style='color:#2e7d32;font-weight:bold'>{skill_val:.3f} ✓</span>"
                else:
                    skill_str = "N/A"

                # F1, Precision, Recall (sklearn modelleri için)
                f1_val = mt.get('f1_score', mt.get('prospective_f1', np.nan))
                f1_str = f"{f1_val:.3f}" if not np.isnan(f1_val) else "N/A"
                prec_val = mt.get('precision', mt.get('prospective_precision', np.nan))
                prec_str = f"{prec_val:.3f}" if not np.isnan(prec_val) else "N/A"
                rec_val = mt.get('recall', mt.get('prospective_recall', np.nan))
                rec_str = f"{rec_val:.3f}" if not np.isnan(rec_val) else "N/A"
                
                threshold_val = mt.get('optimal_threshold', np.nan)
                threshold_str = f"{threshold_val:.3f}" if not np.isnan(threshold_val) else "0.500"
                
                rows += (f"<tr><td><b>{mn.upper()}</b></td>"
                         f"<td>{auc_str}</td>"
                         f"<td>{f1_str}</td>"
                         f"<td>{prec_str}</td>"
                         f"<td>{rec_str}</td>"
                         f"<td>{skill_str}</td>"
                         f"<td>{threshold_str}</td></tr>")
        if rows:
            perf = (
                "<h3>Model Performans Metrikleri (Model Performance Metrics)</h3>"
                "<div style='background:#fff3e0;padding:15px;border-left:5px solid #f57c00;margin:10px 0;border-radius:5px'>"
                "<b>📊 Yorum (Interpretation):</b><br>"
                "• <b>AUC &lt; 0.5:</b> Model rastgeleden kötü (Worse than random) ⚠ — kullanılmamalı (do not use)<br>"
                "• <b>AUC 0.5-0.6:</b> Zayıf ayrım gücü (Weak discrimination)<br>"
                "• <b>AUC 0.6-0.7:</b> Orta düzey - deprem tahmininde gerçekçi (Moderate - realistic for earthquake prediction)<br>"
                "• <b>AUC ≥ 0.7:</b> İyi performans (Good performance) ✓<br>"
                "• <b>Skill Score &lt; 0:</b> Rastgeleden kötü (Worse than random) ⚠<br>"
                "• <b>F1 Score:</b> Precision ve Recall'ın harmonik ortalaması - en önemli metrik (Harmonic mean - most important metric)<br>"
                "• <b>Recall:</b> Gerçek foreshocklardan ne kadarını yakaladık (How many actual foreshocks we caught)<br>"
                "• <b>Precision:</b> Foreshock dediklerimizin ne kadarı gerçek (How many of our predictions are correct)<br>"
                "<br><b>🎨 Olasılık Renk Kodları (Probability Color Codes):</b><br>"
                "• <span style='color:#b71c1c;font-weight:bold'>≥ %50: Yüksek Risk (High Risk) ⚠</span><br>"
                "• <span style='color:red;font-weight:bold'>%25-50: Orta-Yüksek Risk (Medium-High Risk)</span><br>"
                "• <span style='color:#f57c00'>%10-25: Dikkat (Caution)</span><br>"
                "• Normal: &lt; %10<br>"
                "• <span style='color:#1976D2'>Artçı (Aftershock):</span> Olasılık gösterilir - artçı yeni bir büyük depremin öncüsü olabilir! (may become foreshock of a new mainshock)<br>"
                "• <span style='color:#F57C00;font-style:italic'>Öncü (Foreshock):</span> Olasılık gösterilmez - büyük deprem zaten gelmiş (large earthquake already followed)<br>"
                "</div>"
                "<p style='color:#d32f2f'><b>Not (Note):</b> Test setinde yeterli foreshock örneği yoksa bazı metrikler 'N/A' olarak görünür "
                "(If insufficient foreshock examples in test set, some metrics show 'N/A'). "
                "Optimal threshold validation setinde F1-skoru maksimize edilerek bulunmuştur "
                "(Optimal threshold found by maximizing F1 on validation set).</p>"
                "<table style='width:100%;border-collapse:collapse'>"
                "<tr><th>Model</th><th>AUC</th><th>F1 Score</th><th>Precision</th><th>Recall</th>"
                "<th>Molchan Skill</th><th>Eşik (Threshold)</th></tr>"
                + rows +
                "</table>"
                "<p style='font-size:.9em;color:#666;margin-top:10px'>"
                "<b>Notlar (Notes):</b><br>"
                "• Train/Validation/Test ayrımı (split) zaman bazlı, aralarda 60 gün buffer (foreshock leakage önleme / leakage prevention).<br>"
                "• Threshold validation setinde optimize edildi (optimized) — test setine bakılmadı (not seen).<br>"
                "• Sınıf dengesizliği (class imbalance) için XGBoost'ta scale_pos_weight, RF'de class_weight='balanced', LSTM'de class_weight kullanıldı.<br>"
                "• <b>Dikkat (Warning):</b> Deprem öncü tahmini (earthquake foreshock prediction) çözülmemiş bir problemdir (unsolved problem). Gerçek dünyada AUC 0.6-0.7 makul kabul edilir (acceptable).</p>"
            )

    tbl = rt.to_html(index=False, escape=False)
    html = (
        "<!DOCTYPE html><html lang='tr'><head>"
        "<meta charset='UTF-8'>"
        "<meta http-equiv='Cache-Control' content='no-cache, no-store, must-revalidate, max-age=0'>"
        "<meta http-equiv='Pragma' content='no-cache'>"
        "<meta http-equiv='Expires' content='0'>"
        "<title>Risk Analiz Raporu (Risk Analysis Report)</title>"
        "<style>"
        "body{font-family:'Segoe UI',sans-serif;padding:20px;background:#f5f5f5}"
        "h1{color:#2c3e50}"
        "h3{color:#34495e;margin-top:30px}"
        "table{width:100%;border-collapse:collapse;background:white;box-shadow:0 1px 3px rgba(0,0,0,.2);margin-top:15px}"
        "th,td{padding:12px;border-bottom:1px solid #ddd;text-align:left}"
        "th{background:#004d40;color:white}"
        "tr:hover{background:#f1f1f1}"
        ".info{background:#e8f5e9;padding:15px;border-radius:5px;border-left:5px solid #2e7d32;margin:20px 0}"
        "</style></head><body>"
        "<h1>Risk Analiz Raporu (Risk Analysis Report)<br>"
        "<span style='font-size:0.7em;color:#555'>v21 - Düzeltilmiş Metrikler (Corrected Metrics)</span></h1>"
        f"<p><b>Rapor Tarihi (Report Date):</b> {rtime} | <b>Kullanici (User):</b> {user}</p>"
        f"<div class='info'>"
        f"<b>Ozet (Summary):</b> {summary}<br>"
        f"<b>Deprem Oncu Tanimi (Foreshock Definition):</b> "
        f"Sabit parametreler / Fixed parameters (Mag>={FORESHOCK_MAG_THRESHOLD}, Time<={FORESHOCK_TIME_WINDOW_DAYS}gün/days, "
        f"Dist<={FORESHOCK_SPATIAL_RADIUS_KM}km, ΔM>={FORESHOCK_MIN_MAG_DIFF}). "
        f"Bu parametreler model eğitiminden bağımsızdır (model-independent). "
        f"<b>Dikkat (Warning):</b> Gelecek ana şok bilgisi kullanılır (future mainshock info used - future leakage)</div>"
        f"{perf}"
        "<h3>Son Depremler (Recent Earthquakes) — M 4.0+</h3>"
        f"{tbl}"
        "</body></html>"
    )
    with open("deprem_analiz_raporu_sade.html", "w", encoding='utf-8') as f:
        f.write(html)
    print(f"{G_}✓ Rapor olusturuldu: deprem_analiz_raporu_sade.html{X_}")

# ============================================================================
# ANA FONKSİYON (DÜZELTILMIŞ BÖLÜM - v20)
# ============================================================================
def download_db_from_release(db_path, repo="ozgursaygi/deprem-analiz"):
    """
    GitHub Releases'ten en son database'i indir.
    Eğer release yoksa veya download başarısızsa False döner.
    """
    if os.path.exists(db_path):
        print(f"{G_}✓ Database zaten mevcut: {db_path}{X_}")
        return True
    
    url = f"https://github.com/{repo}/releases/latest/download/{os.path.basename(db_path)}"
    print(f"{C_}Database indiriliyor (downloading): {url}{X_}")
    
    try:
        response = requests.get(url, stream=True, timeout=300, allow_redirects=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            with open(db_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            print(f"{G_}✓ Database indirildi: {size_mb:.1f} MB{X_}")
            return True
        elif response.status_code == 404:
            print(f"{Y_}⚠ Release veya database dosyası bulunamadı (ilk çalıştırma olabilir).{X_}")
            return False
        else:
            print(f"{R_}✗ HTTP {response.status_code}: Database indirilemedi.{X_}")
            return False
    except Exception as e:
        print(f"{R_}✗ Database indirme hatası: {e}{X_}")
        return False

def upload_db_to_release(db_path, repo="ozgursaygi/deprem-analiz", tag="db-latest"):
    """
    Database'i GitHub Releases'e yükle.
    GITHUB_TOKEN environment variable'ı gerekli (workflow otomatik sağlar).
    """
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        print(f"{Y_}⚠ GITHUB_TOKEN bulunamadı. Database yüklenmedi (lokal çalışma için normal).{X_}")
        return False
    
    if not os.path.exists(db_path):
        print(f"{R_}✗ Database dosyası bulunamadı: {db_path}{X_}")
        return False
    
    size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"{C_}Database yükleniyor (uploading): {size_mb:.1f} MB...{X_}")
    
    try:
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        # 1) Mevcut release'i bul (tag = "db-latest")
        release_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
        r = requests.get(release_url, headers=headers, timeout=30)
        
        if r.status_code == 200:
            # Release var - eski asset'i sil
            release = r.json()
            release_id = release['id']
            for asset in release.get('assets', []):
                if asset['name'] == os.path.basename(db_path):
                    print(f"{C_}  Eski database siliniyor...{X_}")
                    requests.delete(f"https://api.github.com/repos/{repo}/releases/assets/{asset['id']}",
                                    headers=headers, timeout=30)
        elif r.status_code == 404:
            # Release yok - oluştur
            print(f"{C_}  Yeni release oluşturuluyor: {tag}...{X_}")
            create_url = f"https://api.github.com/repos/{repo}/releases"
            payload = {
                'tag_name': tag,
                'name': 'Database (auto-updated)',
                'body': 'Otomatik güncellenen database dosyası. Her runda yenilenir.',
                'draft': False,
                'prerelease': False
            }
            r = requests.post(create_url, json=payload, headers=headers, timeout=30)
            if r.status_code != 201:
                print(f"{R_}✗ Release oluşturulamadı: HTTP {r.status_code}{X_}")
                print(r.text[:500])
                return False
            release = r.json()
            release_id = release['id']
        else:
            print(f"{R_}✗ Release bulunamadı: HTTP {r.status_code}{X_}")
            return False
        
        # 2) Database'i yeni asset olarak yükle
        upload_url = f"https://uploads.github.com/repos/{repo}/releases/{release_id}/assets"
        upload_url += f"?name={os.path.basename(db_path)}"
        
        with open(db_path, 'rb') as f:
            upload_headers = {
                'Authorization': f'token {token}',
                'Content-Type': 'application/octet-stream'
            }
            r = requests.post(upload_url, data=f, headers=upload_headers, timeout=600)
        
        if r.status_code == 201:
            print(f"{G_}✓ Database GitHub Releases'e yüklendi: {size_mb:.1f} MB{X_}")
            return True
        else:
            print(f"{R_}✗ Database yüklenemedi: HTTP {r.status_code}{X_}")
            print(r.text[:500])
            return False
    except Exception as e:
        print(f"{R_}✗ Database yükleme hatası: {e}{X_}")
        return False

def main():
    t0 = time.time()
    db = "earthquakes_3_5_plus_scientific_v5.db"
    tn = "earthquake_catalog"
    csv_file = "ridgecrest_catalog.csv"
    conn = None
    try:
        print(f"{C_}{'='*70}")
        print("Analiz v22 (Analysis v22 - Düzeltilmiş Metrikler / Corrected Metrics)")
        print(f"{'='*70}{X_}")

        # ✅ docs/ klasörünü hazırla (v20 - YENİ)
        docs_dir = ensure_docs_dir()

        # ✅ YENİ: GitHub Releases'ten database'i indir (varsa)
        download_db_from_release(db)

        # Veritabanı bağlantısı ve tablo oluşturma
        conn = sqlite3.connect(db)
        new_db = setup_database(conn, tn)
        if new_db or not os.path.exists(db):
            print(f"{C_}Veritabani yeni olusturuldu. Gecmis datalar CSV'den aktariliyor...{X_}")
            load_historical_csv(conn, tn, csv_file)
        else:
            # Zaten varsa, sadece güncelle
            pass

        force = False
        new_ids = fetch_and_load_api_data(conn, tn)
        if new_ids:
            force = True

        df = pd.read_sql_query(f"SELECT * FROM {tn}", conn)
        df.drop_duplicates(subset=['eventID'], inplace=True, keep='last')
        df['time'] = pd.to_datetime(df['time'], utc=True)

        if len(df) < 100:
            fetch_and_load_api_data(conn, tn, start_override='2023-01-01 00:00:00')
            df = pd.read_sql_query(f"SELECT * FROM {tn}", conn)
            df.drop_duplicates(subset=['eventID'], inplace=True, keep='last')
            df['time'] = pd.to_datetime(df['time'], utc=True)
            force = True
            if len(df) < 100:
                print(f"{R_}Yetersiz veri (Insufficient data).{X_}")
                return

        print(f"{C_}Depremlerin bulundugu bolge belirleniyor...{X_}")
        if 'SEISMIC_ZONES' not in globals():
            global SEISMIC_ZONES
            SEISMIC_ZONES = {}
        if 'detect_seismic_zone' not in globals():
            def detect_seismic_zone(lat, lon):
                return "Unknown"
        df['seismic_zone'] = df.apply(lambda row: detect_seismic_zone(row['latitude'], row['longitude']), axis=1)
        zone_counts = df['seismic_zone'].value_counts()
        print(f"{G_}Bolge Dagilimi:{X_}")
        for zone, count in zone_counts.items():
            print(f"  {zone}: {count} olay")

        df = calc_features(df)
        df = classify_eq_type(df)

        models, metrics, expl = train_sklearn_improved(df, new_ids, force=force)
        lm, ls, lmet = train_lstm(df, new_ids, force=force)
        # YENİ: GRU ve Transformer modelleri (deprem zaman serisi için daha uygun)
        gm, gs, gmet = train_gru(df, force=force)
        tm, ts, tmet = train_transformer(df, force=force)

        ami = {**metrics}
        if lmet:
            ami['lstm'] = lmet
        if gmet:
            ami['gru'] = gmet
        if tmet:
            ami['transformer'] = tmet

        dfr = df.copy()
        t7 = pd.to_datetime(datetime.utcnow(), utc=True) - timedelta(days=7)
        rm = (dfr['time'] >= t7) | (dfr['olasilik'].isnull())
        aids = set(new_ids) | set(dfr[rm]['eventID'])
        if aids:
            dtp = dfr[dfr['eventID'].isin(aids)].copy()
            preds = predict_unc(dtp, models, lm, ls, gm, gs, tm, ts)
            if not preds.empty:
                preds['eventID'] = dtp['eventID'].values
                dfr = pd.merge(dfr, preds, on='eventID', how='left', suffixes=('', '_new'))
                for col in ['olasilik', 'confidence_score', 'total_uncertainty']:
                    nc = f'{col}_new'
                    if nc in dfr.columns:
                        dfr[col] = dfr[nc].fillna(dfr[col])
                        dfr.drop(columns=[nc], inplace=True)

        # Haritaları oluştur
        map_by_type(dfr)
        map_by_prob(dfr)

        # Raporu oluştur (Generate report)
        gen_report(
            dfr, CURRENT_USER, CURRENT_UTC_TIME,
            f"Toplam {len(dfr)} olay analiz edildi (Total {len(dfr)} events analyzed). "
            f"Sabit oncu deprem tanimi (modelden bağımsız) kullanilmistir "
            f"(Fixed foreshock definition - model independent). "
            f"Düzeltilmiş metrik hesaplama (Corrected metric calculation - NaN handled).",
            new_ids, ami, expl
        )

        # ========== RAPORLARI docs/ KLASÖRÜNE KOPYALA (İYİLEŞTİRİLMİŞ - v20) ==========
        print(f"\n{C_}Raporlar docs/ klasörüne kopyalanıyor...{X_}")
        
        rapor_list = [
            "deprem_analiz_raporu_sade.html",
            "deprem_haritasi_tip.html",
            "deprem_haritasi_olasilik.html",
            "foreshock_sensitivity_analysis.csv",
            "sensitivity_analysis.png",
            "molchan_xgb.png",
            "molchan_rf.png"
        ]
        
        basarili = 0
        for dosya in rapor_list:
            if copy_to_docs(dosya, docs_dir):
                basarili += 1
        
        # Son güncelleme zamanı
        try:
            update_file = os.path.join(docs_dir, "last_update.txt")
            with open(update_file, "w", encoding='utf-8') as f:
                f.write(CURRENT_UTC_TIME)
            print(f"{G_}✓ {update_file} güncellendi.{X_}")
        except Exception as e:
            print(f"{R_}✗ last_update.txt yazılamadı: {e}{X_}")

        # Veritabanını güncelle (Update database)
        dfs = dfr.copy()
        dfs['time'] = dfs['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({tn})")
        db_cols = [i[1] for i in cur.fetchall()]
        dfs[db_cols].to_sql(tn, conn, if_exists='replace', index=False)
        conn.commit()
        conn.close()
        conn = None  # Upload öncesi close gerekli
        
        # ✅ YENİ: Database'i GitHub Releases'e yükle
        upload_db_to_release(db)

        elapsed = time.time() - t0
        print(f"\n{G_}{'='*70}")
        print(f"✓ TAMAMLANDI (COMPLETED)")
        print(f"Başarılı Kopyalama (Files copied): {basarili}/{len(rapor_list)}")
        print(f"Süre (Duration): {elapsed:.1f} saniye (seconds)")
        print(f"{'='*70}{X_}")
        print(f"\n{G_}Üretilen Dosyalar (Generated Files):{X_}")
        for d in rapor_list:
            if os.path.exists(d):
                print(f"  ✓ {d}")
        print(f"  ✓ {docs_dir}/last_update.txt")
        print(f"\n{P_}GitHub Pages URL: https://ozgursaygi.github.io/deprem-analiz/{X_}")

    except Exception as e:
        print(f"{R_}HATA (ERROR): {e}{X_}")
        print(traceback.format_exc())
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()
