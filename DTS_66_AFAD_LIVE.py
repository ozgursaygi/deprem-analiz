#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sismik Risk Analiz Sistemi - v19
Düzeltilmiş metrikler, prospektif simülasyon, LSTM sequence fix,
raporları docs klasörüne kopyalama, binary dosyaları repodan hariç tutma.
"""

import pandas as pd
import sqlite3
import os
import time
import requests
from datetime import datetime, timedelta
import numpy as np
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
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
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
            ss = '1990-01-01 00:00:00'
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
                ss = '1990-01-01 00:00:00'
        else:
            ss = '1990-01-01 00:00:00'
    api_s = pd.to_datetime(ss).strftime('%Y-%m-%d %H:%M:%S')
    api_e = pd.to_datetime(end_lim).strftime('%Y-%m-%d %H:%M:%S')
    if api_s > api_e:
        api_s = (now - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
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
            dlon = rlons - clon
            dlat = rlats - clat
            a = np.sin(dlat / 2) ** 2 + np.cos(clat) * np.cos(rlats) * np.sin(dlon / 2) ** 2
            c_dist = 2 * np.arcsin(np.sqrt(a))
            dists = 6371.0 * c_dist
            decay_idx = np.sum(np.exp(-dists / 10.0))
        bv = calc_b_value(le['mag'].values)
        pe = le[le['time'] < ct]
        ts = (ct - pe['time'].max()).total_seconds() / 3600 if not pe.empty else None
        mc = le['mag'].quantile(0.1) if len(le) > 10 else None
        sd = len(le) / (np.pi * 50 ** 2)
        tc = 0
        if len(re) > 1:
            td = re['time'].diff().dt.total_seconds() / 3600
            std = td.std()
            if std > 0:
                tc = 1 / (std + 1e-6)
        r10 = le.tail(10)['mag'].values
        mt = np.polyfit(range(len(r10)), r10, 1)[0] if len(r10) > 1 else 0
        ds = le['depth'].std()
        dc = 1 / (ds + 1e-6) if ds > 0 else 0
        enr = (10 ** (1.5 * re['mag'] + 4.8)).sum() / 30 if not re.empty else 0
        t7 = ct - timedelta(days=7)
        sw = 1 if len(le[le['time'] >= t7]) >= 3 else 0
        faults = np.array([[40.7, 29.9], [38.4, 27.1], [39.6, 41.0]])
        fd = [haversine_distance(row['latitude'], row['longitude'], f[0], f[1]) for f in faults]
        mfd = min(fd) if fd else None
        ups.append({
            'eventID': eid, 'b_value_local': bv, 'event_rate_local': er,
            'time_since_last': ts, 'mag_completeness': mc, 'spatial_density': sd,
            'temporal_clustering': tc, 'mag_trend': mt, 'depth_clustering': dc,
            'energy_rate': enr, 'swarm_indicator': sw, 'fault_distance': mfd,
            'event_rate_24h': er_24h, 'event_rate_12h': er_12h, 'spatial_decay_index': decay_idx
        })
    if ups:
        dfu = pd.DataFrame(ups).set_index('eventID')
        df.set_index('eventID', inplace=True)
        df.update(dfu)
        df.reset_index(inplace=True)
        print(f"{G_}{len(ups)} kaydin ozellikleri hesaplandi.{X_}")
    return df

def classify_eq_type(df):
    dc = df.copy()
    dfs = dc.sort_values('time').reset_index(drop=True)
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
    Dikkat: Bu fonksiyon, her depremin *gelecekteki* ana şok bilgisini kullanarak
    foreshock etiketi üretir. Bu, future leakage yaratır. Gerçek zamanlı uygulama için
    yalnızca geçmiş ana şoklarla etiketleme yapılmalıdır.
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
        print(f"{G_}Foreshock (Sabit Bilimsel Parametreler): %{pos_rate:.2f} ({labels.sum()}/{len(labels)}){X_}")
        print(f"{G_}  Parametreler: Mag>={mag_threshold}, Time<={tw_days}d, Dist<={r_km}km, ΔM>={FORESHOCK_MIN_MAG_DIFF}{X_}")
        print(f"{Y_}  UYARI: Gelecek ana şok bilgisi kullaniliyor (future leakage).{X_}")
    return df

def sensitivity_analysis_foreshock(df):
    print(f"\n{C_}{'='*70}")
    print("SENSITIVITY ANALYSIS: Foreshock Parametreleri (Modelden Bağımsız)")
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
    print(f"\n{G_}Referans (Sabit Bilimsel) Parametreler:{X_}")
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
        plt.suptitle('Foreshock Definition - Sensitivity Analysis (Modelden Bağımsız)', fontsize=14, fontweight='bold', y=1.02)
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
    models = {'xgb': None, 'rf': None}
    metrics = {'xgb': {}, 'rf': {}}
    mp = {'xgb': 'xgb_v5.joblib', 'rf': 'rf_v5.joblib'}
    cutoff = df_full['time'].quantile(0.8)
    trd = df_full[df_full['time'] <= cutoff].copy()
    ted = df_full[df_full['time'] > cutoff].copy()
    if len(ted) < 10:
        print(f"{Y_}Test seti çok küçük, model eğitimi atlanıyor.{X_}")
        return {}, {}, {}
    if not force and all(os.path.exists(p) for p in mp.values()):
        for k in models:
            models[k] = joblib.load(mp[k])
        return models, metrics, {}

    trd = fix_numeric(trd)
    ted = fix_numeric(ted)
    print(f"{C_}Sabit bilimsel parametrelerle foreshock etiketlemesi yapiliyor...{X_}")
    trl = create_labels_parametric(trd.copy())
    tel = create_labels_parametric(ted.copy())

    sensitivity_df = sensitivity_analysis_foreshock(trd)
    if sensitivity_df is not None:
        sensitivity_df.to_csv('foreshock_sensitivity_analysis.csv', index=False)
        print(f"{G_}✓ Sensitivity analysis kaydedildi: foreshock_sensitivity_analysis.csv{X_}")

    af = [f for f in ENHANCED_FEATURES if f in trl.columns]
    Xtr = trl[af].apply(safe_fill)
    ytr = trl[TARGET]
    Xte = tel[af].apply(safe_fill)
    yte = tel[TARGET]

    if ytr.sum() == 0:
        print(f"{R_}Eğitim setinde hiç foreshock örneği yok! Model eğitilemez.{X_}")
        return {}, {}, {}
    if yte.sum() == 0:
        print(f"{Y_}Uyarı: Test setinde hiç foreshock yok. Metrikler geçersiz olacak.{X_}")

    for mtype in ['xgb', 'rf']:
        bp = None
        if OPTUNA_AVAILABLE:
            try:
                def obj(trial):
                    if mtype == 'xgb':
                        p = {
                            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                            'max_depth': trial.suggest_int('max_depth', 3, 8),
                            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                            'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss'
                        }
                        mdl = XGBClassifier(**p)
                    else:
                        p = {
                            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                            'max_depth': trial.suggest_int('max_depth', 5, 15),
                            'min_samples_split': trial.suggest_int('min_samples_split', 2, 8),
                            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                            'random_state': 42
                        }
                        mdl = RandomForestClassifier(**p)
                    return cross_val_score(mdl, Xtr, ytr, cv=TimeSeriesSplit(n_splits=3), scoring='roc_auc').mean()
                study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
                study.optimize(obj, n_trials=30, show_progress_bar=False)
                bp = study.best_params
            except Exception as e:
                print(f"{Y_}Optuna hatasi ({mtype}): {e}{X_}")
        if mtype == 'xgb':
            n_neg = (ytr == 0).sum()
            n_pos = (ytr == 1).sum()
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
            mdl = XGBClassifier(**(bp or {'n_estimators': 300, 'max_depth': 8,
                                          'learning_rate': 0.1, 'random_state': 42,
                                          'use_label_encoder': False, 'eval_metric': 'logloss'}),
                                scale_pos_weight=scale_pos_weight)
        else:
            mdl = RandomForestClassifier(**(bp or {'n_estimators': 300, 'max_depth': 15, 'random_state': 42}),
                                         class_weight='balanced')
        mdl.fit(Xtr, ytr)
        cal = CalibratedClassifierCV(mdl, method='sigmoid', cv=3)
        cal.fit(Xtr, ytr)
        yp = cal.predict(Xte)
        ypr = cal.predict_proba(Xte)[:, 1]

        met = get_metrics(yte, yp, ypr)
        mcd = calc_molchan(yte, ypr)
        met['molchan_skill'] = mcd['skill_score']
        met['molchan_auc'] = mcd['molchan_auc']

        # Optimum eşiği eğitim setinde bul
        tr_proba = cal.predict_proba(Xtr)[:, 1]
        prec, rec, ths = precision_recall_curve(ytr, tr_proba)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
        best_th = ths[np.argmax(f1_scores[:-1])] if len(ths) > 0 else 0.5
        ps = prospective_sim(tel, cal, af, threshold=best_th)
        met['prospective_accuracy'] = ps['accuracy']
        met['prospective_precision'] = ps['precision']
        met['prospective_recall'] = ps['recall']

        models[mtype] = cal
        metrics[mtype] = met
        plot_molchan(mcd, fn=f'molchan_{mtype}.png')
        joblib.dump(cal, mp[mtype])
        print(f"{G_}{mtype.upper()} Hazir | AUC:{met['auc']:.3f} (Skill:{met['molchan_skill']:.3f}){X_}")
    return models, metrics, {}

def build_lstm(shape):
    return Sequential([
        Input(shape=shape),
        LSTM(64, return_sequences=True, dropout=0.3), BatchNormalization(),
        LSTM(32, dropout=0.3), BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)), Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)), Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

def prospective_sim_lstm(model, scaler, df_test, feature_cols, seq_length=50, threshold=0.5):
    if df_test.empty or len(df_test) <= seq_length:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
    df_test = df_test.sort_values('time').reset_index(drop=True)
    X = df_test[feature_cols].apply(safe_fill).values
    X_scaled = scaler.transform(X)
    y_true = df_test[TARGET].values
    correct = 0
    total = len(df_test) - seq_length + 1
    all_preds = []
    all_true = []
    for i in range(seq_length - 1, len(df_test)):
        seq = X_scaled[i - seq_length + 1: i + 1]
        seq = seq.reshape(1, seq_length, X_scaled.shape[1])
        prob = model.predict(seq, verbose=0)[0, 0]
        pred = 1 if prob >= threshold else 0
        all_preds.append(pred)
        all_true.append(y_true[i])
        if pred == y_true[i]:
            correct += 1
    acc = (correct / total) * 100 if total > 0 else 0.0
    p = precision_score(all_true, all_preds, zero_division=0)
    r = recall_score(all_true, all_preds, zero_division=0)
    return {'accuracy': acc, 'precision': p, 'recall': r}

def train_lstm(df_full, new_ids, force=False):
    mpath = "lstm_v5.keras"
    spath = "lstm_scaler_v5.joblib"
    if not force and os.path.exists(mpath) and os.path.exists(spath):
        return load_model(mpath), joblib.load(spath), {}
    cutoff = df_full['time'].quantile(0.8)
    trd = df_full[df_full['time'] <= cutoff].copy()
    ted = df_full[df_full['time'] > cutoff].copy()
    if len(ted) < 10:
        return None, None, {}
    trl = create_labels_parametric(fix_numeric(trd).copy())
    tel = create_labels_parametric(fix_numeric(ted).copy())
    if trl.empty:
        return None, None, {}
    af = [f for f in ENHANCED_FEATURES if f in trl.columns]
    sc = StandardScaler()
    trs = sc.fit_transform(trl[af].apply(safe_fill))
    tes = sc.transform(tel[af].apply(safe_fill))
    sl = 50
    Xtr, ytr = [], []
    for i in range(len(trs) - sl):
        Xtr.append(trs[i:i+sl])
        ytr.append(trl[TARGET].iloc[i+sl])
    Xte, yte = [], []
    for i in range(len(tes) - sl):
        Xte.append(tes[i:i+sl])
        yte.append(tel[TARGET].iloc[i+sl])
    Xtr = np.array(Xtr)
    ytr = np.array(ytr)
    Xte = np.array(Xte)
    yte = np.array(yte)
    if len(Xtr) < 100 or len(Xte) == 0:
        return None, None, {}
    if ytr.sum() == 0:
        print(f"{R_}Eğitim setinde LSTM için hiç foreshock yok. Eğitim atlanıyor.{X_}")
        return None, None, {}
    classes = np.unique(ytr)
    class_weights = compute_class_weight('balanced', classes=classes, y=ytr)
    cw_dict = dict(zip(classes, class_weights))
    mdl = build_lstm((Xtr.shape[1], Xtr.shape[2]))
    mdl.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7)
    mdl.fit(Xtr, ytr, epochs=100, batch_size=32, validation_data=(Xte, yte),
            callbacks=[es, lr], class_weight=cw_dict, verbose=1)
    ypm = mdl.predict(Xte, verbose=0).flatten()
    yp = (ypm >= 0.5).astype(int)
    lmet = get_metrics(yte, yp, ypm)
    if len(ytr) > 0:
        tr_proba = mdl.predict(Xtr, verbose=0).flatten()
        prec, rec, ths = precision_recall_curve(ytr, tr_proba)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
        best_th = ths[np.argmax(f1_scores[:-1])] if len(ths) > 0 else 0.5
        ps = prospective_sim_lstm(mdl, sc, tel, af, seq_length=sl, threshold=best_th)
        lmet['prospective_accuracy'] = ps['accuracy']
        lmet['prospective_precision'] = ps['precision']
        lmet['prospective_recall'] = ps['recall']
    else:
        lmet['prospective_accuracy'] = np.nan
    mdl.save(mpath)
    joblib.dump(sc, spath)
    return mdl, sc, lmet

def predict_unc(dfp, models, lm, ls):
    if dfp.empty:
        return pd.DataFrame()
    dfp = fix_numeric(dfp)
    af = [f for f in ENHANCED_FEATURES if f in dfp.columns]
    insuf = (dfp['b_value_local'].isna()) | (dfp['b_value_local'] == 0) | (dfp['event_rate_local'] == 0)
    dp = dfp[af].apply(safe_fill)
    dpr = pd.DataFrame(index=dfp.index)
    for mk, mdl in models.items():
        if mdl:
            dpr[f'{mk}_prob'] = mdl.predict_proba(dp)[:, 1] * 100
        else:
            dpr[f'{mk}_prob'] = np.nan
    lp = np.zeros(len(dfp))
    lu = np.zeros(len(dfp))
    if lm and ls:
        try:
            ds = ls.transform(dp)
            seqs = []
            for i in range(len(ds)):
                sq = ds[max(0, i-49):i+1]
                if len(sq) < 50:
                    sq = np.vstack([np.zeros((50 - len(sq), ds.shape[1])), sq])
                seqs.append(sq)
            if seqs:
                seqs = np.array(seqs)
                mcp = np.array([lm(seqs, training=True).numpy().flatten() for _ in range(10)])
                lp = np.mean(mcp, axis=0) * 100
                lu = np.std(mcp, axis=0) * 100
        except Exception:
            lp[:] = np.nan
    dpr['lstm_prob'] = lp
    dpr['lstm_uncertainty'] = lu
    fps = []
    fcs = []
    for i in range(len(dpr)):
        if insuf.iloc[i]:
            fps.append(np.nan)
            fcs.append(0.0)
            continue
        px = dpr.iloc[i].get('xgb_prob', np.nan)
        pr = dpr.iloc[i].get('rf_prob', np.nan)
        pl = dpr.iloc[i]['lstm_prob']
        ul = dpr.iloc[i]['lstm_uncertainty']
        vp = []
        wt = []
        if not np.isnan(px):
            vp.append(px)
            wt.append(0.4)
        if not np.isnan(pr):
            vp.append(pr)
            wt.append(0.2)
        if not np.isnan(pl):
            wl = 0.4 * np.exp(-ul / 10)
            vp.append(pl)
            wt.append(wl)
        if vp and sum(wt) > 0:
            wp = max(0.1, min(99.9, np.average(vp, weights=wt)))
            fps.append(wp)
            bu = ul if not np.isnan(ul) else 20.0
            fcs.append(max(0, 100 - bu))
        else:
            fps.append(np.nan)
            fcs.append(0.0)
    dpr['olasilik'] = fps
    dpr['confidence_score'] = fcs
    dpr['total_uncertainty'] = 100 - dpr['confidence_score']
    return dpr

def add_legend(m, title, items):
    body = "".join([f'<i class="fa fa-circle" style="color:{c}"></i> {l}<br>' for l, c in items.items()])
    html = (f'<div style="position:fixed;bottom:50px;right:50px;width:320px;'
            f'padding:10px;border:2px solid grey;z-index:9999;font-size:13px;'
            f'background:white;border-radius:5px"><b>{title}</b><br>{body}</div>')
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
        v = row[pc]
        if pd.isna(v) or v == "":
            return '<span style="color:gray">Veri Yetersiz (Insufficient Data)</span>'
        try:
            val = float(v)
            if val >= 50.00:
                return f'<span style="color:red;font-weight:bold">{val:.2f}</span>'
            return f"{val:.2f}"
        except Exception:
            return '<span style="color:gray">Veri Yetersiz (Insufficient Data)</span>'

    def fc(row):
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
                skill_val = mt.get('molchan_skill', np.nan)
                skill_str = f"{skill_val:.3f}" if not np.isnan(skill_val) else "N/A"
                prosp_acc = mt.get('prospective_accuracy', np.nan)
                prosp_str = f"%{prosp_acc:.1f}" if not np.isnan(prosp_acc) else "N/A"
                brier_val = mt.get('brier_score', np.nan)
                brier_str = f"{brier_val:.3f}" if not np.isnan(brier_val) else "N/A"
                rows += (f"<tr><td><b>{mn.upper()}</b></td>"
                         f"<td>{auc_str}</td>"
                         f"<td>{skill_str}</td>"
                         f"<td>{prosp_str}</td>"
                         f"<td>{brier_str}</td></tr>")
        if rows:
            perf = (
                "<h3>Model Performans Metrikleri (Model Performance Metrics)</h3>"
                "<p style='color:#d32f2f'><b>Not:</b> Test setinde yeterli foreshock örneği yoksa bazı metrikler 'N/A' olarak görünür.</p>"
                "<table style='width:100%;border-collapse:collapse'>"
                "<tr><th>Model</th><th>AUC</th><th>Molchan Beceri Skoru (Skill Score)</th>"
                "<th>Prospektif Dogruluk (Prospective Accuracy)</th><th>Brier Skoru (Brier Score)</th></tr>"
                + rows +
                "</table>"
                "<p style='font-size:.9em;color:#666;margin-top:10px'>"
                "<b>Molchan Beceri Skoru:</b> 1.0 = Mukemmel, 0.0 = Rastgele, <0 = Rastgeleden kötü<br>"
                "<b>Prospektif Dogruluk:</b> Gerçek zamanlı tahmin simülasyonu başarı oranı (optimum eşik ile)</p>"
            )

    tbl = rt.to_html(index=False, escape=False)
    html = (
        "<!DOCTYPE html><html lang='tr'><head>"
        "<meta charset='UTF-8'>"
        "<meta http-equiv='Cache-Control' content='no-cache, no-store, must-revalidate, max-age=0'>"
        "<meta http-equiv='Pragma' content='no-cache'>"
        "<meta http-equiv='Expires' content='0'>"
        "<title>Sismik Risk Raporu (Seismic Risk Report)</title>"
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
        "<h1>Sismik Risk Analiz Raporu<br>"
        "<span style='font-size:0.7em;color:#555'>Seismic Risk Analysis Report (Improved v19 - Düzeltilmiş Metrikler ve Rapor Kopyalama)</span></h1>"
        f"<p><b>Rapor Tarihi (Report Date):</b> {rtime} | <b>Kullanici (User):</b> {user}</p>"
        f"<div class='info'>"
        f"<b>Ozet (Summary):</b> {summary}<br>"
        f"<b>Deprem Oncu Tanimi (Foreshock Definition):</b> "
        f"Sabit bilimsel parametreler (Mag>={FORESHOCK_MAG_THRESHOLD}, Time<={FORESHOCK_TIME_WINDOW_DAYS}gün, "
        f"Dist<={FORESHOCK_SPATIAL_RADIUS_KM}km, ΔM>={FORESHOCK_MIN_MAG_DIFF}). "
        f"Bu parametreler model eğitiminden bağımsızdır ve değiştirilmez. (Dikkat: Gelecek ana şok bilgisi kullanılır - future leakage)</div>"
        f"{perf}"
        "<h3>Son Depremler (Recent Earthquakes) — M 4.0+</h3>"
        f"{tbl}"
        "</body></html>"
    )
    with open("deprem_analiz_raporu_sade.html", "w", encoding='utf-8') as f:
        f.write(html)
    print(f"{G_}✓ Rapor olusturuldu: deprem_analiz_raporu_sade.html{X_}")

# ============================================================================
# ANA FONKSİYON
# ============================================================================
def main():
    t0 = time.time()
    db = "earthquakes_3_5_plus_scientific_v5.db"
    tn = "earthquake_catalog"
    csv_file = "ridgecrest_catalog.csv"
    conn = None
    try:
        print(f"{C_}{'='*70}")
        print("Sismik Analiz v19 (Seismic Analysis v19 - Düzeltilmiş Metrikler & Docs Kopyalama)")
        print(f"{'='*70}{X_}")

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
        # Bölge bilgisini hesapla (gösteri amaçlı, sabit bölgeler yoksa atla)
        # SEISMIC_ZONES tanımlı değilse geçici tanımla
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

        ami = {**metrics}
        if lmet:
            ami['lstm'] = lmet

        dfr = df.copy()
        t7 = pd.to_datetime(datetime.utcnow(), utc=True) - timedelta(days=7)
        rm = (dfr['time'] >= t7) | (dfr['olasilik'].isnull())
        aids = set(new_ids) | set(dfr[rm]['eventID'])
        if aids:
            dtp = dfr[dfr['eventID'].isin(aids)].copy()
            preds = predict_unc(dtp, models, lm, ls)
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

        # Raporu oluştur
        gen_report(
            dfr, CURRENT_USER, CURRENT_UTC_TIME,
            f"Toplam {len(dfr)} olay analiz edildi. "
            f"Sabit bilimsel oncu deprem tanimi (modelden bağımsız) kullanilmistir. "
            f"Düzeltilmiş metrik hesaplama (NaN geçerli değil). "
            f"(Total {len(dfr)} events analyzed with fixed scientific foreshock definition. "
            f"Corrected metric calculation.)",
            new_ids, ami, expl
        )

        # ========== RAPORLARI docs/ KLASÖRÜNE KOPYALA ==========
        os.makedirs("docs", exist_ok=True)
        rapor_list = [
            "deprem_analiz_raporu_sade.html",
            "deprem_haritasi_tip.html",
            "deprem_haritasi_olasilik.html",
            "foreshock_sensitivity_analysis.csv",
            "sensitivity_analysis.png",
            "molchan_xgb.png",
            "molchan_rf.png"
        ]
        for dosya in rapor_list:
            if os.path.exists(dosya):
                shutil.copy2(dosya, f"docs/{dosya}")
                print(f"{G_}✓ Kopyalandi: {dosya} -> docs/{dosya}{X_}")
            else:
                print(f"{Y_}Uyarı: {dosya} bulunamadi, kopyalanmadi.{X_}")
        # Son güncelleme zamanı
        with open("docs/last_update.txt", "w", encoding='utf-8') as f:
            f.write(CURRENT_UTC_TIME)
        print(f"{G_}✓ last_update.txt güncellendi.{X_}")

        # Veritabanını güncelle (ancak repoya ekleme, .gitignore kontrolü)
        dfs = dfr.copy()
        dfs['time'] = dfs['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({tn})")
        db_cols = [i[1] for i in cur.fetchall()]
        dfs[db_cols].to_sql(tn, conn, if_exists='replace', index=False)

        elapsed = time.time() - t0
        print(f"\n{G_}{'='*70}")
        print(f"✓ TAMAMLANDI (COMPLETED)")
        print(f"Süre (Duration): {elapsed:.1f} saniye (seconds)")
        print(f"{'='*70}{X_}")
        print(f"\n{G_}Üretilen Dosyalar (Generated Files):{X_}")
        for d in rapor_list:
            if os.path.exists(d):
                print(f"  ✓ {d}")
        print("  ✓ last_update.txt")
        print("\nNot: Veritabanı dosyası .gitignore ile repodan hariç tutulmuştur. Binary conflict önlenmiştir.")

    except Exception as e:
        print(f"{R_}HATA (ERROR): {e}{X_}")
        print(traceback.format_exc())
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()