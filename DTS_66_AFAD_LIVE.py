#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sqlite3
import os
import time
import requests
import json
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
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, brier_score_loss, roc_curve)
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
try:
    import optuna; OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
try:
    import shap; SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
import matplotlib.pyplot as plt

R_ = '\033[91m'; G_ = '\033[92m'; P_ = '\033[95m'
C_ = '\033[96m'; Y_ = '\033[93m'; B_ = '\033[94m'; X_ = '\033[0m'
CURRENT_UTC_TIME = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
CURRENT_USER = "ozgursaygi"
ENHANCED_FEATURES = [
    'mag','depth','b_value_local','event_rate_local','time_since_last',
    'mag_completeness','spatial_density','temporal_clustering',
    'mag_trend','depth_clustering','energy_rate','swarm_indicator',
    'fault_distance']
TARGET = 'is_foreshock'

@njit
def haversine_distance_numba(lon1, lat1, lon2, lat2):
    R=6371.0
    lon1=np.radians(lon1); lat1=np.radians(lat1)
    lon2=np.radians(lon2); lat2=np.radians(lat2)
    a=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return R*2*np.arcsin(np.sqrt(a))

def haversine_distance(lat1, lon1, lat2, lon2):
    return haversine_distance_numba(lon1, lat1, lon2, lat2)

def get_neighbors_cKDTree(df, radius_km):
    vc=df[['latitude','longitude']].dropna()
    if vc.empty: return None
    tree=cKDTree(np.deg2rad(vc.values))
    return tree.query_ball_tree(tree, r=radius_km/6371.0)

def standardize_date(dv):
    if pd.isna(dv) or dv=="": return None
    try:
        ds=str(dv).strip()
        if len(ds)<=10: ds+=" 00:00:00"
        dt=pd.to_datetime(ds,yearfirst=True,dayfirst=False,errors='coerce',utc=True)
        if pd.notna(dt): return dt.strftime('%Y-%m-%d %H:%M:%S')
        dt=pd.to_datetime(ds,dayfirst=True,errors='coerce',utc=True)
        if pd.notna(dt): return dt.strftime('%Y-%m-%d %H:%M:%S')
        return None
    except: return None

def fix_future_dates(conn, tn):
    now=datetime.utcnow()
    lim=pd.Timestamp((now+timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S'),tz='UTC')
    try:
        df=pd.read_sql(f"SELECT * FROM {tn}",conn)
        if df.empty: return
        df['time']=pd.to_datetime(df['time'],utc=True,errors='coerce')
        fm=df['time']>lim
        if fm.sum()==0: return
        print(f"{Y_}{fm.sum()} gelecek tarihli kayit bulundu.{X_}")
        drop=[]
        for idx in df[fm].index:
            wd=df.at[idx,'time']
            if pd.isna(wd): drop.append(idx); continue
            ok=False
            if wd.year>now.year+1:
                try:
                    nd=wd.replace(year=wd.year-100)
                    if nd<=lim: df.at[idx,'time']=nd; ok=True
                except ValueError: pass
            if not ok:
                try:
                    if wd.month!=wd.day:
                        sw=wd.replace(month=wd.day,day=wd.month)
                        if sw<=lim: df.at[idx,'time']=sw; ok=True
                except ValueError: pass
            if not ok and wd.year>now.year+1:
                try:
                    tmp=wd.replace(year=wd.year-100)
                    if tmp.month!=tmp.day:
                        sw2=tmp.replace(month=tmp.day,day=tmp.month)
                        if sw2<=lim: df.at[idx,'time']=sw2; ok=True
                except ValueError: pass
            if not ok: drop.append(idx)
        if drop: df.drop(index=drop,inplace=True)
        if (df['time']>lim).sum()>0: df=df[df['time']<=lim]
        df['time']=df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        conn.cursor().execute(f"DELETE FROM {tn}")
        df.to_sql(tn,conn,if_exists='append',index=False); conn.commit()
        print(f"{G_}Tarihler duzeltildi. Kayit:{len(df)}{X_}")
    except Exception as e:
        print(f"{R_}Tarih hatasi:{e}{X_}")

def setup_database(conn, tn):
    cur=conn.cursor()
    cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{tn}'")
    exists=cur.fetchone()
    cols={"time":"TEXT","latitude":"REAL","longitude":"REAL","depth":"REAL",
          "mag":"REAL","eventID":"TEXT","place":"TEXT","b_value_local":"REAL",
          "event_rate_local":"REAL","time_since_last":"REAL",
          "mag_completeness":"REAL","spatial_density":"REAL",
          "temporal_clustering":"REAL","mag_trend":"REAL",
          "depth_clustering":"REAL","energy_rate":"REAL",
          "swarm_indicator":"INTEGER","fault_distance":"REAL",
          "earthquake_type":"TEXT","is_foreshock":"INTEGER",
          "olasilik":"REAL","confidence_score":"REAL","total_uncertainty":"REAL"}
    created=False
    if not exists:
        cs=", ".join([f'"{k}" {v}' for k,v in cols.items()])
        cur.execute(f"CREATE TABLE {tn} ({cs});"); created=True
    else:
        cur.execute(f"PRAGMA table_info({tn});")
        ex={r[1] for r in cur.fetchall()}
        for c,ct in cols.items():
            if c not in ex: cur.execute(f"ALTER TABLE {tn} ADD COLUMN {c} {ct};")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_time ON {tn} (time);")
    conn.commit(); return created

def fetch_and_load_api_data(conn, tn, start_override=None):
    now=datetime.utcnow()
    end_lim=(now+timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
    cur=conn.cursor()
    fix_future_dates(conn,tn)
    if start_override:
        ss=standardize_date(start_override)
        if not ss: ss='1990-01-01 00:00:00'
    else:
        cur.execute(f"SELECT MAX(time) FROM {tn}")
        r=cur.fetchone(); latest=r[0] if r else None
        if latest:
            try:
                ldt=pd.to_datetime(latest,utc=True)
                if ldt>pd.Timestamp(end_lim,tz='UTC'):
                    ss=(now-timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    ss=(ldt+timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S')
            except: ss='1990-01-01 00:00:00'
        else: ss='1990-01-01 00:00:00'
    
    # SAAT EKLENEREK DÜZELTİLEN KISIM:
    api_s=pd.to_datetime(ss).strftime('%Y-%m-%d %H:%M:%S')
    api_e=pd.to_datetime(end_lim).strftime('%Y-%m-%d %H:%M:%S')
    if api_s>api_e:
        api_s=(now-timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        
    params={'start':api_s,'end':api_e,'orderby':'time-asc','minmag':'3.5'}
    for att in range(5):
        try:
            print(f"{C_}API ({att+1}/5)... {api_s} -> {api_e}{X_}")
            resp=requests.get("https://deprem.afad.gov.tr/apiv2/event/filter",
                              params=params,timeout=300)
            resp.raise_for_status(); data=resp.json()
            if not data: return []
            df=pd.DataFrame(data)
            df.rename(columns={'date':'time','magnitude':'mag','location':'place'},inplace=True)
            for c in ['latitude','longitude','depth','mag']:
                df[c]=pd.to_numeric(df[c],errors='coerce')
            df=df[df['mag']>=3.5]
            df['time']=df['time'].apply(standardize_date)
            df.dropna(subset=['time'],inplace=True)
            df['_tc']=pd.to_datetime(df['time'],utc=True,errors='coerce')
            fut=df['_tc']>pd.Timestamp(end_lim,tz='UTC')
            if fut.any(): df=df[~fut]
            df.drop(columns=['_tc'],inplace=True)
            cur.execute(f"SELECT eventID FROM {tn}")
            ex_ids={r[0] for r in cur.fetchall()}
            dfn=df[~df['eventID'].isin(ex_ids)]
            if dfn.empty: return []
            lc=['time','latitude','longitude','depth','mag','place','eventID']
            dfn[lc].to_sql(tn,conn,if_exists='append',index=False,chunksize=1000)
            print(f"{G_}{len(dfn)} yeni kayit eklendi.{X_}")
            return dfn['eventID'].tolist()
        except requests.exceptions.RequestException as e:
            print(f"{Y_}API Hatasi:{e}{X_}"); time.sleep(5)
    return []

def calc_b_value(mags):
    if len(mags)<20: return None
    try:
        mc=pd.Series(mags).value_counts().idxmax()
        cm=mags[mags>=mc]
        if len(cm)<10: return None
        b=np.log10(np.e)/(np.mean(cm)-mc+0.05)
        return b if 0.3<=b<=2.5 else None
    except: return None

def fix_numeric(df):
    for c in ENHANCED_FEATURES+['latitude','longitude']:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors='coerce')
    return df

def safe_fill(s):
    try:
        ns=pd.to_numeric(s,errors='coerce'); return ns.fillna(ns.median())
    except: return pd.to_numeric(s,errors='coerce').fillna(0)

def calc_features(df_all):
    df=df_all.copy(); df=fix_numeric(df)
    cids=set(df.loc[df['b_value_local'].isnull(),'eventID'])
    if not cids: return df_all
    print(f"{len(cids)} kayit hesaplaniyor...")
    dfs=df.sort_values('time').reset_index(drop=True)
    ni=get_neighbors_cKDTree(dfs,50)
    if ni is None: return df
    ups=[]
    try:
        from tqdm import tqdm; it=tqdm(range(len(dfs)),desc="Hesap")
    except: it=range(len(dfs))
    for si in it:
        row=dfs.loc[si]; eid=row['eventID']
        if eid not in cids: continue
        ct=row['time']
        pi=[i for i in ni[si] if dfs.loc[i,'time']<=ct]
        if not pi: continue
        le=dfs.iloc[pi]; t30=ct-timedelta(days=30)
        re=le[le['time']>=t30]; er=len(re)/30.0
        bv=calc_b_value(le['mag'].values)
        pe=le[le['time']<ct]
        ts=(ct-pe['time'].max()).total_seconds()/3600 if not pe.empty else None
        mc=le['mag'].quantile(0.1) if len(le)>10 else None
        sd=len(le)/(np.pi*50**2); tc=0
        if len(re)>1:
            td=re['time'].diff().dt.total_seconds()/3600
            std=td.std()
            if std>0: tc=1/(std+1e-6)
        r10=le.tail(10)['mag'].values
        mt=np.polyfit(range(len(r10)),r10,1)[0] if len(r10)>1 else 0
        ds=le['depth'].std(); dc=1/(ds+1e-6) if ds>0 else 0
        enr=(10**(1.5*re['mag']+4.8)).sum()/30 if not re.empty else 0
        t7=ct-timedelta(days=7)
        sw=1 if len(le[le['time']>=t7])>=3 else 0
        faults=np.array([[40.7,29.9],[38.4,27.1],[39.6,41.0]])
        fd=[haversine_distance(row['latitude'],row['longitude'],f[0],f[1]) for f in faults]
        mfd=min(fd) if fd else None
        ups.append({'eventID':eid,'b_value_local':bv,'event_rate_local':er,
            'time_since_last':ts,'mag_completeness':mc,'spatial_density':sd,
            'temporal_clustering':tc,'mag_trend':mt,'depth_clustering':dc,
            'energy_rate':enr,'swarm_indicator':sw,'fault_distance':mfd})
    if ups:
        dfu=pd.DataFrame(ups).set_index('eventID')
        df.set_index('eventID',inplace=True); df.update(dfu)
        df.reset_index(inplace=True)
        print(f"{G_}{len(ups)} kaydin ozellikleri hesaplandi.{X_}")
    return df

def classify_eq_type(df):
    dc=df.copy(); dfs=dc.sort_values('time').reset_index(drop=True)
    ni=get_neighbors_cKDTree(dfs,120)
    if ni is None: df['earthquake_type']="Tekil Deprem"; return df
    types=np.full(len(dfs),"Tekil Deprem",dtype=object)
    types[dfs['mag']>=6.0]="Ana Deprem"
    mags=dfs['mag'].values; times=dfs['time'].values
    tw=np.timedelta64(60,'D')
    for i in np.where(dfs['mag']<6.0)[0]:
        cm,ct=mags[i],times[i]; nt=times[ni[i]]; nm=mags[ni[i]]
        pl,fl=ct-tw,ct+tw
        wm=(nt>=pl)&(nt<=fl); pm=wm&(nt<ct); fm_=wm&(nt>ct)
        lpm=np.max(nm[pm]) if np.any(pm) else -1
        lfm=np.max(nm[fm_]) if np.any(fm_) else -1
        if cm<lpm-0.8: types[i]="Artci Deprem"
        elif cm<lfm-0.8: types[i]="Oncu Deprem"
    dfs['earthquake_type']=types
    tr=dfs[['eventID','earthquake_type']]
    if 'earthquake_type' in df.columns: df=df.drop(columns=['earthquake_type'])
    return pd.merge(df,tr,on='eventID',how='left')

def create_labels(df, thresh=5.5, tw_days=30, r_km=50):
    df=df.sort_values('time').reset_index(drop=True)
    twns=timedelta(days=tw_days).total_seconds()*1e9
    ni=get_neighbors_cKDTree(df,radius_km=r_km)
    if ni is None: df[TARGET]=0; return df
    labels=np.zeros(len(df),dtype=int)
    mags=df['mag'].values; tn_=df['time'].astype(np.int64).values
    for i in range(len(df)):
        fi=[idx for idx in ni[i] if idx>i]
        if not fi: continue
        td=tn_[fi]-tn_[i]; itm=td<=twns
        if np.any(itm):
            ri=np.array(fi)[itm]
            if np.max(mags[ri])>=thresh: labels[i]=1
    df[TARGET]=labels
    pr=(labels.sum()/len(labels))*100
    print(f"{G_}Pozitif: %{pr:.1f} ({labels.sum()}/{len(labels)}){X_}")
    return df

def get_metrics(yt, yp, ypr):
    m={'accuracy':accuracy_score(yt,yp),
       'precision':precision_score(yt,yp,zero_division=0),
       'recall':recall_score(yt,yp,zero_division=0),
       'f1_score':f1_score(yt,yp,zero_division=0),
       'auc':roc_auc_score(yt,ypr) if len(np.unique(yt))>1 else 0.5,
       'brier_score':brier_score_loss(yt,ypr),'ece':0}
    if len(yt)>0 and len(np.unique(yt))>1:
        pt,pp=calibration_curve(yt,ypr,n_bins=10,strategy='uniform')
        m['ece']=np.mean(np.abs(pt-pp))
    return m

def calc_molchan(yt, ypr):
    if len(np.unique(yt))<2:
        return {'skill_score':0,'molchan_auc':0.5,
                'miss_rate':np.array([0,1]),'alarm_rate':np.array([0,1]),
                'thresholds':np.array([0])}
    fpr,tpr,th=roc_curve(yt,ypr)
    mr=1-tpr; ar=fpr; ma=np.trapz(mr,ar); ss=1-(2*ma)
    return {'skill_score':ss,'molchan_auc':ma,'miss_rate':mr,
            'alarm_rate':ar,'thresholds':th}

def plot_molchan(md, fn='molchan.png'):
    try:
        plt.figure(figsize=(8,6))
        plt.plot(md['alarm_rate'],md['miss_rate'],'b-',lw=2,
                 label=f"Model (Skill={md['skill_score']:.3f})")
        plt.plot([0,1],[0,1],'r--',lw=1,label='Rastgele (Random)')
        plt.xlabel('Alarm Rate'); plt.ylabel('Miss Rate')
        plt.title('Molchan Diagram'); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(fn,dpi=150); plt.close()
    except: pass

def prospective_sim(df_test, model, fl):
    res=[]
    for i in range(len(df_test)):
        ev=df_test.iloc[i]; X=ev[fl].values.reshape(1,-1)
        pp=model.predict_proba(X)[0,1]; ao=ev[TARGET]
        res.append({'eventID':ev['eventID'],'time':ev['time'],'mag':ev['mag'],
                    'predicted_proba':pp,'actual_outcome':ao,
                    'correct':(pp>=0.5)==ao})
    return pd.DataFrame(res)

def optuna_opt(Xtr, ytr, mt='xgb', n_trials=30):
    if not OPTUNA_AVAILABLE: return None
    def obj(trial):
        if mt=='xgb':
            p={'n_estimators':trial.suggest_int('n_estimators',100,400),
               'max_depth':trial.suggest_int('max_depth',3,8),
               'learning_rate':trial.suggest_float('learning_rate',0.01,0.2),
               'subsample':trial.suggest_float('subsample',0.7,1.0),
               'colsample_bytree':trial.suggest_float('colsample_bytree',0.7,1.0),
               'random_state':42,'use_label_encoder':False,'eval_metric':'logloss'}
            mdl=XGBClassifier(**p)
        else:
            p={'n_estimators':trial.suggest_int('n_estimators',100,400),
               'max_depth':trial.suggest_int('max_depth',5,15),
               'min_samples_split':trial.suggest_int('min_samples_split',2,8),
               'min_samples_leaf':trial.suggest_int('min_samples_leaf',1,4),
               'class_weight':'balanced','random_state':42}
            mdl=RandomForestClassifier(**p)
        return cross_val_score(mdl,Xtr,ytr,cv=TimeSeriesSplit(n_splits=3),
                               scoring='roc_auc').mean()
    try:
        study=optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler())
        study.optimize(obj,n_trials=n_trials,show_progress_bar=False)
        return study.best_params
    except: return None

def train_sklearn(df_full, new_ids, force=False):
    models={'xgb':None,'rf':None}
    metrics={'xgb':{},'rf':{}}; expl={'xgb':None,'rf':None}
    mp={'xgb':'xgb_v5.joblib','rf':'rf_v5.joblib'}
    cutoff=df_full['time'].quantile(0.8)
    trd=df_full[df_full['time']<=cutoff].copy()
    ted=df_full[df_full['time']>cutoff].copy()
    if len(ted)<10: return {},{},{}
    if not force and all(os.path.exists(p) for p in mp.values()):
        for k in models: models[k]=joblib.load(mp[k])
        return models,metrics,expl
    trd=fix_numeric(trd); ted=fix_numeric(ted)
    trl=create_labels(trd.copy()); tel=create_labels(ted.copy())
    af=[f for f in ENHANCED_FEATURES if f in trl.columns]
    Xtr=trl[af].apply(safe_fill); ytr=trl[TARGET]
    Xte=tel[af].apply(safe_fill); yte=tel[TARGET]
    sw=compute_sample_weight('balanced',ytr)
    for mtype in ['xgb','rf']:
        bp=optuna_opt(Xtr,ytr,mtype,n_trials=30)
        if mtype=='xgb':
            mdl=XGBClassifier(**(bp or {'n_estimators':300,'max_depth':8,
                'learning_rate':0.1,'random_state':42,
                'use_label_encoder':False,'eval_metric':'logloss'}))
        else:
            mdl=RandomForestClassifier(**(bp or {'n_estimators':300,
                'max_depth':15,'class_weight':'balanced','random_state':42}))
        mdl.fit(Xtr,ytr,sample_weight=sw if mtype=='xgb' else None)
        cal=CalibratedClassifierCV(mdl,method='isotonic',cv=3)
        cal.fit(Xtr,ytr)
        yp=cal.predict(Xte); ypr=cal.predict_proba(Xte)[:,1]
        models[mtype]=cal; metrics[mtype]=get_metrics(yte,yp,ypr)
        mcd=calc_molchan(yte,ypr)
        metrics[mtype]['molchan_skill']=mcd['skill_score']
        metrics[mtype]['molchan_auc']=mcd['molchan_auc']
        plot_molchan(mcd,fn=f'molchan_{mtype}.png')
        psr=prospective_sim(tel,cal,af)
        metrics[mtype]['prospective_accuracy']=(
            psr['correct'].mean()*100 if not psr.empty else 0)
        joblib.dump(cal,mp[mtype])
        print(f"{G_}{mtype.upper()} Hazir | AUC:{metrics[mtype]['auc']:.3f}{X_}")
    return models,metrics,expl

def build_lstm(shape):
    return Sequential([
        Input(shape=shape),
        LSTM(128,return_sequences=True,dropout=0.3),BatchNormalization(),
        LSTM(64,return_sequences=True,dropout=0.3),BatchNormalization(),
        LSTM(32,dropout=0.3),BatchNormalization(),
        Dense(64,activation='relu',kernel_regularizer=l2(0.01)),Dropout(0.5),
        Dense(32,activation='relu',kernel_regularizer=l2(0.01)),Dropout(0.3),
        Dense(1,activation='sigmoid')])

def train_lstm(df_full, new_ids, force=False):
    mpath="lstm_v5.keras"; spath="lstm_scaler_v5.joblib"
    if not force and os.path.exists(mpath) and os.path.exists(spath):
        return load_model(mpath),joblib.load(spath),{}
    cutoff=df_full['time'].quantile(0.8)
    trd=df_full[df_full['time']<=cutoff].copy()
    ted=df_full[df_full['time']>cutoff].copy()
    if len(ted)<10: return None,None,{}
    trl=create_labels(fix_numeric(trd).copy())
    tel=create_labels(fix_numeric(ted).copy())
    if trl.empty: return None,None,{}
    af=[f for f in ENHANCED_FEATURES if f in trl.columns]
    sc=StandardScaler()
    trs=sc.fit_transform(trl[af].apply(safe_fill))
    tes=sc.transform(tel[af].apply(safe_fill))
    sl=50; Xtr,ytr=[],[]
    for i in range(len(trs)-sl):
        Xtr.append(trs[i:i+sl]); ytr.append(trl[TARGET].iloc[i+sl])
    Xte,yte=[],[]
    for i in range(len(tes)-sl):
        Xte.append(tes[i:i+sl]); yte.append(tel[TARGET].iloc[i+sl])
    Xtr=np.array(Xtr); ytr=np.array(ytr)
    Xte=np.array(Xte); yte=np.array(yte)
    if len(Xtr)<100 or len(Xte)==0: return None,None,{}
    mdl=build_lstm((Xtr.shape[1],Xtr.shape[2]))
    mdl.compile(optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',metrics=['accuracy'])
    es=EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True)
    lr=ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=7)
    cw={0:1.0,1:len(ytr)/(2*np.sum(ytr)) if np.sum(ytr)>0 else 1.0}
    mdl.fit(Xtr,ytr,epochs=100,batch_size=32,validation_data=(Xte,yte),
            callbacks=[es,lr],class_weight=cw,verbose=1)
    ypm=mdl.predict(Xte,verbose=0).flatten()
    mdl.save(mpath); joblib.dump(sc,spath)
    return mdl,sc,get_metrics(yte,(ypm>0.5).astype(int),ypm)

def predict_unc(dfp, models, lm, ls):
    if dfp.empty: return pd.DataFrame()
    dfp=fix_numeric(dfp)
    af=[f for f in ENHANCED_FEATURES if f in dfp.columns]
    insuf=(dfp['b_value_local'].isna())|(dfp['b_value_local']==0)|(dfp['event_rate_local']==0)
    dp=dfp[af].apply(safe_fill)
    dpr=pd.DataFrame(index=dfp.index)
    for mk,mdl in models.items():
        if mdl: dpr[f'{mk}_prob']=mdl.predict_proba(dp)[:,1]*100
        else: dpr[f'{mk}_prob']=np.nan
    lp=np.zeros(len(dfp)); lu=np.zeros(len(dfp))
    if lm and ls:
        try:
            ds=ls.transform(dp); seqs=[]
            for i in range(len(ds)):
                sq=ds[max(0,i-49):i+1]
                if len(sq)<50: sq=np.vstack([np.zeros((50-len(sq),ds.shape[1])),sq])
                seqs.append(sq)
            if seqs:
                seqs=np.array(seqs)
                mcp=np.array([lm(seqs,training=True).numpy().flatten() for _ in range(10)])
                lp=np.mean(mcp,axis=0)*100; lu=np.std(mcp,axis=0)*100
        except: lp[:]=np.nan
    dpr['lstm_prob']=lp; dpr['lstm_uncertainty']=lu
    fps=[]; fcs=[]
    for i in range(len(dpr)):
        if insuf.iloc[i]: fps.append(np.nan); fcs.append(0.0); continue
        px=dpr.iloc[i].get('xgb_prob',np.nan)
        pr=dpr.iloc[i].get('rf_prob',np.nan)
        pl=dpr.iloc[i]['lstm_prob']; ul=dpr.iloc[i]['lstm_uncertainty']
        vp=[]; wt=[]
        if not np.isnan(px): vp.append(px); wt.append(0.4)
        if not np.isnan(pr): vp.append(pr); wt.append(0.2)
        if not np.isnan(pl):
            wl=0.4*np.exp(-ul/10); vp.append(pl); wt.append(wl)
        if vp and sum(wt)>0:
            wp=max(0.1,min(99.9,np.average(vp,weights=wt)))
            fps.append(wp)
            bu=ul if not np.isnan(ul) else 20.0
            fcs.append(max(0,100-bu))
        else: fps.append(np.nan); fcs.append(0.0)
    dpr['olasilik']=fps; dpr['confidence_score']=fcs
    dpr['total_uncertainty']=100-dpr['confidence_score']
    return dpr

def add_legend(m, title, items):
    body="".join([f'<i class="fa fa-circle" style="color:{c}"></i> {l}<br>'
                  for l,c in items.items()])
    # SAĞA ALINAN LEJANT KISMI (right:50px):
    html=(f'<div style="position:fixed;bottom:50px;right:50px;width:280px;'
          f'padding:10px;border:2px solid grey;z-index:9999;font-size:13px;'
          f'background:white;border-radius:5px"><b>{title}</b><br>{body}</div>')
    m.get_root().html.add_child(folium.Element(html))
    m.get_root().header.add_child(folium.Element(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/'
        'font-awesome/4.7.0/css/font-awesome.min.css">'))

def base_map(dm):
    if dm.empty:
        return folium.Map(location=[39.93,32.86],zoom_start=6,tiles="CartoDB positron")
    return folium.Map(location=[dm['latitude'].mean(),dm['longitude'].mean()],
                      zoom_start=6,tiles="CartoDB positron")

def map_by_type(dfr, fn="deprem_haritasi_tip.html"):
    ms=(dfr['mag']>=4.0)&(dfr['earthquake_type'].isin(
        ['Ana Deprem','Oncu Deprem','Artci Deprem']))
    mi=(dfr['mag']>=4.5)&((dfr['earthquake_type']=='Tekil Deprem')|
        (dfr['earthquake_type'].isnull()))
    dm=dfr[ms|mi].copy(); m=base_map(dm)
    tcm={"Ana Deprem (Mainshock)":"red",
         "Oncu Deprem (Foreshock)":"orange",
         "Artci Deprem (Aftershock)":"blue",
         "Tekil Deprem (Isolated)":"gray"}
    type_map={"Ana Deprem":"Ana Deprem (Mainshock)",
              "Oncu Deprem":"Oncu Deprem (Foreshock)",
              "Artci Deprem":"Artci Deprem (Aftershock)",
              "Tekil Deprem":"Tekil Deprem (Isolated)"}
    color_map={"Ana Deprem":"red","Oncu Deprem":"orange",
               "Artci Deprem":"blue","Tekil Deprem":"gray"}
    for _,r in dm.iterrows():
        et=r.get('earthquake_type','Tekil Deprem')
        if pd.isna(et): et="Tekil Deprem"
        et_label=type_map.get(et,et)
        ts=pd.to_datetime(r['time']).strftime('%Y-%m-%d %H:%M')
        ph=(f"<b>Yer (Location):</b> {r['place']}<br>"
            f"<b>Buyukluk (Magnitude):</b> {r.get('mag')}<br>"
            f"<b>Tip (Type):</b> {et_label}<br>"
            f"<b>Tarih (Date):</b> {ts}")
        clr=color_map.get(et,'gray')
        folium.CircleMarker(
            location=[r['latitude'],r['longitude']],
            radius=min(r.get('mag',1)*2,12),
            popup=folium.Popup(ph,max_width=300),
            color=clr,fill=True,fill_color=clr,fill_opacity=0.7,weight=1
        ).add_to(m)
    add_legend(m,"Deprem Tipi (Earthquake Type)",tcm)
    m.save(fn)

def map_by_prob(dfr, fn="deprem_haritasi_olasilik.html"):
    if 'olasilik' not in dfr.columns: return
    dm=dfr[(dfr['mag']>=4.5)&(dfr['olasilik'].notna())].copy()
    m=base_map(dm)
    for _,r in dm.iterrows():
        p=r.get('olasilik',0); c=r.get('confidence_score',50)
        if p>30: clr='darkred' if c>70 else 'red'
        elif p>15: clr='darkorange' if c>70 else 'orange'
        else: clr='yellow'
        ts=pd.to_datetime(r['time']).strftime('%Y-%m-%d %H:%M')
        ph=(f"<b>Yer (Location):</b> {r['place']}<br>"
            f"<b>Tarih (Date):</b> {ts}<br>"
            f"<b>Buyukluk (Magnitude):</b> {r.get('mag')}<br>"
            f"<b>Olasilik (Probability):</b> {p:.1f}%<br>"
            f"<b>Guven (Confidence):</b> {c:.1f}")
        folium.CircleMarker(
            location=[r['latitude'],r['longitude']],
            radius=min(r.get('mag',1)*2,15),
            popup=folium.Popup(ph,max_width=350),
            color=clr,fill=True,fill_color=clr,fill_opacity=0.7
        ).add_to(m)
    li={"Yuksek Risk (High Risk) >%30 / Yuksek Guven (High Conf.)":"darkred",
        "Yuksek Risk (High Risk) >%30 / Dusuk Guven (Low Conf.)":"red",
        "Orta Risk (Medium Risk) >%15 / Yuksek Guven (High Conf.)":"darkorange",
        "Orta Risk (Medium Risk) >%15 / Dusuk Guven (Low Conf.)":"orange",
        "Dusuk Risk (Low Risk)":"yellow"}
    add_legend(m,"Olasilik ve Guven (Probability & Confidence)",li)
    m.save(fn)

def gen_report(dfr, user, rtime, summary, new_ids, minfo, expl):
    recent=dfr.sort_values('time',ascending=False).head(2000)
    filt=recent[recent['mag']>=4.0].copy()
    filt['time']=pd.to_datetime(filt['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    rt=filt[['time','place','mag','depth',
             'earthquake_type','olasilik','confidence_score']].copy()
    rt.columns=[
        'Zaman (Time UTC)',
        'Yer (Location)',
        'Buyukluk (Magnitude)',
        'Derinlik km (Depth km)',
        'Deprem Tipi (Earthquake Type)',
        'M>=5.5 Oncu Olasiligi % (Foreshock Probability %)',
        'Guven Skoru (Confidence Score)']

    pc='M>=5.5 Oncu Olasiligi % (Foreshock Probability %)'
    gc='Guven Skoru (Confidence Score)'

    def fp(row):
        v=row[pc]
        if pd.isna(v) or v=="":
            return ('<span style="color:gray">'
                    'Veri Yetersiz (Insufficient Data)</span>')
        try: 
            val = float(v)
            # Eğer değer 20.00 veya daha büyükse kırmızı ve kalın yaz
            if val >= 20.00:
                return f'<span style="color:red;font-weight:bold">{val:.2f}</span>'
            return f"{val:.2f}"
        except:
            return ('<span style="color:gray">'
                    'Veri Yetersiz (Insufficient Data)</span>')

    def fc(row):
        if pd.isna(row[pc]) or row[pc]=="": return "-"
        try: return f"{float(row[gc]):.2f}"
        except: return "-"

    def ft(v):
        clrs={"Ana Deprem":"#D32F2F","Oncu Deprem":"#F57C00",
              "Artci Deprem":"#1976D2","Tekil Deprem":"#616161"}
        labels={"Ana Deprem":"Ana Deprem (Mainshock)",
                "Oncu Deprem":"Oncu Deprem (Foreshock)",
                "Artci Deprem":"Artci Deprem (Aftershock)",
                "Tekil Deprem":"Tekil Deprem (Isolated)"}
        cl=clrs.get(v,"#333"); lb=labels.get(v,v)
        return f'<span style="color:{cl};font-weight:bold">{lb}</span>'

    rt[pc]=rt.apply(fp,axis=1)
    rt[gc]=rt.apply(fc,axis=1)
    rt['Deprem Tipi (Earthquake Type)']=rt[
        'Deprem Tipi (Earthquake Type)'].apply(ft)

    perf=""
    if minfo:
        rows=""
        for mn,mt in minfo.items():
            if mt:
                rows+=(
                    f"<tr><td><b>{mn.upper()}</b></td>"
                    f"<td>{mt.get('auc',0):.3f}</td>"
                    f"<td>{mt.get('molchan_skill',0):.3f}</td>"
                    f"<td>%{mt.get('prospective_accuracy',0):.1f}</td>"
                    f"<td>{mt.get('brier_score',0):.3f}</td></tr>")
        if rows:
            perf=(
                "<h3>Model Performans Metrikleri "
                "(Model Performance Metrics)</h3>"
                "<table style='width:100%;border-collapse:collapse'>"
                "<tr>"
                "<th>Model</th>"
                "<th>AUC</th>"
                "<th>Molchan Beceri Skoru (Skill Score)</th>"
                "<th>Prospektif Dogruluk (Prospective Accuracy)</th>"
                "<th>Brier Skoru (Brier Score)</th>"
                "</tr>"
                +rows+
                "</table>"
                "<p style='font-size:.9em;color:#666;margin-top:10px'>"
                "<b>Molchan Beceri Skoru (Molchan Skill Score):</b> "
                "1.0 = Mukemmel (Perfect), "
                "0.0 = Rastgele (Random), "
                "&lt;0 = Rastgeleden kotu (Worse than random)<br>"
                "<b>Prospektif Dogruluk (Prospective Accuracy):</b> "
                "Gercek zamanli tahmin simulasyonu basari orani "
                "(Real-time prediction simulation success rate)</p>")

    tbl=rt.to_html(index=False,escape=False)

    html=(
        "<!DOCTYPE html><html lang='tr'><head>"
        "<meta charset='UTF-8'>"
        "<title>Bilimsel Sismik Risk Raporu "
        "(Scientific Seismic Risk Report)</title>"
        "<style>"
        "body{font-family:'Segoe UI',sans-serif;padding:20px;"
        "background:#f5f5f5}"
        "h1{color:#2c3e50}"
        "h3{color:#34495e;margin-top:30px}"
        "table{width:100%;border-collapse:collapse;background:white;"
        "box-shadow:0 1px 3px rgba(0,0,0,.2);margin-top:15px}"
        "th,td{padding:12px;border-bottom:1px solid #ddd;text-align:left}"
        "th{background:#004d40;color:white}"
        "tr:hover{background:#f1f1f1}"
        ".info{background:#e8f5e9;padding:15px;border-radius:5px;"
        "border-left:5px solid #2e7d32;margin:20px 0}"
        "</style></head><body>"

        "<h1>Bilimsel Sismik Risk Analiz Raporu<br>"
        "<span style='font-size:0.7em;color:#555'>"
        "Scientific Seismic Risk Analysis Report</span></h1>"

        f"<p><b>Rapor Tarihi (Report Date):</b> {rtime} | "
        f"<b>Kullanici (User):</b> {user}</p>"

        f"<div class='info'>"
        f"<b>Ozet (Summary):</b> {summary}"
        f"</div>"

        f"{perf}"

        "<h3>Son Depremler (Recent Earthquakes) — M 4.0+</h3>"
        f"{tbl}"

        "<div style='margin-top:20px;font-size:.9em;color:#666'>"

        "<p><b>Not (Note):</b> "
        "'Veri Yetersiz (Insufficient Data)' ifadesi, ilgili depremin "
        "cevresinde yeterli sismik gecmis verisi bulunamadigini belirtir. "
        "(Indicates that not enough seismic history data was found "
        "in the surrounding area of the earthquake.)</p>"

        "<p><b>Bilimsel Metodoloji (Scientific Methodology):</b> "
        "Bu rapor, zamansal nedensellik korunan retrospektif egitim ve "
        "prospektif test metodolojisi ile olusturulmustur. "
        "(This report was generated using retrospective training and "
        "prospective testing methodology that preserves "
        "temporal causality.)</p>"

        "<p><b>Model Hedefi (Model Target):</b> "
        "Onumuzdeki 30 gun icinde, 50 km yaricapinda M&gt;=5.5 "
        "buyuklugunde bir ana deprem olma olasiligi. "
        "(Probability of a M&gt;=5.5 mainshock occurring within "
        "30 days and 50 km radius.)</p>"

        "</div></body></html>")

    with open("deprem_analiz_raporu_sade.html","w",encoding='utf-8') as f:
        f.write(html)
    print(f"{G_}Rapor olusturuldu (Report generated).{X_}")


def main():
    t0=time.time()
    db="earthquakes_3_5_plus_scientific_v5.db"
    tn="earthquake_catalog"
    conn=None
    try:
        print(f"{C_}{'='*70}")
        print("Sismik Analiz v16 (Seismic Analysis v16)")
        print(f"{'='*70}{X_}")

        conn=sqlite3.connect(db)
        new_db=setup_database(conn,tn)

        if not new_db:
            try:
                dfc=pd.read_sql(f"SELECT * FROM {tn}",conn)
                dfc['time']=dfc['time'].apply(standardize_date)
                dfc.dropna(subset=['time'],inplace=True)
                conn.execute(f"DELETE FROM {tn}")
                dfc.to_sql(tn,conn,if_exists='append',index=False)
                conn.commit()
            except: pass

        force=False
        new_ids=fetch_and_load_api_data(conn,tn)
        if new_ids: force=True

        df=pd.read_sql_query(f"SELECT * FROM {tn}",conn)
        df.drop_duplicates(subset=['eventID'],inplace=True,keep='last')
        df['time']=pd.to_datetime(df['time'],utc=True)

        if len(df)<100:
            fetch_and_load_api_data(conn,tn,
                                    start_override='2023-01-01 00:00:00')
            df=pd.read_sql_query(f"SELECT * FROM {tn}",conn)
            df.drop_duplicates(subset=['eventID'],inplace=True,keep='last')
            df['time']=pd.to_datetime(df['time'],utc=True)
            force=True
            if len(df)<100:
                print(f"{R_}Yetersiz veri (Insufficient data).{X_}")
                return

        df=calc_features(df)
        df=classify_eq_type(df)

        models,metrics,expl=train_sklearn(df,new_ids,force=force)
        lm,ls,lmet=train_lstm(df,new_ids,force=force)

        ami={**metrics}
        if lmet: ami['lstm']=lmet

        dfr=df.copy()
        t7=pd.to_datetime(datetime.utcnow(),utc=True)-timedelta(days=7)
        rm=(dfr['time']>=t7)&(dfr['olasilik'].isnull())
        aids=set(new_ids)|set(dfr[rm]['eventID'])

        if aids:
            dtp=dfr[dfr['eventID'].isin(aids)].copy()
            preds=predict_unc(dtp,models,lm,ls)
            if not preds.empty:
                preds['eventID']=dtp['eventID'].values
                dfr=pd.merge(dfr,preds,on='eventID',
                             how='left',suffixes=('','_new'))
                for col in ['olasilik','confidence_score','total_uncertainty']:
                    nc=f'{col}_new'
                    if nc in dfr.columns:
                        dfr[col]=dfr[nc].fillna(dfr[col])
                        dfr.drop(columns=[nc],inplace=True)

        dfs=dfr.copy()
        dfs['time']=dfs['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        cur=conn.cursor()
        db_cols=[i[1] for i in cur.execute(
            f"PRAGMA table_info({tn})").fetchall()]
        dfs[db_cols].to_sql(tn,conn,if_exists='replace',index=False)

        map_by_type(dfr)
        map_by_prob(dfr)
        gen_report(
            dfr, CURRENT_USER, CURRENT_UTC_TIME,
            f"Toplam {len(dfr)} olay analiz edildi. "
            f"(Total {len(dfr)} events analyzed.)",
            new_ids, ami, expl)

        elapsed=time.time()-t0
        print(f"\n{G_}{'='*70}")
        print(f"Tamamlandi (Completed). "
              f"Sure (Duration): {elapsed:.1f} saniye (seconds)")
        print(f"{'='*70}{X_}")

    except Exception as e:
        print(f"{R_}Hata (Error): {e}{X_}")
        print(traceback.format_exc())
    finally:
        if conn: conn.close()


if __name__=="__main__":
    main()


# In[ ]:




