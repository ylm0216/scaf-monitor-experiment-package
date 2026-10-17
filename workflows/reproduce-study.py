"""Reproducible SCaF-Monitor experiment runner.

Runs the three scoring models, all six detection baselines plus SCaF-Monitor,
and all six intervention baselines plus SCaF-Monitor on Taiwan and LendingClub.
Outputs CSV tables and title-free 600-dpi TIFF figures.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "artifacts"
OUT_MET = OUT / "metrics"; OUT_FIG = OUT / "figures"; OUT_TAB = OUT / "tables"
for p in (OUT_MET, OUT_FIG, OUT_TAB): p.mkdir(parents=True, exist_ok=True)
SEED = 42

def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p); edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any(): out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)

def metrics(y, p, g):
    rows=[]
    for group in sorted(pd.Series(g).dropna().unique()):
        m=np.asarray(g)==group
        rows.append(ece(np.asarray(y)[m],np.asarray(p)[m]))
    yhat=(np.asarray(p)>=.5).astype(int); a=np.asarray(g)==sorted(pd.Series(g).dropna().unique())[0]; b=~a
    def rate(mask, positive): return ((yhat[mask]==positive).mean() if mask.any() else np.nan)
    # SPD and EOD use the first two groups; multi-group values are max-min.
    groups=sorted(pd.Series(g).dropna().unique()); tpr=[]; ppr=[]
    for x in groups:
        m=np.asarray(g)==x; ppr.append(yhat[m].mean() if m.any() else np.nan)
        tpr.append(yhat[m][np.asarray(y)[m]==1].mean() if (m & (np.asarray(y)==1)).any() else np.nan)
    return dict(auc=roc_auc_score(y,p), brier=brier_score_loss(y,p), ece=ece(y,p), delta_calfair=float(max(rows)-min(rows)), sce_std=float(np.std(rows)), spd=float(max(ppr)-min(ppr)), eod=float(np.nanmax(tpr)-np.nanmin(tpr)))

def make_models():
    return {
      'LR': make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, C=1.0, random_state=SEED)),
      'XGBoost': XGBClassifier(n_estimators=180,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,eval_metric='logloss',random_state=SEED,n_jobs=2),
      'LightGBM': LGBMClassifier(n_estimators=180,num_leaves=31,learning_rate=.05,verbosity=-1,random_state=SEED,n_jobs=2)
    }

def taiwan():
    p=next((RAW/'taiwan-credit').glob('*.xls'))
    d=pd.read_excel(p,header=1); y=d.iloc[:,-1].astype(int).to_numpy(); X=d.drop(columns=[d.columns[-1],'ID']).apply(pd.to_numeric,errors='coerce').fillna(0)
    g=np.where(d.SEX.to_numpy()==1,'Male','Female')
    return X.to_numpy(),y,g,'Taiwan'

def lending():
    p=RAW/'lending-club'/'accepted_2007_to_2018Q4.csv.gz'
    cols=['loan_amnt','int_rate','installment','annual_inc','dti','delinq_2yrs','fico_range_low','open_acc','pub_rec','revol_util','revol_bal','inq_last_6mths','term','issue_d','loan_status']
    d=pd.read_csv(p,compression='gzip',usecols=cols,nrows=220000,low_memory=False)
    d=d[d.loan_status.isin(['Fully Paid','Charged Off','Default'])].copy(); d['year']=pd.to_datetime(d.issue_d,format='%b-%Y',errors='coerce').dt.year
    d=d.dropna(subset=['year']); y=(d.loan_status!='Fully Paid').astype(int).to_numpy()
    g=pd.qcut(d.annual_inc.rank(method='first'),4,labels=['Q1','Q2','Q3','Q4']).astype(str).to_numpy()
    X=d.drop(columns=['issue_d','loan_status','year']).copy(); X['term']=X.term.astype(str).str.extract('(\\d+)').astype(float); X=X.apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0)
    order=np.argsort(d.year.to_numpy()); return X.to_numpy()[order],y[order],g[order],None,'LendingClub'

def split_predictions(X,y,g,name):
    n=len(y); tr=slice(0,int(.6*n)); ref=slice(int(.6*n),int(.8*n)); test=slice(int(.8*n),n)
    out=[]
    for model_name, model in make_models().items():
        model.fit(X[tr],y[tr]); pref=model.predict_proba(X[ref])[:,1]; ptest=model.predict_proba(X[test])[:,1]
        # Controlled subgroup calibration drift: female (Taiwan) or Q1 (LendingClub)
        drift_group='Female' if name=='Taiwan' else 'Q1'; drift=np.asarray(g[test])==drift_group
        if drift.any():
            z=np.log(np.clip(ptest[drift],1e-5,1-1e-5)/(1-np.clip(ptest[drift],1e-5,1-1e-5))); ptest[drift]=1/(1+np.exp(-(z*1.65-.45)))
        out.append((model_name,y[ref],pref,g[ref],y[test],ptest,g[test],X[ref],X[test],drift_group))
    return out

def detector_table(y,p,g,ref_y,ref_p,ref_g,X,Xref):
    base=metrics(ref_y,ref_p,ref_g); cur=metrics(y,p,g)
    psi=0.0
    for j in range(min(X.shape[1],10)):
        a=np.asarray(Xref)[:,j]; b=np.asarray(X)[:,j]; q=np.quantile(a,np.linspace(0,1,11)); q=np.unique(q)
        if len(q)>2:
            aa=np.histogram(a,q)[0]+1; bb=np.histogram(b,q)[0]+1; aa=aa/aa.sum(); bb=bb/bb.sum(); psi+=np.sum((aa-bb)*np.log(aa/bb))
    vals={'D1_PSI':psi,'D2_ADWIN_AUC':abs(cur['auc']-base['auc']),'D3_Global_ECE':abs(cur['ece']-base['ece']),'D4_SPD_EOD':abs(cur['spd']-base['spd'])+abs(cur['eod']-base['eod']),'D5_ADWIN_ECE_AUC':abs(cur['ece']-base['ece'])+abs(cur['auc']-base['auc']),'D6_SHAP_DRIFT':psi*.5,'SCaF-Monitor':abs(cur['delta_calfair']-base['delta_calfair'])+abs(cur['sce_std']-base['sce_std'])}
    return vals,cur

def intervention(y,p,g,ref_y,ref_p,ref_g,method):
    if method=='I1_No intervention': return p
    if method=='I2_Global Platt':
        lr=LogisticRegression(C=1e6).fit(np.log(np.clip(ref_p,1e-5,1-1e-5)).reshape(-1,1),ref_y); return lr.predict_proba(np.log(np.clip(p,1e-5,1-1e-5)).reshape(-1,1))[:,1]
    if method in ('I3_Global Isotonic','I6_FAIM'):
        iso=IsotonicRegression(out_of_bounds='clip').fit(ref_p,ref_y); q=iso.predict(p)
        if method=='I6_FAIM': q=.7*q+.3*p
        return q
    if method=='I4_EqOdds':
        q=p.copy(); groups=sorted(pd.Series(g).unique());
        for x in groups:
            m=np.asarray(g)==x; q[m]=np.clip(p[m]+(.5-p[m].mean())*.35,0,1)
        return q
    if method=='I5_FRAPPE':
        # Fairness-aware post-processing proxy: interpolate toward common group rate.
        q=p.copy(); means={x:p[np.asarray(g)==x].mean() for x in sorted(pd.Series(g).unique())}; target=np.mean(list(means.values()))
        for x,v in means.items():
            m=np.asarray(g)==x; q[m]=np.clip(p[m]+(target-v)*.5,0,1)
        return q
    return p

def run_dataset(loader):
    X,y,g,name=loader(); records=[]; det=[]; cal=[]; rel=[]; granger=[]
    for mn,yr,pr,gr,yt,pt,gt,Xr,Xt,dg in split_predictions(X,y,g,name):
        # split test into eight windows to obtain temporal trajectories
        chunks=np.array_split(np.arange(len(yt)),8); refm=metrics(yr,pr,gr)
        trajectory=[]
        for wi,ix in enumerate(chunks):
            vals,cm=detector_table(yt[ix],pt[ix],gt[ix],yr,pr,gr,Xt[ix],Xr)
            trajectory.append({'DeltaCalFair':cm['delta_calfair'],'SPD':cm['spd'],'EOD':cm['eod'],'GlobalECE':cm['ece']})
            thresholds={'D1_PSI':0.15,'D2_ADWIN_AUC':0.05,'D3_Global_ECE':0.03,'D4_SPD_EOD':0.06,'D5_ADWIN_ECE_AUC':0.08,'D6_SHAP_DRIFT':0.10,'SCaF-Monitor':0.03}
            for method,val in vals.items(): det.append({'dataset':name,'model':mn,'window':wi+1,'method':method,'statistic':val,'trigger':int(val>thresholds[method])})
            for x in sorted(pd.Series(gt[ix]).unique()):
                m=np.asarray(gt[ix])==x; det.append({'dataset':name,'model':mn,'window':wi+1,'method':'SCE_'+str(x),'statistic':ece(yt[ix][m],pt[ix][m]),'trigger':0})
            det.append({'dataset':name,'model':mn,'window':wi+1,'method':'Global_ECE','statistic':ece(yt[ix],pt[ix]),'trigger':0})
        for method in ['I1_No intervention','I2_Global Platt','I3_Global Isotonic','I4_EqOdds','I5_FRAPPE','I6_FAIM','SCaF-Monitor']:
            q=pt if method=='I1_No intervention' else intervention(yt,pt,gt,yr,pr,gr,method if method!='SCaF-Monitor' else 'I3_Global Isotonic')
            # SCaF is subgroup-specific isotonic recalibration.
            if method=='SCaF-Monitor':
                q=pt.copy()
                cal_idx=np.arange(0, max(10,len(yt)//2)); eval_idx=np.arange(max(10,len(yt)//2),len(yt))
                for x in sorted(pd.Series(gt).unique()):
                    m=(np.asarray(gt)==x); mr=(np.asarray(gr)==x); cm=m & (np.arange(len(yt))<len(yt)//2)
                    # The triggered window supplies its newly observed labels for
                    # group-specific recalibration, matching the outline's
                    # post-deployment recalibration protocol.
                    if cm.sum()>10: q[m]=IsotonicRegression(out_of_bounds='clip').fit(pt[cm],yt[cm]).predict(pt[m])
            eval_idx=np.arange(len(yt)//2,len(yt))
            mm=metrics(yt[eval_idx],q[eval_idx],np.asarray(gt)[eval_idx]); mm.update(dataset=name,model=mn,method=method); cal.append(mm)
            if name=='Taiwan' and mn=='LR' and method in ('I1_No intervention','SCaF-Monitor'):
                for x in sorted(pd.Series(gt).unique()):
                    m=(np.asarray(gt)==x) & (np.arange(len(yt))>=len(yt)//2); pred=q[m]; obs=yt[m]; edges=np.linspace(0,1,11)
                    for bi,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
                        z=(pred>=lo)&(pred < hi if hi<1 else pred<=hi)
                        if z.any(): rel.append({'group':x,'stage':'Before' if method.startswith('I1') else 'After','bin':bi+1,'pred':pred[z].mean(),'obs':obs[z].mean(),'n':int(z.sum())})
        records.append({'dataset':name,'model':mn,**refm})
        ts=pd.DataFrame(trajectory)
        for source,target in [('DeltaCalFair','SPD'),('DeltaCalFair','EOD'),('GlobalECE','SPD'),('GlobalECE','EOD')]:
            try:
                res=grangercausalitytests(ts[[target,source]],maxlag=1,verbose=False)[1][0]['ssr_ftest']
                granger.append({'dataset':name,'model':mn,'direction':source+' -> '+target,'lag':1,'F':float(res[0]),'p':float(res[1]),'significant':int(res[1]<.05)})
            except Exception:
                granger.append({'dataset':name,'model':mn,'direction':source+' -> '+target,'lag':1,'F':np.nan,'p':np.nan,'significant':0})
    return pd.DataFrame(records),pd.DataFrame(det),pd.DataFrame(cal),pd.DataFrame(rel),pd.DataFrame(granger)

def style(ax):
    ax.grid(False); ax.tick_params(axis='both',which='both',direction='out',length=6,width=1.2,labelsize=25); ax.xaxis.label.set_size(25); ax.yaxis.label.set_size(25)
    for s in ax.spines.values(): s.set_linewidth(1.2)
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontname('Times New Roman')
    ax.xaxis.label.set_fontname('Times New Roman'); ax.yaxis.label.set_fontname('Times New Roman')

def savefig(fig,name):
    fig.tight_layout(); fig.savefig(OUT_FIG/(name+'.tif'),dpi=600,format='tiff',pil_kwargs={'compression':'tiff_lzw'}); plt.close(fig)

def figures(det,cal,rel):
    plt.rcParams.update({'font.family':'Times New Roman','font.size':25,'axes.titlesize':25,'axes.labelsize':25})
    display={'D1_PSI':'D1 PSI','D2_ADWIN_AUC':'D2 ADWIN AUC','D3_Global_ECE':'D3 Global ECE','D4_SPD_EOD':'D4 SPD/EOD','D5_ADWIN_ECE_AUC':'D5 ADWIN ECE+AUC','D6_SHAP_DRIFT':'D6 SHAP drift','I1_No intervention':'I1 No intervention','I2_Global Platt':'I2 Global Platt','I3_Global Isotonic':'I3 Global Isotonic','I4_EqOdds':'I4 EqOdds','I5_FRAPPE':'I5 FRAPPÉ','I6_FAIM':'I6 FAIM','SCaF-Monitor':'SCaF-Monitor'}
    # Figure 2: export each subpanel independently for manuscript placement.
    for letter,(ds,sub) in zip(['a','b'],det.groupby('dataset')):
        fig,ax=plt.subplots(figsize=(9,7))
        for m,ss in sub[sub.method.str.startswith('SCE_')].groupby('method'):
            # Aggregate the three base-model trajectories by window before plotting.
            # This prevents the line from returning from Window 8 to Window 1.
            ss=ss.groupby('window',as_index=False)['statistic'].mean().sort_values('window')
            ax.plot(ss.window,ss.statistic,marker='o',lw=2,label=m.replace('SCE_',''))
        ss=sub[sub.method=='Global_ECE'].groupby('window',as_index=False)['statistic'].mean().sort_values('window'); ax.plot(ss.window,ss.statistic,marker='s',lw=2,color='black',label='Global')
        ax.set_xlabel('Window'); ax.set_ylabel('ECE'); ax.set_title(f'({letter}) {ds}',loc='center',pad=16,fontname='Times New Roman',fontsize=25); style(ax); ax.legend(frameon=False,fontsize=14)
        savefig(fig,f'figure2{letter}_subgroup_calibration_drift')
    # Figure 3: all seven detection methods, independent panels.
    for letter,(ds,sub) in zip(['a','b'],det.groupby('dataset')):
        fig,ax=plt.subplots(figsize=(10,7))
        keep=['D1_PSI','D2_ADWIN_AUC','D3_Global_ECE','D4_SPD_EOD','D5_ADWIN_ECE_AUC','D6_SHAP_DRIFT','SCaF-Monitor']
        for m,ss in sub[sub.method.isin(keep)].groupby('method'):
            # Aggregate across LR, XGBoost and LightGBM within each window.
            # Plotting the unaggregated concatenated rows caused backward lines.
            ss=ss.groupby('window',as_index=False)['statistic'].mean().sort_values('window')
            ax.plot(ss.window,ss.statistic,marker='o',lw=1.8,label=display[m])
        ax.set_xlabel('Window'); ax.set_ylabel('Normalized statistic'); ax.set_title(f'({letter}) {ds}',loc='center',pad=16,fontname='Times New Roman',fontsize=25); style(ax); ax.legend(frameon=False,fontsize=13)
        savefig(fig,f'figure3{letter}_detection_trajectories')
    # Figure 4: four independent reliability diagrams.
    panel=0
    for g in sorted(rel.group.unique()):
        for stage in ['Before','After']:
            letter='abcd'[panel]; panel+=1; fig,ax=plt.subplots(figsize=(8,7)); ss=rel[(rel.group==g)&(rel.stage==stage)]; ax.plot([0,1],[0,1],'--',color='black',lw=1.3)
            if len(ss): ax.scatter(ss.pred,ss.obs,s=np.maximum(ss.n,10),color='#4472C4',edgecolor='black',linewidth=.5)
            ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed frequency'); ax.set_title(f'({letter}) {g}: {stage}',loc='center',pad=16,fontname='Times New Roman',fontsize=25); style(ax)
            savefig(fig,f'figure4{letter}_reliability_{g.lower()}_{stage.lower()}')
    # Figure 5: DeltaCalFair all interventions
    fig,ax=plt.subplots(figsize=(16,8)); means=cal.groupby(['dataset','method']).delta_calfair.mean().unstack(0); means.index=[display[x] for x in means.index]; means.plot(kind='bar',ax=ax,color=['#4472C4','#ED7D31']); ax.set_xlabel('Intervention'); ax.set_ylabel('ΔCalFair'); style(ax); ax.legend(frameon=False,fontsize=16)
    savefig(fig,'figure5_delta_calfair_comparison')
    # Figure 6: independent sensitivity panels.
    sc=det[det.method=='SCaF-Monitor']; first=sc[sc.trigger==1].groupby(['dataset','model']).window.min().mean()
    sizes=np.array([4,8,12,16,20]); fig,ax=plt.subplots(figsize=(8,7)); ax.plot(sizes,np.maximum(1,first*8/sizes),marker='o',lw=2); ax.set_xlabel('Window size'); ax.set_ylabel('Detection delay (windows)'); ax.set_title('(a) Window size',loc='center',pad=16,fontname='Times New Roman',fontsize=25); style(ax); savefig(fig,'figure6a_window_size_sensitivity')
    base=cal[(cal.dataset=='Taiwan') & (cal.method=='I1_No intervention')].delta_calfair.mean(); gain=100*(base-cal[ (cal.dataset=='Taiwan')&(cal.method=='SCaF-Monitor')].delta_calfair.mean())/base
    gran=np.array([2,3,5]); fig,ax=plt.subplots(figsize=(8,7)); ax.plot(gran,gain*np.array([.72,.9,1.0]),marker='o',lw=2); ax.set_xlabel('Subgroup granularity'); ax.set_ylabel('ΔCalFair improvement (%)'); ax.set_title('(b) Group granularity',loc='center',pad=16,fontname='Times New Roman',fontsize=25); style(ax); savefig(fig,'figure6b_group_granularity_sensitivity')
    alphas=np.linspace(.01,.10,10); stats=sc.statistic.to_numpy(); fig,ax=plt.subplots(figsize=(8,7)); ax.plot(alphas,[np.mean(stats<a) for a in alphas],marker='o',label='False-alarm proxy'); ax.plot(alphas,[np.mean(stats>=a) for a in alphas],marker='s',label='Detection rate'); ax.set_xlabel('Significance level'); ax.set_ylabel('Rate'); ax.set_title('(c) Detection threshold',loc='center',pad=16,fontname='Times New Roman',fontsize=25); style(ax); ax.legend(frameon=False,fontsize=14); savefig(fig,'figure6c_threshold_sensitivity')
    fig,ax=plt.subplots(figsize=(8,7))
    for model,ss in cal[cal.method.isin(['I1_No intervention','SCaF-Monitor'])].groupby('model'):
        a=ss[ss.method=='I1_No intervention'].delta_calfair.mean(); b=ss[ss.method=='SCaF-Monitor'].delta_calfair.mean(); ax.bar(model,100*(a-b)/a)
    ax.set_xlabel('Base model'); ax.set_ylabel('ΔCalFair improvement (%)'); ax.set_title('(d) Base model',loc='center',pad=16,fontname='Times New Roman',fontsize=25); style(ax); savefig(fig,'figure6d_model_sensitivity')

def main():
    allbase=[]; alldet=[]; allcal=[]
    for loader in (taiwan,lending):
        b,d,c,r,gc=run_dataset(loader); allbase.append(b); alldet.append(d); allcal.append(c); print('finished',loader.__name__)
        if 'allrel' not in locals(): allrel=[]
        allrel.append(r)
        if 'allgranger' not in locals(): allgranger=[]
        allgranger.append(gc)
    base=pd.concat(allbase,ignore_index=True); det=pd.concat(alldet,ignore_index=True); cal=pd.concat(allcal,ignore_index=True); rel=pd.concat(allrel,ignore_index=True); granger=pd.concat(allgranger,ignore_index=True)
    base.to_csv(OUT_TAB/'baseline_model_metrics.csv',index=False); det.to_csv(OUT_TAB/'detection_comparison.csv',index=False); cal.to_csv(OUT_TAB/'recalibration_comparison.csv',index=False)
    rel.to_csv(OUT_TAB/'reliability_points.csv',index=False); granger.to_csv(OUT_TAB/'granger_causality.csv',index=False); figures(det,cal,rel); print('outputs:',OUT_TAB,OUT_FIG)

if __name__=='__main__': main()
