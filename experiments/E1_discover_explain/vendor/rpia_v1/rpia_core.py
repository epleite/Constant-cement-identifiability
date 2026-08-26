import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

# -------- field-informed baseline --------
P_PORE_MPA = 32.77
T_C = 106.2
P_EFF_MPA = 39.0
OIL_DEN = 0.883
GAS_G = 0.75
GOR0 = 114.0
SAL0 = 0.070
PHIC_PACK = 0.40

KQ,GQ,RQ = 37e9,44e9,2.65e3
KKF,GKF,RKF = 37.5e9,15e9,2.62e3
KKAO,GKAO,RKAO = 12e9,6e9,2.60e3
KILL,GILL,RILL = 27.7e9,17.8e9,2.79e3
FKF = 0.15
FILL = 0.10
KCEM,GCEM = 37e9,44e9

MEAS = {"Vp":50.0,"Vs":40.0,"rho":0.01}
NUI_SCALES = {
    "phi_bias":0.01,
    "vsh_bias":0.03,
    "sw_bias":0.05,
    "logP_eff_shift":0.18,
    "f_kf_shift":0.10,
    "log_clay_mod_scale":0.25,
    "brine_salinity_shift":0.03,
    "GOR_shift":20.0,
    "log_cement_mod_scale":0.20,
    "phic_pack_shift":0.02,
}
PARAM_SCALES = {
    "soft_sand":np.array([0.03,0.20]),
    "stiff_sand":np.array([0.03,0.20]),
    "contact_cement":np.array([0.03,0.20]),
    "constant_cement":np.array([0.015,0.20]),
    "DEM":np.array([0.30]),
}

def select_hugin_sand(df, interval, phi_min=.08, phi_max=.30, vsh_max=.30):
    d = df[(df.depth_m >= interval[0]) & (df.depth_m <= interval[1])].copy()
    d = d[(d.phi >= phi_min) & (d.phi <= phi_max) & (d.vsh <= vsh_max)].copy()
    return d.reset_index(drop=True)

# -------- fluids --------
def bw_live_oil(P,T,den,G,Rg):
    B=.972+.00038*(2.4*Rg*(G/den)**0.5+T+17.8)**1.175
    rp=den*(1+.001*Rg)**-1*B**-1
    v=2096*(rp/(2.6-rp))**0.5-3.7*T+4.64*P + \
      .0115*(4.12*(1.08/rp-1)**0.5-1)*T*P
    rho=(den+.0012*G*Rg)/B
    return rho,rho*v*v*1e3,v

def bw_water(T,P):
    rho=1+1e-6*(-80*T-3.3*T*T+.00175*T**3+489*P-2*T*P+
      .016*P*T*T-1.3e-5*T**3*P-.333*P*P-.002*T*P*P)
    w=np.array([[1.40285e3,1.524,3.437e-3,-1.197e-5],
      [4.871,-1.110e-2,1.739e-4,-1.628e-6],
      [-4.783e-2,2.747e-4,-2.135e-6,1.237e-8],
      [1.487e-4,-6.503e-7,-1.455e-8,1.327e-10],
      [-2.197e-7,7.987e-10,5.230e-11,-4.614e-13]])
    v=sum(w[i,j]*T**i*P**j for i in range(5) for j in range(4))
    return rho,rho*v*v*1e3,v

def bw_brine(T,P,S):
    rw,_,vw=bw_water(T,P)
    x=300*P-2400*P*S+T*(80+3*T-3300*S-13*P+47*P*S)
    rho=rw+S*(.668+.44*S+1e-6*x)
    s1=1170-9.6*T+.055*T*T-8.5e-5*T**3+2.6*P-.0029*T*P-.0476*P*P
    s15=780-10*P+.16*P*P
    v=vw+s1*S+s15*S**1.5-820*S*S
    return rho,rho*v*v*1e3,v

def fluid(sw,S=SAL0,Rg=GOR0):
    ro,Ko,_=bw_live_oil(P_PORE_MPA,T_C,OIL_DEN,GAS_G,Rg)
    rb,Kb,_=bw_brine(T_C,P_PORE_MPA,S)
    return 1/(sw/Kb+(1-sw)/Ko),(sw*rb+(1-sw)*ro)*1000

# -------- mineral matrix --------
def vrh(vol,K,G):
    vol=np.asarray(vol,float); vol=vol/vol.sum()
    K=np.asarray(K,float); G=np.asarray(G,float)
    return .5*(np.dot(vol,K)+1/np.dot(vol,1/K)), .5*(np.dot(vol,G)+1/np.dot(vol,1/G))

def matrix(vcl,fkf=FKF,clay_scale=1.):
    vcl=float(np.clip(vcl,0,.5)); fkf=float(np.clip(fkf,0,.45))
    vols=[(1-vcl)*(1-fkf),(1-vcl)*fkf,vcl*(1-FILL),vcl*FILL]
    K=[KQ,KKF,KKAO*clay_scale,KILL*clay_scale]
    G=[GQ,GKF,GKAO*clay_scale,GILL*clay_scale]
    Km,Gm=vrh(vols,K,G)
    rho=np.dot(vols,[RQ,RKF,RKAO,RILL])
    return Km,Gm,rho

def gassmann(Kd,Gd,Km,Kf,phi):
    den=phi/Kf+(1-phi)/Km-Kd/Km**2
    return Kd+(1-Kd/Km)**2/den,Gd

def elastic(Kd,Gd,Km,rhom,phi,sw,S=SAL0,Rg=GOR0):
    Kf,rf=fluid(sw,S,Rg)
    K,G=gassmann(Kd,Gd,Km,Kf,phi)
    rho=(1-phi)*rhom+phi*rf
    return np.sqrt((K+4*G/3)/rho),np.sqrt(G/rho),rho/1000

# -------- RPMs --------
def hm(Ks,Gs,pc,Cn,P_MPa=P_EFF_MPA):
    P=P_MPa*1e6
    nu=(3*Ks-2*Gs)/(2*(3*Ks+Gs))
    Kh=(Cn**2*(1-pc)**2*Gs**2*P/(18*np.pi**2*(1-nu)**2))**(1/3)
    Gh=((5-4*nu)/(5*(2-nu)))*(3*Cn**2*(1-pc)**2*Gs**2*P/(2*np.pi**2*(1-nu)**2))**(1/3)
    return Kh,Gh

def soft_dry(phi,Ks,Gs,pc,Cn,P=P_EFF_MPA):
    Kh,Gh=hm(Ks,Gs,pc,Cn,P); x=phi/pc
    z=Gh/6*(9*Kh+8*Gh)/(Kh+2*Gh)
    K=1/(x/(Kh+4*Gh/3)+(1-x)/(Ks+4*Gh/3))-4*Gh/3
    G=1/(x/(Gh+z)+(1-x)/(Gs+z))-z
    return K,G

def stiff_dry(phi,Ks,Gs,pc,Cn,P=P_EFF_MPA):
    Kh,Gh=hm(Ks,Gs,pc,Cn,P); x=phi/pc
    z=Gs/6*(9*Ks+8*Gs)/(Ks+2*Gs)
    K=1/(x/(Kh+4*Gs/3)+(1-x)/(Ks+4*Gs/3))-4*Gs/3
    G=1/(x/(Gh+z)+(1-x)/(Gs+z))-z
    return K,G

def contact(K0,G0,Kc,Gc,phi,phic,Cn,scheme=1):
    nu0=(3*K0-2*G0)/(6*K0+2*G0); nuc=(3*Kc-2*Gc)/(6*Kc+2*Gc)
    if scheme==1: a=2*((phic-phi)/(3*Cn*(1-phic)))**.25
    else: a=((2*(phic-phi))/(3*(1-phic)))**.5
    LN=2*Gc*(1-nu0)*(1-nuc)/(np.pi*G0*(1-2*nuc))
    Sn=(-.024153*LN**-1.3646)*a*a+(.20405*LN**-.89008)*a+.00024649*LN**-1.9864
    LT=Gc/(np.pi*G0)
    T1=-1e-2*(2.26*nu0**2+2.07*nu0+2.3)*LT**(.079*nu0**2+.1754*nu0-1.342)
    T2=(.0573*nu0**2+.0937*nu0+.202)*LT**(.0274*nu0**2+.0529*nu0-.8765)
    T3=1e-4*(9.654*nu0**2+4.945*nu0+3.1)*LT**(.01867*nu0**2+.4011*nu0-1.8186)
    St=T1*a*a+T2*a+T3
    K=(1/6)*Cn*(1-phic)*(Kc+4*Gc/3)*Sn
    G=3*K/5+(3/20)*Cn*(1-phic)*Gc*St
    return K,G

def constant(K0,G0,Kc,Gc,phi,phic,phib,Cn,scheme=1):
    Kb,Gb=contact(K0,G0,Kc,Gc,phib,phic,Cn,scheme)
    T=phi/phib
    Z=Gb/6*(9*Kb+8*Gb)/(Kb+2*Gb)
    K=(T/(Kb+4*Gb/3)+(1-T)/(K0+4*Gb/3))**-1-4*Gb/3
    G=(T/(Gb+Z)+(1-T)/(G0+Z))**-1-Z
    return K,G

def nuKG(K,G): return (3*K-2*G)/(6*K+2*G)
def pq_oblate(nu,a):
    p=np.pi
    Pm1=4*(1-nu**2)/(3*p*(1-2*nu)); Qm1=8*(1-nu)*(5-nu)/(15*p*(2-nu))
    P0=(1/6)*(1-nu)*(1-2*nu)
    P1=((1+nu)*(1-nu)/(12*(1-2*nu)))*(p*(1-2*nu)**2+8*(7-8*nu)/p)
    Q0=(2/15)*((5-2*nu**2)+48*(1-nu)*(3-nu)/(p**2*(2-nu)**2))
    Q1=(p/120)*(37-8*nu*(3+4*nu-2*nu**3))/(1-nu)+(4*(1-nu)/(15*p*(2-nu)**2))*(-8*(7+nu**3)+3*nu*(9*nu-1)+96*(3-nu)**2/(p**2*(2-nu)))
    return Pm1/a+P0+P1*a,Qm1/a+Q0+Q1*a

def dem(phi,K0,G0,a,n=80):
    h=phi/n; K=float(K0); G=float(G0); x=0.
    def f(xx,KK,GG):
        P,Q=pq_oblate(nuKG(KK,GG),a)
        return -KK*P/(1-xx),-GG*Q/(1-xx)
    for _ in range(n):
        k1=f(x,K,G); k2=f(x+h/2,K+h*k1[0]/2,G+h*k1[1]/2)
        k3=f(x+h/2,K+h*k2[0]/2,G+h*k2[1]/2); k4=f(x+h,K+h*k3[0],G+h*k3[1])
        K+=h*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
        G+=h*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6; x+=h
    return K,G

def observed(df): return np.c_[df.vp_mps,df.vs_mps,df.rho_gcc]

def forward(df,model,theta,nui=None):
    nui=nui or {}
    phi=np.clip(df.phi.to_numpy()+nui.get("phi_bias",0),.01,.45)
    vcl=np.clip(df.vsh.to_numpy()+nui.get("vsh_bias",0),0,.5)
    sw=np.clip(df.sw.to_numpy()+nui.get("sw_bias",0),0,1)
    P=P_EFF_MPA*np.exp(nui.get("logP_eff_shift",0))
    fkf=np.clip(FKF+nui.get("f_kf_shift",0),0,.45)
    clay=np.exp(nui.get("log_clay_mod_scale",0))
    sal=np.clip(SAL0+nui.get("brine_salinity_shift",0),.005,.20)
    Rg=np.clip(GOR0+nui.get("GOR_shift",0),20,250)
    cs=np.exp(nui.get("log_cement_mod_scale",0))
    phic_pack=np.clip(PHIC_PACK+nui.get("phic_pack_shift",0),.34,.46)
    out=[]
    for i in range(len(df)):
        Km,Gm,rm=matrix(vcl[i],fkf,clay)
        if model in ("soft_sand","stiff_sand","contact_cement"):
            pc,lnCn=theta; Cn=np.exp(lnCn)
            if phi[i]>=pc*.999: out.append([np.nan]*3); continue
            if model=="soft_sand": Kd,Gd=soft_dry(phi[i],Km,Gm,pc,Cn,P)
            elif model=="stiff_sand": Kd,Gd=stiff_dry(phi[i],Km,Gm,pc,Cn,P)
            else: Kd,Gd=contact(Km,Gm,KCEM*cs,GCEM*cs,phi[i],pc,Cn,1)
        elif model=="constant_cement":
            vcem,lnCn=theta; Cn=np.exp(lnCn); phib=phic_pack-vcem
            if phi[i]>=phib*.999: out.append([np.nan]*3); continue
            Kd,Gd=constant(Km,Gm,KCEM*cs,GCEM*cs,phi[i],phic_pack,phib,Cn,1)
        else:
            Kd,Gd=dem(phi[i],Km,Gm,np.exp(theta[0]))
        out.append(elastic(Kd,Gd,Km,rm,phi[i],sw[i],sal,Rg))
    return np.asarray(out)

def residual(theta,df,model):
    p=forward(df,model,theta); o=observed(df)
    if not np.all(np.isfinite(p)): return np.full(2*len(df),1e6)
    return np.r_[(p[:,0]-o[:,0])/150,(p[:,1]-o[:,1])/100]

def calibrate(df,model,expanded=False):
    if model=="DEM":
        f=lambda lna:np.mean(residual(np.array([lna]),df,model)**2)
        return np.array([minimize_scalar(f,bounds=(np.log(.015),np.log(.30)),method="bounded").x])
    if model=="constant_cement":
        lo=np.array([.001,np.log(3.)]); hi=np.array([.060,np.log(18.)]); x0=np.array([.025,np.log(8.5)])
    elif expanded:
        lo=np.array([df.phi.max()+.0005,np.log(2.)]); hi=np.array([.60,np.log(30.)]); x0=np.array([max(.40,lo[0]+.02),np.log(8.)])
    else:
        if model=="contact_cement": lo=np.array([max(df.phi.max()+.005,.30),np.log(3.)]); hi=np.array([.45,np.log(18.)])
        else: lo=np.array([max(df.phi.max()+.005,.30),np.log(4.)]); hi=np.array([.50,np.log(16.)])
        x0=np.array([max(.36,df.phi.max()+.03),np.log(8.5)])
    x0=np.minimum(np.maximum(x0,lo+1e-6),hi-1e-6)
    return least_squares(residual,x0,bounds=(lo,hi),args=(df,model),max_nfev=5000,xtol=1e-11,ftol=1e-11,gtol=1e-11).x

def nuisance_names(model):
    n=["phi_bias","vsh_bias","sw_bias","f_kf_shift","log_clay_mod_scale","brine_salinity_shift","GOR_shift"]
    if model in ("soft_sand","stiff_sand"): n+=["logP_eff_shift"]
    if model=="contact_cement": n+=["log_cement_mod_scale"]
    if model=="constant_cement": n+=["log_cement_mod_scale","phic_pack_shift"]
    return n

def stack(df,model,t,names,nui=None):
    p=forward(df,model,t,nui); ix={"Vp":0,"Vs":1,"rho":2}
    return np.concatenate([p[:,ix[n]] for n in names])

def jac(df,model,t,names,scale):
    sig=np.concatenate([np.full(len(df),scale[n]) for n in names])
    Jt=np.zeros((len(sig),len(t)))
    for j,s in enumerate(PARAM_SCALES[model]):
        h=1e-4; tp=t.copy(); tm=t.copy(); tp[j]+=h*s; tm[j]-=h*s
        Jt[:,j]=(stack(df,model,tp,names)-stack(df,model,tm,names))/(2*h)/sig
    nn=nuisance_names(model); Jn=np.zeros((len(sig),len(nn)))
    for j,n in enumerate(nn):
        h=1e-4; s=NUI_SCALES[n]
        Jn[:,j]=(stack(df,model,t,names,{n:h*s})-stack(df,model,t,names,{n:-h*s}))/(2*h)/sig
    return Jt,Jn,nn

def geom(Jt,Jn):
    G=Jt.T@Jt; C=Jt.T@Jn; N=Jn.T@Jn
    Ga=G-C@np.linalg.solve(N+np.eye(N.shape[0]),C.T); Ga=(Ga+Ga.T)/2
    raw=np.linalg.eigvalsh(G)[::-1]; adj=np.linalg.eigvalsh(Ga)[::-1]
    vals,V=np.linalg.eigh(G); keep=vals>max(vals.max()*1e-10,1e-12)
    L=V[:,keep]@np.diag(1/np.sqrt(vals[keep])); R=L.T@Ga@L
    ret=np.clip(np.linalg.eigvalsh((R+R.T)/2)[::-1],0,1)
    return raw,adj,ret,-np.linalg.pinv(Jt)@Jn
