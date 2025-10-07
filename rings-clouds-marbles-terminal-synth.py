#!/usr/bin/env python3
"""
Rings x Clouds x Marbles — Terminal Synth (v1.0 Stable)
-------------------------------------------------------
Classic Mutable Instruments–inspired patch that runs
entirely in your macOS / iTerm terminal.

Controls
---------
space      Start / Stop transport
tab        Switch panel (Marbles / Rings / Clouds / Mixer)
← / →      Select parameter
↑ / ↓      Adjust parameter
[ / ]      Change step size
r          Reseed Marbles
g          Toggle quantization
1..7       Choose musical scale
, / .      Shift base note down / up
c          Toggle Clouds freeze
f / F      Manual pluck
p          Tap tempo
q / Esc    Quit
"""

import math, time, random, curses
from collections import deque
import numpy as np
import sounddevice as sd

def db_to_lin(db): return 10 ** (db / 20.0)

SCALES = {
    1: ("Ionian (Major)", [0,2,4,5,7,9,11]),
    2: ("Dorian", [0,2,3,5,7,9,10]),
    3: ("Phrygian",[0,1,3,5,7,8,10]),
    4: ("Lydian",[0,2,4,6,7,9,11]),
    5: ("Mixolydian",[0,2,4,5,7,9,10]),
    6: ("Aeolian (Minor)",[0,2,3,5,7,8,10]),
    7: ("Locrian",[0,1,3,5,6,8,10]),
}

# ---------------- MARBLES ---------------- #
class Marbles:
    def __init__(self, sr):
        self.sr = sr; self.bpm = 90; self.jitter = 0.05
        self.bias = 0.5; self.spread = 0.5
        self.quantize = True; self.scale_id = 6; self.base_note = 57
        self._interval = 60.0 / self.bpm; self._last = time.time()
        self._seq = deque(maxlen=32); self._tap_times = deque(maxlen=8)
        self._rng = random.Random(1234)

    def reseed(self): self._rng.seed(self._rng.random()*1e9); self._seq.clear()
    def tap(self):
        now=time.time(); self._tap_times.append(now)
        if len(self._tap_times)>=2:
            iv=[t2-t1 for t1,t2 in zip(self._tap_times,list(self._tap_times)[1:])]
            avg=sum(iv)/len(iv)
            if 0.15<avg<3.0: self.bpm=60.0/avg; self._interval=avg
    def _quantize_midi(self,m):
        if not self.quantize: return m
        _,sc=SCALES[self.scale_id]; oc=int(math.floor(m/12))
        cands=[oc*12+s for s in sc]+[(oc+1)*12+s for s in sc]
        return min(cands,key=lambda x:abs(x-m))
    def next_trigger_and_pitch(self):
        now=time.time(); did=False; f0=None; vel=0.0
        if now-self._last>=self._interval*(1+(self._rng.random()*2-1)*self.jitter):
            self._last=now; did=True
            if self._rng.random()<0.8 and self._seq:
                midi=self._rng.choice(list(self._seq))
            else:
                c=self.base_note+(12*(self.bias-0.5)*1.5)
                w=12*(0.5+self.spread*1.5)
                midi=c+self._rng.gauss(0,w*0.3)
            midi=self._quantize_midi(midi); self._seq.append(midi)
            f0=440.0*(2**((midi-69)/12))
            vel=min(1.0,max(0.05,0.8+self._rng.uniform(-0.2,0.2)))
        return did,f0,vel

# ---------------- RINGS ---------------- #
class ModalResonator:
    def __init__(self,sr,n_modes=8):
        self.sr=sr; self.n=n_modes
        self.struct=0.5; self.brightness=0.6; self.damping=0.4; self.position=0.5
        self.strike_gain=1.0
        self._m_freqs=np.zeros(self.n); self._m_q=np.ones(self.n)*200
        self._m_state=np.zeros((self.n,2)); self._exc_buf=np.zeros(256)
        self._target_f0=110.0; self._f0=110.0
    def set_f0(self,f): self._target_f0=max(20,min(2000,float(f)))
    def excite(self,v=0.7):
        L=len(self._exc_buf); n=np.random.randn(L)
        a=0.3+0.6*self.position
        for i in range(1,L): n[i]=a*n[i]+(1-a)*n[i-1]
        n*=np.hanning(L)*(0.4+0.6*self.brightness); n[0]+=1.0
        self._exc_buf[:]=n*self.strike_gain*v
    def _update_modes(self):
        self._f0+=(self._target_f0-self._f0)*0.05; base=self._f0
        for i in range(self.n):
            mult=(i+1)*(1+self.struct*0.02*i)
            self._m_freqs[i]=base*mult
            q=100+1200*self.brightness; q*=(1-0.7*self.damping)
            self._m_q[i]=max(50,q)
    def process(self,in_block):
        self._update_modes(); out=np.zeros_like(in_block)
        take=min(len(in_block),len(self._exc_buf))
        if take>0: out[:take]+=self._exc_buf[:take]; self._exc_buf=np.roll(self._exc_buf,-take); self._exc_buf[-take:]=0
        for i in range(self.n):
            f=min(self._m_freqs[i],self.sr*0.3)
            if f<20: continue
            q=max(50.0,self._m_q[i]); g=math.tan(math.pi*f/self.sr); g=min(g,1.2); k=1.0/q
            z1,z2=self._m_state[i]; den=1.0+k*g+g*g; inv=1.0/den
            for j in range(out.size):
                x=out[j]; v1=(x-k*z1-z2)*g*inv; bp=g*v1+z1; z1=g*v1+bp; z2=g*bp+z2
                z1=max(-5,min(5,z1)); z2=max(-5,min(5,z2)); out[j]=math.tanh(bp*1.2)
            self._m_state[i]=(z1,z2)
        return out

# ---------------- CLOUDS ---------------- #
class GranularCloud:
    def __init__(self,sr,max_time=2.0):
        self.sr=sr; self.size=0.25; self.density=0.5; self.pitch=0.0
        self.feedback=0.3; self.blend=0.35; self.freeze=False
        self._buf_len=int(max_time*sr); self._buf=np.zeros(self._buf_len)
        self._w=0; self._rng=random.Random(7); self._reverb_z=np.zeros(4)
    def process(self,b):
        out=np.copy(b)
        if not self.freeze:
            for i,x in enumerate(b):
                self._buf[self._w]=0.5*x+self.feedback*self._buf[self._w]
                self._w=(self._w+1)%self._buf_len
        grains=np.zeros_like(b)
        hop=max(1,int((1-self.density*0.9)*len(b)))
        for s in range(0,len(b),hop):
            if self._rng.random()<self.density:
                d=int(self.size*self.sr*(0.5+self._rng.random()))
                r=(self._w-d)%self._buf_len; gl=min(int(self.size*self.sr),len(b)-s)
                win=np.hanning(gl); ratio=2**(self.pitch/12.0)
                idx=(np.arange(gl)*ratio).astype(int); idx=np.clip(idx,0,gl-1)
                src=np.array([self._buf[(r+k)%self._buf_len] for k in range(gl)])
                grains[s:s+gl]+=src[idx]*win
        wet=self._reverb(grains); return (1-self.blend)*b+self.blend*wet
    def _reverb(self,x):
        y=np.copy(x); cds=[int(self.sr*t) for t in (0.0297,0.0371,0.0411)]
        cgs=[0.77,0.74,0.73]
        for d,g in zip(cds,cgs):
            buf=np.zeros(d)
            for i in range(len(y)):
                j=i%d; v=y[i]+g*buf[j]; buf[j]=v; y[i]=v
        d=int(self.sr*0.005); ap=np.zeros(d); g=0.5
        for i in range(len(y)):
            j=i%d; xin=y[i]; v=-g*xin+ap[j]+g*self._reverb_z[0]
            ap[j]=xin+g*v; self._reverb_z[0]=v; y[i]=v
        return np.tanh(y*1.2)

# ---------------- SYNTH CORE ---------------- #
class Synth:
    def __init__(self,sr=48000,block=256):
        self.sr=sr; self.block=block; self.transport=False
        self.marbles=Marbles(sr); self.rings=ModalResonator(sr); self.clouds=GranularCloud(sr)
        self.master_db=-6.0; self.pan=0.0; self._event_text=deque(maxlen=8); self.note_history=deque(maxlen=64)
    def audio_callback(self,outdata,frames,time_info,status):
        if not self.transport: outdata[:]=0; return
        did,f0,vel=self.marbles.next_trigger_and_pitch()
        if did:
            self.rings.set_f0(f0); self.rings.excite(vel)
            self._event_text.appendleft(f"TRIG f0={f0:6.1f}Hz vel={vel:.2f}")
            try: midi=int(round(69+12*math.log2(max(1e-6,f0/440.0)))); self.note_history.append(midi)
            except: pass
        mono=self.rings.process(np.zeros(frames)); wet=self.clouds.process(mono)
        sig=np.nan_to_num(wet,nan=0,posinf=0,neginf=0); sig=np.clip(sig,-1,1)
        g=db_to_lin(self.master_db); l=math.cos((self.pan*0.5+0.5)*math.pi/2); r=math.sin((self.pan*0.5+0.5)*math.pi/2)
        outdata[:]=np.column_stack((sig*l*g,sig*r*g))

# ---------------- UI ---------------- #
class UI:
    def __init__(self,synth:Synth): self.s=synth; self.panel=0; self.sel=0; self.step=1; self._running=True
    def run(self):
        sd.default.samplerate=self.s.sr; sd.default.blocksize=self.s.block; sd.default.latency=('low','low')
        with sd.OutputStream(channels=2,callback=self.s.audio_callback): curses.wrapper(self._loop)
    def small(self,w,y,x,t,c=0):
        try:w.addstr(y,x,t,curses.color_pair(c))
        except:pass
    def bar(self,w,y,x,n,val,label):
        val=max(0,min(1,val)); f=int(val*n)
        w.addstr(y,x,label); w.addstr(y,x+len(label)+1,'[')
        for i in range(n):
            ch='█' if i<f else ' '; color=curses.color_pair(2 if i<f else 1)
            try:w.addstr(y,x+len(label)+2+i,ch,color)
            except:pass
        w.addstr(y,x+len(label)+2+n,']')
    def draw_staff(self,w,y,x,W,H,notes):
        pcs=["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]; rows=12; vis=min(rows,H-2); left=4; grid=max(1,W-left-2)
        for r in range(vis):
            lbl=pcs[(rows-1-r)%12]; self.small(w,y+r,x,f"{lbl:>3}")
            try:w.addstr(y+r,x+left,'·'*grid)
            except:pass
        last=notes[-grid:] if notes else []
        for ci,m in enumerate(last):
            if not isinstance(m,int): continue
            pc=m%12; oc=(m//12)%10; row=(rows-1-pc)
            if 0<=row<vis:
                col=x+left+(grid-len(last)+ci)
                try:w.addstr(y+row,col,str(oc),curses.color_pair(2))
                except:pass
        try:
            w.addstr(y+vis,x+left,'─'*grid)
            if last:
                latest=last[-1]; pcname=pcs[latest%12]; oc=latest//12; self.small(w,y+vis+1,x,f"Latest: {pcname}{oc}",5)
        except:pass
    def _loop(self,st):
        curses.curs_set(0); curses.start_color(); curses.use_default_colors()
        for i,col in enumerate([curses.COLOR_BLACK,curses.COLOR_CYAN,curses.COLOR_MAGENTA,curses.COLOR_GREEN,curses.COLOR_YELLOW,curses.COLOR_RED],1): curses.init_pair(i,col,-1)
        st.nodelay(True); last=0
        while self._running:
            now=time.time()
            if now-last>1/60:self.draw(st); last=now
            c=st.getch()
            if c!=-1:self.handle_key(c)
            time.sleep(0.001)
    def draw(self,w):
        w.erase(); h,W=w.getmaxyx()
        self.small(w,0,max(0,(W-40)//2),"Rings x Clouds x Marbles — Terminal Synth",5)
        self.small(w,1,2,f"Transport: {'RUN' if self.s.transport else 'STOP'}",4)
        self.draw_staff(w,2,2,W-4,min(16,max(8,h//3)),list(self.s.note_history))
        y=2+min(16,max(8,h//3))+1; self.small(w,y,2,"Events:",3)
        for i,t in enumerate(list(self.s._event_text)[:4]): self.small(w,y+i,12,t,2)
        y+=5; panels=["Marbles","Rings","Clouds","Mixer"]
        tabs='  '.join([f"[{p}]" if i==self.panel else p for i,p in enumerate(panels)]); self.small(w,y,2,tabs,5); y+=2
        getattr(self,f"draw_{panels[self.panel].lower()}")(w,y,W)
        self.small(w,h-1,2,"space run/stop  tab panel  arrows value  f pluck  c freeze  q quit",1); w.refresh()
    def draw_marbles(self,w,y,W):
        m=self.s.marbles; it=[("bpm",(m.bpm-40)/200,f"{m.bpm:5.1f}"),("jitter",m.jitter,None),("bias",m.bias,None),("spread",m.spread,None)]
        for i,(n,v,t) in enumerate(it): self.bar(w,y+i,4,30,v,f"{'> ' if self.sel==i else '  '}{n:7}"); 
        if t:self.small(w,y+i,40,t)
    def draw_rings(self,w,y,W):
        r=self.s.rings; 
        for i,(n,v) in enumerate([("struct",r.struct),("bright",r.brightness),("damping",r.damping),("pos",r.position),("strike",r.strike_gain)]): self.bar(w,y+i,4,30,v,f"{'> ' if self.sel==i else '  '}{n:7}")
    def draw_clouds(self,w,y,W):
        c=self.s.clouds; it=[("blend",c.blend,None),("size",min(1.0,c.size),f"{c.size*1000:.0f} ms"),("dens",c.density,None),("pitch",(c.pitch+12)/24,f"{c.pitch:+.1f} st"),("fb",c.feedback,None)]
        for i,(n,v,t) in enumerate(it): self.bar(w,y+i,4,30,v,f"{'> ' if self.sel==i else '  '}{n:7}"); 
        if t:self.small(w,y+i,40,t)
        self.small(w,y+len(it),4,f"freeze: {'ON' if c.freeze else 'OFF'}",4)
    def draw_mixer(self,w,y,W):
        it=[("master",(self.s.master_db+24)/36,f"{self.s.master_db:.1f} dB"),("pan",(self.s.pan+1)/2,f"{self.s.pan:+.2f}")]
        for i,(n,v,t) in enumerate(it): self.bar(w,y+i,4,30,v,f"{'> ' if self.sel==i else '  '}{n:7}"); 
        if t:self.small(w,y+i,40,t)
    def handle_key(self,c):
        if c in (ord('q'),27): self._running=False
        elif c==ord(' '): self.s.transport=not self.s.transport
        elif c==ord('\t'): self.panel=(self.panel+1)%4; self.sel=0
        elif c in (curses.KEY_LEFT,curses.KEY_RIGHT):
            n=[4,5,5,2][self.panel]; self.sel=(self.sel+(1 if c==curses.KEY_RIGHT else -1))%n
        elif c in (curses.KEY_UP,curses.KEY_DOWN):
            d=0.05*(1 if c==curses.KEY_UP else -1)
            if self.panel==0:
                m=self.s.marbles; 
                if self.sel==0: m.bpm=max(20,min(240,m.bpm+d*100)); m._interval=60.0/m.bpm
                elif self.sel==1: m.jitter=float(np.clip(m.jitter+d,0,0.5))
                elif self.sel==2: m.bias=float(np.clip(m.bias+d,0,1))
                elif self.sel==3: m.spread=float(np.clip(m.spread+d,0,1))
            elif self.panel==1:
                r=self.s.rings
                if self.sel==0:r.struct=float(np.clip(r.struct+d,0,1))
                elif self.sel==1:r.brightness=float(np.clip(r.brightness+d,0,1))
                elif self.sel==2:r.damping=float(np.clip(r.damping+d,0,1))
                elif self.sel==3:r.position=float(np.clip(r.position+d,0,1))
                elif self.sel==4:r.strike_gain=float(np.clip(r.strike_gain+d,0,1.5))
            elif self.panel==2:
                cl=self.s.clouds
                if self.sel==0:cl.blend=float(np.clip(cl.blend+d,0,1))
                elif self.sel==1:cl.size=float(np.clip(cl.size+d*0.1,0.02,0.75))
                elif self.sel==2:cl.density=float(np.clip(cl.density+d,0,1))
                elif self.sel==3:cl.pitch=float(np.clip(cl.pitch+d*12,-12,12))
                elif self.sel==4:cl.feedback=float(np.clip(cl.feedback+d,0,0.95))
            elif self.panel==3:
                if self.sel==0:self.s.master_db=float(np.clip(self.s.master_db+d*12,-24,12))
                elif self.sel==1:self.s.pan=float(np.clip(self.s.pan+d*2,-1,1))
        elif c==ord('f') or c==ord('F'): self.s.rings.excite(1.0)
        elif c==ord('r'): self.s.marbles.reseed()
        elif c==ord('g'): self.s.marbles.quantize=not self.s.marbles.quantize
        elif c in (ord(','),ord('.')): self.s.marbles.base_note+=-1 if c==ord(',') else 1
        elif c in (ord('1'),ord('2'),ord('3'),ord('4'),ord('5'),ord('6'),ord('7')): self.s.marbles.scale_id=int(chr(c))
        elif c==ord('c'):
            self.s.clouds.freeze=not self.s.clouds.freeze
            self.s._event_text.appendleft(f"Clouds FREEZE: {'ON' if self.s.clouds.freeze else 'OFF'}")
        elif c==ord('p'): self.s.marbles.tap()

# ---------------- MAIN ---------------- #
if __name__=="__main__":
    synth=Synth(sr=48000,block=256)
    ui=UI(synth)
    ui.run()

