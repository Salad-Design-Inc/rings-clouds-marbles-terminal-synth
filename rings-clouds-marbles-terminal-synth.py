#!/usr/bin/env python3
"""
Rings x Clouds x Marbles — Terminal Synth (staff view, crash-proof)
-------------------------------------------------------------------
- Replaces oscilloscope with a musical staff showing recent notes.
- Staff: rows = C..B pitch classes, columns = recent notes (rightmost = latest).
- Each plotted note shows its octave digit (A4 -> "4" on row A).

Controls:
  space  run/stop    tab  panel switch
  arrows select/adjust    [ / ]  smaller/larger step
  r reseed    g quantize on/off    1..7 choose scale
  , / . base note down/up          p tap tempo
  c FREEZE (Clouds)
  f / F pluck now (auto-starts transport)
  q quit

Requires: pip install numpy sounddevice   |   brew install portaudio
"""

import math, time, random, curses
from collections import deque
import numpy as np
import sounddevice as sd

def db_to_lin(db: float) -> float:
    return 10 ** (db / 20.0)

SCALES = {
    1: ("Ionian (Major)", [0,2,4,5,7,9,11]),
    2: ("Dorian", [0,2,3,5,7,9,10]),
    3: ("Phrygian",[0,1,3,5,7,8,10]),
    4: ("Lydian",[0,2,4,6,7,9,11]),
    5: ("Mixolydian",[0,2,4,5,7,9,10]),
    6: ("Aeolian (Minor)",[0,2,3,5,7,8,10]),
    7: ("Locrian",[0,1,3,5,6,8,10]),
}
A4 = 440.0

# ------------------------------- Marbles-ish ------------------------------- #
class Marbles:
    def __init__(self, sr):
        self.sr = sr
        self.bpm = 90.0
        self.jitter = 0.05
        self.bias = 0.5
        self.spread = 0.5
        self.quantize = True
        self.scale_id = 6
        self.base_note = 57  # A3
        self._interval = 60.0 / self.bpm
        self._last = time.time()
        self._seq = deque(maxlen=32)
        self._tap_times = deque(maxlen=8)
        self._rng = random.Random(1234)

    def reseed(self):
        self._rng.seed(self._rng.random()*1e9); self._seq.clear()

    def tap(self):
        now = time.time(); self._tap_times.append(now)
        if len(self._tap_times) >= 2:
            ivals = [t2-t1 for t1,t2 in zip(self._tap_times, list(self._tap_times)[1:])]
            avg = sum(ivals)/len(ivals)
            if 0.15 < avg < 3.0:
                self.bpm = 60.0/avg; self._interval = avg

    def _quantize_midi(self, midi):
        if not self.quantize: return midi
        _, scale = SCALES[self.scale_id]
        octave = int(math.floor(midi/12))
        cands = [octave*12+s for s in scale] + [(octave+1)*12+s for s in scale]
        return min(cands, key=lambda x: abs(x-midi))

    def next_trigger_and_pitch(self):
        now = time.time()
        did, freq, vel = False, None, 0.0
        if now - self._last >= self._interval * (1 + (self._rng.random()*2-1)*self.jitter):
            self._last = now; did = True
            if self._rng.random()<0.8 and self._seq:
                midi = self._rng.choice(list(self._seq))
            else:
                center = self.base_note + (12*(self.bias-0.5)*1.5)
                width = 12*(0.5+self.spread*1.5)
                midi = center + self._rng.gauss(0, width*0.3)
            midi = self._quantize_midi(midi); self._seq.append(midi)
            freq = 440.0*(2**((midi-69)/12))
            vel = min(1.0, max(0.05, 0.8 + self._rng.uniform(-0.2, 0.2)))
        return did, freq, vel

# -------------------------------- Rings-ish -------------------------------- #
class ModalResonator:
    def __init__(self, sr, n_modes=8):
        self.sr = sr; self.n = n_modes
        self.struct = 0.5; self.brightness = 0.6; self.damping = 0.4; self.position = 0.5
        self.strike_gain = 1.0
        self._m_freqs = np.zeros(self.n); self._m_q = np.ones(self.n)*200
        self._m_state = np.zeros((self.n,2))
        self._exc_buf = np.zeros(256)
        self._target_f0 = 110.0; self._f0 = 110.0
    def set_f0(self, f): self._target_f0 = max(20.0, min(2000.0, float(f)))
    def _update_modes(self):
        self._f0 += (self._target_f0 - self._f0)*0.05; base = self._f0
        for i in range(self.n):
            mult = (i+1)*(1+self.struct*0.02*i)
            self._m_freqs[i] = base*mult
            q = 100 + 1200*self.brightness; q *= (1.0 - 0.7*self.damping)
            self._m_q[i] = max(50, q)
    def excite(self, velocity=0.7):
        L = len(self._exc_buf); n = np.random.randn(L)
        alpha = 0.3 + 0.6*self.position
        for i in range(1, L): n[i] = alpha*n[i] + (1-alpha)*n[i-1]
        n *= np.hanning(L) * (0.4+0.6*self.brightness); n[0] += 1.0
        self._exc_buf[:] = n * self.strike_gain * velocity
    def process(self, in_block):
        self._update_modes(); out = np.zeros_like(in_block)
        take = min(len(in_block), len(self._exc_buf))
        if take>0:
            out[:take] += self._exc_buf[:take]; self._exc_buf = np.roll(self._exc_buf, -take); self._exc_buf[-take:] = 0.0
        for i in range(self.n):
            f = min(self._m_freqs[i], self.sr*0.45)
            if f<20: continue
            q = self._m_q[i]; g = math.tan(math.pi*f/self.sr); R = 1.0/q
            z1, z2 = self._m_state[i]
            for j, x in enumerate(out):
                v1 = (x - R*z1 - z2) * (g / (1 + R*g + g*g))
                bp = g*v1 + z1; z1 = g*v1 + bp; z2 = g*bp + z2
                out[j] = bp
            self._m_state[i,0], self._m_state[i,1] = z1, z2
        return np.tanh(out*1.5)

# -------------------------------- Clouds-ish ------------------------------- #
class GranularCloud:
    def __init__(self, sr, max_time=2.0):
        self.sr = sr; self.size = 0.25; self.density = 0.5; self.pitch = 0.0
        self.feedback = 0.3; self.blend = 0.35; self.freeze = False
        self._buf_len = int(max_time*sr); self._buf = np.zeros(self._buf_len); self._w = 0
        self._rng = random.Random(7); self._reverb_z = np.zeros(4)
    def process(self, block):
        out = np.copy(block)
        if not self.freeze:
            L = len(block)
            for i in range(L):
                self._buf[self._w] = 0.5*block[i] + self.feedback*self._buf[self._w]
                self._w = (self._w + 1) % self._buf_len
        grains = np.zeros_like(block)
        hop = max(1, int((1.0 - self.density*0.9)*len(block)))
        for start in range(0, len(block), hop):
            if self._rng.random() < self.density:
                delay = int(self.size*self.sr*(0.5 + self._rng.random()))
                r = (self._w - delay) % self._buf_len
                glen = min(int(self.size*self.sr), len(block) - start)
                win = np.hanning(glen); ratio = 2 ** (self.pitch/12.0)
                idx = (np.arange(glen)*ratio).astype(int); idx = np.clip(idx, 0, glen-1)
                src = np.array([self._buf[(r+k)%self._buf_len] for k in range(glen)])
                grains[start:start+glen] += src[idx] * win
        wet = self._reverb(grains)
        return (1-self.blend)*block + self.blend*wet
    def _reverb(self, x):
        y = np.copy(x)
        comb_delays = [int(self.sr*t) for t in (0.0297, 0.0371, 0.0411)]
        comb_g = [0.77, 0.74, 0.73]
        for d,g in zip(comb_delays, comb_g):
            buf = np.zeros(d)
            for i in range(len(y)):
                j = i % d; v = y[i] + g*buf[j]; buf[j] = v; y[i] = v
        d = int(self.sr*0.005); ap = np.zeros(d); g = 0.5
        for i in range(len(y)):
            j = i % d; xin = y[i]
            v = -g*xin + ap[j] + g*self._reverb_z[0]
            ap[j] = xin + g*v; self._reverb_z[0] = v; y[i] = v
        return np.tanh(y*1.2)

# --------------------------------- Synth / UI ------------------------------ #
class Synth:
    def __init__(self, sr=48000, block=256):
        self.sr = sr; self.block = block; self.transport = False
        self.marbles = Marbles(sr); self.rings = ModalResonator(sr); self.clouds = GranularCloud(sr)
        self.master_db = -6.0; self.pan = 0.0
        self._event_text = deque(maxlen=8)
        self.note_history = deque(maxlen=64)  # for staff

    def audio_callback(self, outdata, frames, time_info, status):
        if not self.transport:
            outdata[:] = 0; return
        did, f0, vel = self.marbles.next_trigger_and_pitch()
        if did:
            self.rings.set_f0(f0); self.rings.excite(vel)
            self._event_text.appendleft(f"TRIG f0={f0:6.1f}Hz vel={vel:.2f}")
            try:
                midi = int(round(69 + 12*math.log2(max(1e-6, f0/440.0))))
                self.note_history.append(midi)
            except Exception:
                pass
        mono = self.rings.process(np.zeros(frames))
        wet = self.clouds.process(mono)
        sig = wet
        # sanitize any NaNs/Infs before output
        if isinstance(sig, np.ndarray):
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        g = db_to_lin(self.master_db)
        l = math.cos((self.pan*0.5+0.5)*math.pi/2); r = math.sin((self.pan*0.5+0.5)*math.pi/2)
        out_st = np.column_stack((sig*l*g, sig*r*g))
        outdata[:] = out_st

class UI:
    def __init__(self, synth: Synth):
        self.s = synth; self.panel = 0; self.sel = 0; self.step = 1; self._running = True
    def run(self):
        sd.default.samplerate = self.s.sr
        sd.default.blocksize = self.s.block
        sd.default.latency = ('low','low')
        # If you needed a specific output device id, set it here:
        # sd.default.device = 3
        with sd.OutputStream(channels=2, callback=self.s.audio_callback):
            curses.wrapper(self._loop)

    # ---------- draw helpers ---------- #
    def bar(self, win, y, x, w, val, label):
        val = max(0.0, min(1.0, val)); n = int(val*w)
        win.addstr(y, x, label); win.addstr(y, x+len(label)+1, '[')
        for i in range(w):
            ch = '█' if i < n else ' '; color = curses.color_pair(2 if i<n else 1)
            try: win.addstr(y, x+len(label)+2+i, ch, color)
            except: pass
        win.addstr(y, x+len(label)+2+w, ']')

    def small(self, win, y, x, text, color=0):
        try: win.addstr(y, x, text, curses.color_pair(color))
        except: pass

    def draw_staff(self, win, y, x, w, h, notes):
        """Chromatic staff: rows C..B, columns recent notes; draw octave digits."""
        if w <= 10 or h < 6: return
        pcs = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]; rows = 12
        vis_rows = min(rows, h-2); left_lab_w = 4; grid_w = max(1, w-left_lab_w-2)
        # rows and baselines
        for r in range(vis_rows):
            label = pcs[(rows-1-r) % 12]
            self.small(win, y+r, x, f"{label:>3}")
            try: win.addstr(y+r, x+left_lab_w, '·'*grid_w)
            except: pass
        # map notes to right-aligned columns
        last = notes[-grid_w:] if notes else []
        for ci, m in enumerate(last):
            if not isinstance(m, int): continue
            pc = m % 12; octv = (m // 12) % 10
            row = (rows-1-pc)
            if 0 <= row < vis_rows:
                col = x + left_lab_w + (grid_w - len(last) + ci)
                try: win.addstr(y+row, col, str(octv), curses.color_pair(2))
                except: pass
        # axis + latest
        try:
            win.addstr(y+vis_rows, x+left_lab_w, '─'*grid_w)
            if last:
                latest = last[-1]; pcname = pcs[latest % 12]; oc = latest // 12
                self.small(win, y+vis_rows+1, x, f"Latest: {pcname}{oc}", 5)
        except: pass

    # ---------- main loop ---------- #
    def _loop(self, stdscr):
        curses.curs_set(0); curses.start_color(); curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLACK, -1); curses.init_pair(2, curses.COLOR_CYAN, -1)
        curses.init_pair(3, curses.COLOR_MAGENTA, -1); curses.init_pair(4, curses.COLOR_GREEN, -1)
        curses.init_pair(5, curses.COLOR_YELLOW, -1); curses.init_pair(6, curses.COLOR_RED, -1)
        stdscr.nodelay(True); last_draw = 0
        while self._running:
            now = time.time()
            if now - last_draw > 1/60:
                self.draw(stdscr); last_draw = now
            c = stdscr.getch()
            if c != -1: self.handle_key(c)
            time.sleep(0.001)

    def draw(self, win):
        win.erase(); h, w = win.getmaxyx()
        title = "Rings x Clouds x Marbles — Terminal Synth (Staff View)"
        self.small(win, 0, max(0, (w-len(title))//2), title, 5)
        self.small(win, 1, 2, f"Transport: {'RUN' if self.s.transport else 'STOP'}  |  SR {self.s.sr}  Block {self.s.block}", 4)

        # STAFF replaces scope
        staff_h = min(16, max(8, h//3))
        self.draw_staff(win, 2, 2, w-4, staff_h, list(self.s.note_history))

        # Events
        y = 2 + staff_h + 1
        self.small(win, y, 2, "Events:", 3)
        for i, t in enumerate(list(self.s._event_text)[:4]): self.small(win, y+i, 12, t, 2)
        y += 5

        # Panels
        panels = ["Marbles", "Rings", "Clouds", "Mixer"]
        tabs = '  '.join([f"[{p}]" if i==self.panel else p for i,p in enumerate(panels)])
        self.small(win, y, 2, tabs, 5); y += 2
        if self.panel == 0: self.draw_marbles(win, y, w)
        elif self.panel == 1: self.draw_rings(win, y, w)
        elif self.panel == 2: self.draw_clouds(win, y, w)
        else: self.draw_mixer(win, y, w)

        foot = "space run/stop  tab panel  arrows value  [/ ] step  r reseed  g quant  1-7 scale  ,/. base  c freeze  f/F pluck  p tap  q quit"
        self.small(win, h-1, 2, foot, 1); win.refresh()

    def draw_marbles(self, win, y, w):
        m = self.s.marbles
        items = [("bpm", (m.bpm-40)/200, f"{m.bpm:5.1f}"),
                 ("jitter", m.jitter, None),
                 ("bias", m.bias, None),
                 ("spread", m.spread, None)]
        for i,(name,val,txt) in enumerate(items):
            self.bar(win, y+i, 4, 30, val, f"{'> ' if self.sel==i else '  '}{name:7}")
            if txt: self.small(win, y+i, 40, txt)
        y += len(items); name,_ = SCALES[m.scale_id]
        self.small(win, y, 4, f"quantize: {'ON' if m.quantize else 'OFF'} | scale: {name}")
        self.small(win, y+1, 4, f"base note: {m.base_note} (MIDI) — {440.0*(2**((m.base_note-69)/12)):.1f} Hz")

    def draw_rings(self, win, y, w):
        r = self.s.rings
        items = [("struct", r.struct, None),
                 ("bright", r.brightness, None),
                 ("damping", r.damping, None),
                 ("pos", r.position, None),
                 ("strike", r.strike_gain, None)]
        for i,(name,val,txt) in enumerate(items):
            self.bar(win, y+i, 4, 30, val, f"{'> ' if self.sel==i else '  '}{name:7}")

    def draw_clouds(self, win, y, w):
        c = self.s.clouds
        items = [("blend", c.blend, None),
                 ("size", min(1.0,c.size), f"{c.size*1000:.0f} ms"),
                 ("dens", c.density, None),
                 ("pitch", (c.pitch+12)/24, f"{c.pitch:+.1f} st"),
                 ("fb", c.feedback, None)]
        for i,(name,val,txt) in enumerate(items):
            self.bar(win, y+i, 4, 30, val, f"{'> ' if self.sel==i else '  '}{name:7}")
            if txt: self.small(win, y+i, 40, txt)
        self.small(win, y+len(items), 4, f"freeze: {'ON' if c.freeze else 'OFF'} (press 'c')")

    def draw_mixer(self, win, y, w):
        items = [("master", (self.s.master_db+24)/36, f"{self.s.master_db:.1f} dB"),
                 ("pan", (self.s.pan+1)/2, f"{self.s.pan:+.2f}")]
        for i,(name,val,txt) in enumerate(items):
            self.bar(win, y+i, 4, 30, val, f"{'> ' if self.sel==i else '  '}{name:7}")
            if txt: self.small(win, y+i, 40, txt)

    def handle_key(self, c):
        if c in (ord('q'),27): self._running=False
        elif c == ord(' '): self.s.transport = not self.s.transport
        elif c == ord('\t'): self.panel=(self.panel+1)%4; self.sel=0
        elif c in (curses.KEY_LEFT, curses.KEY_RIGHT):
            n=[4,5,5,2][self.panel]
            self.sel = (self.sel - 1) % n if c==curses.KEY_LEFT else (self.sel + 1) % n
        elif c in (curses.KEY_UP, curses.KEY_DOWN, ord('['), ord(']')):
            fine=0.01; coarse=0.05
            if c==ord('['): self.step=max(1,self.step-1); return
            if c==ord(']'): self.step=min(10,self.step+1); return
            mult=self.step; d=(fine if self.panel!=0 else coarse) * (1 if c==curses.KEY_UP else -1) * mult
            if self.panel==0:
                m=self.s.marbles
                if self.sel==0: m.bpm=max(20,min(240,m.bpm + d*100)); m._interval=60.0/m.bpm
                elif self.sel==1: m.jitter=float(np.clip(m.jitter+d,0,0.5))
                elif self.sel==2: m.bias=float(np.clip(m.bias+d,0,1))
                elif self.sel==3: m.spread=float(np.clip(m.spread+d,0,1))
            elif self.panel==1:
                r=self.s.rings
                if self.sel==0: r.struct=float(np.clip(r.struct+d,0,1))
                elif self.sel==1: r.brightness=float(np.clip(r.brightness+d,0,1))
                elif self.sel==2: r.damping=float(np.clip(r.damping+d,0,1))
                elif self.sel==3: r.position=float(np.clip(r.position+d,0,1))
                elif self.sel==4: r.strike_gain=float(np.clip(r.strike_gain+d,0,1.5))
            elif self.panel==2:
                cld=self.s.clouds
                if self.sel==0: cld.blend=float(np.clip(cld.blend+d,0,1))
                elif self.sel==1: cld.size=float(np.clip(cld.size+d*0.1,0.02,0.75))
                elif self.sel==2: cld.density=float(np.clip(cld.density+d,0,1))
                elif self.sel==3: cld.pitch=float(np.clip(cld.pitch+d*12,-12,12))
                elif self.sel==4: cld.feedback=float(np.clip(cld.feedback+d,0,0.95))
            else:
                if self.sel==0: self.s.master_db=float(np.clip(self.s.master_db + d*12, -24, 12))
                elif self.sel==1: self.s.pan=float(np.clip(self.s.pan + d*2, -1, 1))
        elif c==ord('r'): self.s.marbles.reseed()
        elif c==ord('g'): self.s.marbles.quantize = not self.s.marbles.quantize
        elif c in (ord(','), ord('.')): self.s.marbles.base_note += -1 if c==ord(',') else 1
        elif c in (ord('1'),ord('2'),ord('3'),ord('4'),ord('5'),ord('6'),ord('7')): self.s.marbles.scale_id=int(chr(c))
        elif c==ord('c'): self.s.clouds.freeze = not self.s.clouds.freeze
        elif c in (ord('f'), ord('F')):
            if not self.s.transport: self.s.transport=True
            self.s.rings.excite(1.0); self.s._event_text.appendleft("PLUCK vel=1.00")
        elif c==ord('p'): self.s.marbles.tap()

# --------------------------------- Main ------------------------------------ #
if __name__ == '__main__':
    synth = Synth(sr=48000, block=256)
    ui = UI(synth)
    ui.run()
