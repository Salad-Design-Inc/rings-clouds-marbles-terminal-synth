# 🎛️ Rings × Clouds × Marbles — Terminal Synth
_A text-based homage to Mutable Instruments classics — built for macOS and iTerm2._

So far this is just a proof of concept, the audio quality and abilities need quite a bit of work. 

<img width="826" height="483" alt="Screenshot 2025-10-07 at 4 30 24 PM" src="https://github.com/user-attachments/assets/62f92e4e-86e4-4eee-99bf-c5ba3d15368d" />

---

## Overview

This project recreates the *spirit* of the iconic Mutable Instruments trio — **Rings**, **Clouds**, and **Marbles** — as a colorful, interactive **terminal synthesizer** that runs entirely in your Mac’s command line.

It’s not a DSP port or an emulation — it’s a minimalist re-imagining using Python and `sounddevice`, designed for immediacy, visual feedback, and fun.

The synth combines:
- **Marbles** for clocking & random melody generation  
- **Rings** for modal resonator tones  
- **Clouds** for granular reverb and spatial texture  

All within a vibrant, ASCII-driven UI.

---

## 🎹 Controls

| Key | Function |
|-----|-----------|
| `space` | Start / Stop transport |
| `tab` | Switch panel (Marbles / Rings / Clouds / Mixer) |
| `←` `→` | Select parameter |
| `↑` `↓` | Adjust parameter |
| `[` `]` | Change step size |
| `r` | Reseed Marbles |
| `g` | Toggle quantization |
| `1`..`7` | Choose musical scale |
| `,` `.` | Shift base note down / up |
| `c` | Toggle Clouds FREEZE |
| `f` / `F` | Manual pluck (Rings excite) |
| `p` | Tap tempo |
| `q` / `esc` | Quit |

---

## ⚙️ Installation

### Requirements
- macOS with **Python 3.9+**
- [Homebrew](https://brew.sh)
- iTerm2 recommended (TrueColor + Metal rendering)

### Install Steps
```bash
cd ~/Music
git clone https://github.com/salad-design-inc/rings-clouds-marbles-terminal-synth.git
cd rings-clouds-marbles-terminal-synth
python3 -m venv .venv
source .venv/bin/activate
brew install portaudio
pip install numpy sounddevice
python3 rings-clouds-marbles-terminal-synth.py
🧩 One-Line Install
You can also copy-paste this single command to set everything up automatically:

bash
Copy code
bash <(curl -s https://raw.githubusercontent.com/salad-design-inc/rings-clouds-marbles-terminal-synth/main/install.sh)
(Optional: add an install.sh script later — it’ll automate all the steps above.)

🧩 Credits & Licensing
This terminal synth is an independent reinterpretation inspired by the open-source work of Emilie Gillet / Mutable Instruments.
It is not affiliated with or endorsed by Mutable Instruments.

Original Mutable Instruments documentation:

Rings

Clouds

Marbles

Mutable Instruments’ source code is available under the MIT License.

This project’s source is also released under the MIT License, with gratitude and respect for the open-hardware community that made the originals possible.

✳️ About
Salad Design, Inc.
A collective of creators, builders, and sound lovers designing bold, beautiful audio tools that bridge high craft and low-key rebellion.

🌐 salad.design

🌀 Crafted in Washington State

💡 Inspired by mid-century modern philosophy and analog warmth

“This patch lives in text — but sounds alive.”
