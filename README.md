# synthhd-keithley-scan

Sweep the frequency of a **Windfreak SynthHD Mini** RF synthesizer and read one
voltage from a **Keithley 2700** DMM at each step. Voltage vs. frequency is
plotted live in a small matplotlib GUI.

## Hardware

| Instrument | Interface | Default port |
|---|---|---|
| Windfreak SynthHD Mini | USB serial | `COM10` |
| Keithley 2700 DMM | RS-232, 9600 baud | `COM6` |

## Install

```
pip install -r requirements.txt
```

## Run

```
python synthhd_scan.py
python synthhd_scan.py --synth-port COM10 --dmm-port COM6
```

## Usage

- On launch the synth is set to 6675 MHz / 0 dBm with **RF output off**.
  Press **RF: ON** to emit.
- While no scan is running, editing **Center (MHz)** or **Power (dBm)** retunes
  the synth immediately.
- **Start Scan** ramps slowly from the center down to `center - span/2`, then
  steps up to `center + span/2`. At each point it waits the settle time and
  takes exactly one triggered Keithley reading (DC volts, 10 V range, 1 NPLC).
  When finished or stopped, it ramps back to the center.
- **Stop Scan** aborts at the next point.
- **Save CSV** writes `frequency_MHz,voltage_V` next to the script.

If the Keithley is not connected, the GUI still opens for manual synth tuning,
but scanning is disabled.
