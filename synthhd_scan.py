"""
synthhd_scan.py — Windfreak SynthHD Mini frequency scan + Keithley 2700 readout.

Controls a Windfreak SynthHD Mini RF synthesizer (serial, COM10) and reads a
Keithley 2700 DMM (serial, COM6) one point per frequency.

Workflow
--------
  * On launch the synth is preloaded with the start-up state (center =
    6675 MHz, power = 0 dBm) but its RF output stays OFF. Press the "RF: ON"
    button in the UI to actually emit; press it again to turn the output off.
  * Before a scan you can retune live: editing "Center (MHz)" immediately
    moves the synth to that center frequency, and editing "Power (dBm)"
    immediately updates the output power. This lets you park the synth on a
    feature and adjust power while watching the experiment.
  * "Start Scan" ramps the synth *slowly* down from the center to the minimum
    frequency (center - span/2), then steps upward in increments of the
    defined step size up to the maximum (center + span/2). At every frequency
    it waits the settle time, then sends the Keithley a single trigger to
    capture exactly one reading, waits for that reading to come back, and
    plots voltage vs frequency live.
  * "Stop Scan" aborts cleanly at the next point.
  * "Save CSV" writes the current trace (frequency_MHz, voltage_V).

Keithley acquisition (one point per frequency)
----------------------------------------------
The DMM is configured for a single triggered reading (:TRIG:COUN 1,
:SAMP:COUN 1, immediate trigger). Each frequency point issues one ``:READ?``
which arms, triggers, takes exactly one reading, and returns it — so the DMM
is *not* free-running; it produces one data point per frequency on demand.

SynthHD serial command reference (Windfreak)
--------------------------------------------
  f{MHz}   set RF frequency in MHz          e.g. f6675.000000
  W{dBm}   set RF power in dBm              e.g. W0.000
  C{0|1}   select control channel
  E{0|1}   PLL chip enable  (output stage)
  r{0|1}   RF power-amp (output) enable
  f? / W?  query current frequency / power
Enabling the output uses E1 then r1 (disable with E0 r0). If your unit needs a
different enable sequence, adjust ENABLE_ON / ENABLE_OFF below.

Usage
-----
    python synthhd_scan.py
    python synthhd_scan.py --synth-port COM10 --dmm-port COM6
"""

import argparse
import logging
import os
import re
import threading
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np
import serial

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYNTH_PORT = "COM10"          # Windfreak SynthHD Mini (USB Serial Device)
DMM_PORT = "COM6"             # Keithley 2700 (RS-232)

SYNTH_BAUD = 9600             # USB-CDC — baud is nominal but pyserial needs one
DMM_BAUD = 9600
TERMINATOR = "\r\n"           # Keithley line terminator

# SynthHD output enable / disable sequences (see command reference above).
ENABLE_ON = "E1r1"
ENABLE_OFF = "E0r0"

# Start-up / default UI values.
DEFAULT_CENTER_MHZ = 6675.0
DEFAULT_POWER_DBM = 0.0
DEFAULT_SPAN_MHZ = 100.0
DEFAULT_STEP_MHZ = 1.0
DEFAULT_SETTLE_S = 0.20       # dwell at each frequency before triggering the DMM

# Gentle ramp used to walk from the center down to the scan minimum so the
# synth (and whatever it drives) is not slammed with a big frequency jump.
RAMP_STEP_MHZ = 2.0
RAMP_DELAY_S = 0.02

# SynthHD Mini frequency limits (MHz) — used only for clamping.
SYNTH_MIN_MHZ = 12.5
SYNTH_MAX_MHZ = 6800.0

DMM_READ_TIMEOUT = 10.0       # seconds allowed for a single :READ?
PLOT_INTERVAL_MS = 150

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# Keithley single-point-per-trigger configuration.
DMM_INIT_COMMANDS = [
    "*RST",
    "*CLS",
    ":FUNC 'VOLT:DC'",
    ":VOLT:DC:RANG 10",
    ":VOLT:DC:NPLC 1",
    ":TRIG:SOUR IMM",     # immediate trigger — :READ? fires it
    ":TRIG:COUN 1",       # exactly one trigger per :READ?
    ":SAMP:COUN 1",       # one sample per trigger
    ":FORM:ELEM READ",    # return only the reading value
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


# ---------------------------------------------------------------------------
# Windfreak SynthHD Mini serial driver
# ---------------------------------------------------------------------------


class SynthHDMini:
    """Minimal serial driver for a Windfreak SynthHD Mini."""

    def __init__(self, port: str = SYNTH_PORT, timeout: float = 1.0) -> None:
        self.port = port
        self.ser = serial.Serial(
            port=port, baudrate=SYNTH_BAUD, timeout=timeout,
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE, xonxoff=False, rtscts=False,
        )
        time.sleep(0.2)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    # -- low-level I/O -----------------------------------------------------

    def _write(self, cmd: str) -> None:
        self.ser.write(cmd.encode("ascii"))

    def _query(self, cmd: str, timeout: float = 1.0) -> str:
        """Send a query (e.g. 'f?') and return the response line, stripped."""
        self.ser.reset_input_buffer()
        self.ser.write(cmd.encode("ascii"))
        prev = self.ser.timeout
        try:
            self.ser.timeout = timeout
            line = self.ser.read_until(b"\n")
            return line.decode("ascii", errors="replace").strip()
        finally:
            self.ser.timeout = prev

    # -- high-level operations --------------------------------------------

    def select_channel(self, channel: int = 0) -> None:
        self._write(f"C{int(channel)}")

    def set_frequency(self, mhz: float) -> None:
        mhz = max(SYNTH_MIN_MHZ, min(SYNTH_MAX_MHZ, float(mhz)))
        self._write(f"f{mhz:.6f}")

    def set_power(self, dbm: float) -> None:
        self._write(f"W{float(dbm):.3f}")

    def enable_output(self, on: bool) -> None:
        self._write(ENABLE_ON if on else ENABLE_OFF)

    def get_frequency(self) -> float:
        return _first_float(self._query("f?"))

    def get_power(self) -> float:
        return _first_float(self._query("W?"))

    def identify(self) -> str:
        try:
            return self._query("+")  # '+' = firmware version on SynthHD
        except Exception as exc:  # noqa: BLE001
            return f"<version query failed: {exc}>"

    def close(self) -> None:
        try:
            self.enable_output(False)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.ser.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Keithley 2700 serial driver (single triggered point)
# ---------------------------------------------------------------------------


class Keithley:
    """SCPI driver for a Keithley 2700 DMM over RS-232, one point per trigger."""

    def __init__(self, port: str = DMM_PORT, timeout: float = 2.0) -> None:
        self.port = port
        self.ser = serial.Serial(
            port=port, baudrate=DMM_BAUD, timeout=timeout,
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE, xonxoff=False, rtscts=False,
        )
        time.sleep(0.3)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def write(self, cmd: str) -> None:
        self.ser.write((cmd + TERMINATOR).encode("ascii"))

    def query(self, cmd: str, timeout: float = 2.0) -> str:
        """Send a query and return the decoded, stripped response.

        Terminator-agnostic (adapted from keithley_live.py): blocks up to
        *timeout* for the first byte, then drains until a CR/LF terminator and
        the input goes quiet — works whether the DMM terminates with CR, LF,
        or CR+LF.
        """
        self.ser.reset_input_buffer()
        self.ser.write((cmd + TERMINATOR).encode("ascii"))
        prev = self.ser.timeout
        try:
            self.ser.timeout = timeout
            first = self.ser.read(1)
            if not first:
                raise RuntimeError(f"No response to query: {cmd!r}")
            buf = bytearray(first)
            self.ser.timeout = 0.3
            while True:
                chunk = self.ser.read(self.ser.in_waiting or 1)
                if chunk:
                    buf += chunk
                    continue
                if buf.endswith(b"\r") or buf.endswith(b"\n"):
                    break
                break
            # The 2700 brackets replies with XON/XOFF flow-control bytes
            # (\x11 / \x13); drop them so error-code and float parsing is clean.
            text = buf.decode("ascii", errors="replace")
            text = text.replace("\x11", "").replace("\x13", "")
            return text.strip()
        finally:
            self.ser.timeout = prev

    def send_checked(self, cmd: str) -> None:
        """Send *cmd*, then read the error queue and log any complaint."""
        self.write(cmd)
        time.sleep(0.15)   # let the slow 9600-baud reply drain before we query
        try:
            err = self.query(":SYST:ERR?")
        except Exception as exc:  # noqa: BLE001
            log.warning("  %-22s -> :SYST:ERR? failed: %s", cmd, exc)
            return
        code = err.split(",", 1)[0].strip()
        if not code:
            # Empty error queue reply (e.g. queried mid-reset) — not an error.
            log.info("  %-22s -> ok", cmd)
        elif code not in ("0", "+0"):
            log.error("  %-22s -> %s", cmd, err)
        else:
            log.info("  %-22s -> ok", cmd)

    def initialise(self) -> None:
        log.info("Initialising Keithley for single triggered readings ...")
        # *RST/*CLS reset the instrument; querying :SYST:ERR? immediately after
        # is unreliable, so send them plainly and let the front end settle
        # before checking the remaining configuration commands.
        self.write("*RST")
        time.sleep(0.6)
        self.write("*CLS")
        time.sleep(0.2)
        self.ser.reset_input_buffer()
        for cmd in DMM_INIT_COMMANDS:
            if cmd in ("*RST", "*CLS"):
                continue
            self.send_checked(cmd)
        time.sleep(0.3)
        log.info("Keithley ready: %s", self.identify())

    def identify(self) -> str:
        try:
            return self.query("*IDN?")
        except Exception as exc:  # noqa: BLE001
            return f"<*IDN? failed: {exc}>"

    def read_point(self) -> float:
        """Trigger one reading and return it as a float (volts).

        :READ? arms the trigger model, fires the immediate trigger, takes
        exactly one sample, and returns it — one point per call.
        """
        raw = self.query(":READ?", timeout=DMM_READ_TIMEOUT)
        return _first_float(raw)

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:  # noqa: BLE001
            pass


def _first_float(text: str) -> float:
    """Return the first numeric value found in *text* (raises if none)."""
    m = _NUM_RE.search(text)
    if not m:
        raise RuntimeError(f"No number in response: {text!r}")
    return float(m.group())


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_state = {
    "synth": None,
    "dmm": None,
    "io_lock": threading.Lock(),     # guards all serial access
    "data_lock": threading.Lock(),   # guards the trace arrays
    "center": DEFAULT_CENTER_MHZ,
    "span": DEFAULT_SPAN_MHZ,
    "step": DEFAULT_STEP_MHZ,
    "power": DEFAULT_POWER_DBM,
    "settle": DEFAULT_SETTLE_S,
    "scanning": False,
    "stop": False,
    "rf_on": False,                  # RF output state (off until On is pressed)
    "status": "idle",
    "freq": None,                    # np.ndarray of frequencies (MHz)
    "volt": None,                    # np.ndarray of voltages (V)
    "data_version": 0,
}


def _set_synth_frequency(mhz: float) -> None:
    synth = _state["synth"]
    if synth is None:
        return
    with _state["io_lock"]:
        synth.set_frequency(mhz)


def _set_synth_power(dbm: float) -> None:
    synth = _state["synth"]
    if synth is None:
        return
    with _state["io_lock"]:
        synth.set_power(dbm)


def _set_rf_output(on: bool) -> None:
    synth = _state["synth"]
    if synth is None:
        return
    with _state["io_lock"]:
        synth.enable_output(on)
    _state["rf_on"] = on


def _ramp_frequency(start_mhz: float, end_mhz: float,
                    honour_stop: bool = True) -> None:
    """Slowly walk the synth frequency from *start_mhz* to *end_mhz*.

    Moves in steps of at most RAMP_STEP_MHZ with a short dwell between each so
    the synth is never slammed with a large jump. If *honour_stop* is True the
    ramp aborts early when a stop is requested (used on the way down); the
    ramp-back-to-center at the end passes False so it always completes.
    """
    if start_mhz == end_mhz:
        _set_synth_frequency(end_mhz)
        return
    n = max(1, int(np.ceil(abs(end_mhz - start_mhz) / RAMP_STEP_MHZ)))
    for f in np.linspace(start_mhz, end_mhz, n, endpoint=False):
        if honour_stop and _state["stop"]:
            return
        _set_synth_frequency(f)
        time.sleep(RAMP_DELAY_S)
    _set_synth_frequency(end_mhz)


def run_scan():
    """Background worker: ramp to the minimum then step up, reading the DMM."""
    synth = _state["synth"]
    dmm = _state["dmm"]
    if synth is None:
        _state["status"] = "no synth"
        return
    if dmm is None:
        _state["status"] = "no Keithley"
        return

    center = _state["center"]
    span = _state["span"]
    step = _state["step"]
    settle = _state["settle"]

    if span <= 0 or step <= 0:
        _state["status"] = "span/step must be > 0"
        return

    fmin = center - span / 2.0
    fmax = center + span / 2.0
    freqs = np.arange(fmin, fmax + step / 2.0, step)

    _state["scanning"] = True
    _state["stop"] = False

    # Fresh trace.
    with _state["data_lock"]:
        _state["freq"] = np.array([], dtype=float)
        _state["volt"] = np.array([], dtype=float)
        _state["data_version"] += 1

    last_f = center   # tracks where the synth is, so we can ramp back at the end
    try:
        # --- Slow ramp from center down to the scan minimum ---
        _state["status"] = f"ramping to {fmin:.3f} MHz ..."
        _ramp_frequency(center, fmin)
        last_f = fmin
        time.sleep(max(settle, 0.05))

        # --- Step through the band, one triggered DMM reading per point ---
        fbuf = []
        vbuf = []
        for i, f in enumerate(freqs):
            if _state["stop"]:
                _state["status"] = "stopped"
                break
            _set_synth_frequency(f)
            last_f = float(f)
            time.sleep(settle)                       # dwell so the physics settles
            with _state["io_lock"]:
                v = dmm.read_point()                 # trigger + wait for the value
            fbuf.append(float(f))
            vbuf.append(float(v))
            with _state["data_lock"]:
                _state["freq"] = np.array(fbuf)
                _state["volt"] = np.array(vbuf)
                _state["data_version"] += 1
            _state["status"] = (f"scan {i + 1}/{len(freqs)} — "
                                f"{f:.3f} MHz  {v:.5f} V")
        else:
            _state["status"] = f"done — {len(freqs)} pts"
    except Exception as exc:  # noqa: BLE001
        log.error("Scan failed: %s: %s", type(exc).__name__, exc)
        _state["status"] = f"error: {exc}"
    finally:
        # Slowly sweep the synth back to the center frequency (always completes,
        # even if the scan was stopped or errored).
        try:
            done_status = _state["status"]
            _state["status"] = f"ramping back to {center:.3f} MHz ..."
            _ramp_frequency(last_f, center, honour_stop=False)
            _state["status"] = done_status
        except Exception:  # noqa: BLE001
            pass
        _state["scanning"] = False
        _state["stop"] = False


def save_csv(filename: str) -> str:
    """Write the current trace (frequency_MHz, voltage_V) to CSV."""
    with _state["data_lock"]:
        f = _state["freq"]
        v = _state["volt"]
    if f is None or v is None or len(v) == 0:
        raise RuntimeError("no data to save — run a scan first")

    name = filename.strip() or time.strftime("synthhd_scan_%Y%m%d_%H%M%S")
    if not name.lower().endswith(".csv"):
        name += ".csv"
    if os.path.isabs(name):
        path = name
    else:
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = os.path.join(SAVE_DIR, name)
    path = os.path.abspath(path)

    np.savetxt(
        path, np.column_stack([f, v]), delimiter=",",
        header="frequency_MHz,voltage_V", comments="",
    )
    return path


# ---------------------------------------------------------------------------
# UI colours (matched to keithley_live.py / main.py)
# ---------------------------------------------------------------------------

BG        = "#1a1a1a"
PANEL_BG  = "#232323"
BORDER    = "#3a3a3a"
WHITE     = "#e8e8e8"
DIM       = "#888888"
CYAN      = "#00bfff"
RED       = "#ff6b6b"
GREEN     = "#00e676"
AMBER     = "#ffab00"
BTN_FACE  = "#2a2a2a"
BTN_HOVER = "#3a3a3a"


def _style_textbox(box):
    try:
        box.text_disp.set_color(WHITE)
        box.text_disp.set_fontsize(10)
    except AttributeError:
        pass


def _add_textbox(fig, label, x, y, w, h, initial):
    fig.text(x - 0.005, y + h / 2.0, label, color=DIM, fontsize=10,
             va="center", ha="right")
    ax = fig.add_axes([x, y, w, h])
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    box = mwidgets.TextBox(ax, "", initial=initial, color=BTN_FACE,
                           hovercolor=BTN_HOVER)
    _style_textbox(box)
    return box


def build_ui():
    fig = plt.figure(figsize=(12, 7.2), facecolor=BG)
    fig.canvas.manager.set_window_title("SynthHD Mini scan — Keithley readout")

    ax = fig.add_axes([0.09, 0.40, 0.87, 0.52])
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=DIM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.set_xlabel("Frequency (MHz)", color=DIM, fontsize=9)
    ax.set_ylabel("Voltage (V)", color=DIM, fontsize=9)
    ax.grid(True, color=BORDER, lw=0.4, alpha=0.5)
    line, = ax.plot([], [], color=CYAN, lw=1.2, marker=".", ms=4)

    fig.text(0.09, 0.965, "SynthHD Mini scan", color=WHITE, fontsize=12,
             fontweight="bold", va="top")
    fig.text(0.30, 0.967, "— voltage vs RF frequency", color=DIM, fontsize=10,
             va="top")
    range_text = fig.text(0.60, 0.967, "", color=CYAN, fontsize=10, va="top",
                          fontfamily="monospace")

    fig.add_artist(plt.Line2D([0.06, 0.96], [0.32, 0.32],
                              transform=fig.transFigure, color=BORDER, lw=0.8))

    # --- Row 1: scan parameters (live-editable) ---
    y1, h = 0.225, 0.05
    box_center = _add_textbox(fig, "Center (MHz):", 0.155, y1, 0.09, h,
                              f"{DEFAULT_CENTER_MHZ:g}")
    box_span   = _add_textbox(fig, "Span (MHz):",   0.35, y1, 0.07, h,
                              f"{DEFAULT_SPAN_MHZ:g}")
    box_step   = _add_textbox(fig, "Step (MHz):",   0.51, y1, 0.06, h,
                              f"{DEFAULT_STEP_MHZ:g}")
    box_power  = _add_textbox(fig, "Power (dBm):",  0.68, y1, 0.06, h,
                              f"{DEFAULT_POWER_DBM:g}")
    box_settle = _add_textbox(fig, "Settle (s):",   0.86, y1, 0.06, h,
                              f"{DEFAULT_SETTLE_S:g}")

    # --- Row 2: actions ---
    ax_start = fig.add_axes([0.09, 0.05, 0.14, 0.075])
    btn_start = mwidgets.Button(ax_start, "Start Scan", color=BTN_FACE,
                                hovercolor="#0d3020")
    btn_start.label.set_color(GREEN)
    btn_start.label.set_fontsize(11)

    ax_stop = fig.add_axes([0.25, 0.05, 0.14, 0.075])
    btn_stop = mwidgets.Button(ax_stop, "Stop Scan", color=BTN_FACE,
                               hovercolor="#301010")
    btn_stop.label.set_color(RED)
    btn_stop.label.set_fontsize(11)

    # RF output toggle — starts OFF; the synth is preloaded with the start-up
    # frequency and power but does not emit until this is pressed.
    ax_rf = fig.add_axes([0.44, 0.05, 0.13, 0.075])
    btn_rf = mwidgets.Button(ax_rf, "RF: OFF", color=BTN_FACE,
                             hovercolor="#301010")
    btn_rf.label.set_color(RED)
    btn_rf.label.set_fontsize(11)

    box_csv = _add_textbox(fig, "Save as:", 0.62, 0.055, 0.16, 0.06, "")

    ax_save = fig.add_axes([0.80, 0.05, 0.14, 0.075])
    btn_save = mwidgets.Button(ax_save, "Save CSV", color=BTN_FACE,
                               hovercolor="#30280d")
    btn_save.label.set_color(AMBER)
    btn_save.label.set_fontsize(11)

    status_text = fig.text(0.09, 0.16, "idle", color=DIM, fontsize=10,
                           va="center", fontfamily="monospace")

    return (fig, ax, line, range_text, box_center, box_span, box_step,
            box_power, box_settle, box_csv, btn_start, btn_stop, btn_rf,
            btn_save, status_text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="SynthHD Mini frequency scan with Keithley 2700 readout")
    parser.add_argument("--synth-port", default=SYNTH_PORT,
                        help=f"SynthHD serial port (default: {SYNTH_PORT})")
    parser.add_argument("--dmm-port", default=DMM_PORT,
                        help=f"Keithley serial port (default: {DMM_PORT})")
    args = parser.parse_args()

    # --- Connect the SynthHD ---
    log.info("Connecting to SynthHD Mini on %s ...", args.synth_port)
    try:
        synth = SynthHDMini(port=args.synth_port)
        synth.select_channel(0)
        synth.set_power(DEFAULT_POWER_DBM)
        synth.set_frequency(DEFAULT_CENTER_MHZ)
        synth.enable_output(False)   # preloaded but silent — press "RF: ON" to emit
        _state["synth"] = synth
        _state["rf_on"] = False
        log.info("SynthHD ready (fw %s): %.3f MHz @ %.2f dBm, output OFF "
                 "(press RF: ON to enable)",
                 synth.identify(), DEFAULT_CENTER_MHZ, DEFAULT_POWER_DBM)
    except serial.SerialException as exc:
        log.error("Could not open SynthHD on %s: %s", args.synth_port, exc)
        return
    except Exception as exc:  # noqa: BLE001
        log.error("SynthHD setup failed: %s", exc)
        return

    # --- Connect the Keithley (optional — synth tuning still works without) ---
    log.info("Connecting to Keithley on %s ...", args.dmm_port)
    dmm = None
    try:
        dmm = Keithley(port=args.dmm_port)
        dmm.initialise()
        _state["dmm"] = dmm
    except Exception as exc:  # noqa: BLE001
        # Close any half-open handle so COM6 isn't left locked for the next run.
        if dmm is not None:
            dmm.close()
        log.error("Keithley unavailable on %s (%s) — scanning disabled, "
                  "synth tuning still works.", args.dmm_port, exc)
        _state["dmm"] = None

    # --- Build UI ---
    (fig, ax, line, range_text, box_center, box_span, box_step, box_power,
     box_settle, box_csv, btn_start, btn_stop, btn_rf, btn_save,
     status_text) = build_ui()

    def _update_range_text():
        c, s = _state["center"], _state["span"]
        range_text.set_text(f"[{c - s / 2.0:.2f} .. {c + s / 2.0:.2f}] MHz")

    _update_range_text()

    # --- Live parameter callbacks ---
    def on_center(text):
        try:
            c = float(text)
        except ValueError:
            log.warning("Center %r not a number", text)
            box_center.set_val(f"{_state['center']:g}")
            return
        c = max(SYNTH_MIN_MHZ, min(SYNTH_MAX_MHZ, c))
        _state["center"] = c
        _update_range_text()
        if _state["scanning"]:
            log.info("Center stored (%.3f MHz) — applies to next scan", c)
        else:
            _set_synth_frequency(c)
            _state["status"] = f"synth @ {c:.3f} MHz"
            log.info("Synth moved to center %.3f MHz", c)

    def on_span(text):
        try:
            _state["span"] = max(0.0, float(text))
        except ValueError:
            log.warning("Span %r not a number", text)
        box_span.set_val(f"{_state['span']:g}")
        _update_range_text()

    def on_step(text):
        try:
            _state["step"] = max(0.0, float(text))
        except ValueError:
            log.warning("Step %r not a number", text)
        box_step.set_val(f"{_state['step']:g}")

    def on_power(text):
        try:
            p = float(text)
        except ValueError:
            log.warning("Power %r not a number", text)
            box_power.set_val(f"{_state['power']:g}")
            return
        _state["power"] = p
        _set_synth_power(p)
        _state["status"] = f"power {p:.2f} dBm"
        log.info("Synth power set to %.2f dBm", p)

    def on_settle(text):
        try:
            _state["settle"] = max(0.0, float(text))
        except ValueError:
            log.warning("Settle %r not a number", text)
        box_settle.set_val(f"{_state['settle']:g}")

    box_center.on_submit(on_center)
    box_span.on_submit(on_span)
    box_step.on_submit(on_step)
    box_power.on_submit(on_power)
    box_settle.on_submit(on_settle)

    # --- Buttons ---
    def on_start(_):
        if _state["scanning"]:
            log.warning("Scan already running")
            return
        if _state["dmm"] is None:
            _state["status"] = "no Keithley — cannot scan"
            log.warning("Cannot scan: Keithley not connected")
            return
        threading.Thread(target=run_scan, daemon=True).start()

    def on_stop(_):
        if _state["scanning"]:
            _state["stop"] = True
            _state["status"] = "stopping ..."

    def on_save(_):
        try:
            path = save_csv(box_csv.text)
        except Exception as exc:  # noqa: BLE001
            log.error("Save failed: %s", exc)
            _state["status"] = f"error: {exc}"
            return
        _state["status"] = f"saved {os.path.basename(path)}"
        log.info("Saved CSV: %s", path)

    def on_rf(_):
        turn_on = not _state["rf_on"]
        try:
            _set_rf_output(turn_on)
        except Exception as exc:  # noqa: BLE001
            log.error("RF toggle failed: %s", exc)
            _state["status"] = f"error: {exc}"
            return
        if turn_on:
            btn_rf.label.set_text("RF: ON")
            btn_rf.label.set_color(GREEN)
            btn_rf.hovercolor = "#0d3020"
            _state["status"] = "RF output ON"
        else:
            btn_rf.label.set_text("RF: OFF")
            btn_rf.label.set_color(RED)
            btn_rf.hovercolor = "#301010"
            _state["status"] = "RF output OFF"
        fig.canvas.draw_idle()
        log.info("RF output %s", "ON" if turn_on else "OFF")

    btn_start.on_clicked(on_start)
    btn_stop.on_clicked(on_stop)
    btn_rf.on_clicked(on_rf)
    btn_save.on_clicked(on_save)

    # --- Animation: poll shared state, redraw on new data ---
    drawn = {"version": -1}

    def update(_frame):
        status_text.set_text(_state["status"])
        status_text.set_color(
            AMBER if _state["scanning"] else
            (RED if _state["status"].startswith("error") else GREEN))

        if _state["data_version"] != drawn["version"]:
            with _state["data_lock"]:
                f = _state["freq"]
                v = _state["volt"]
                drawn["version"] = _state["data_version"]
            if f is not None and v is not None and len(v) > 0:
                line.set_data(f, v)
                ax.relim()
                ax.autoscale_view()
                vmin, vmax = float(np.min(v)), float(np.max(v))
                pad = max(1e-6, 0.05 * (vmax - vmin))
                ax.set_ylim(vmin - pad, vmax + pad)
                if len(f) > 1 and f[-1] != f[0]:
                    ax.set_xlim(float(np.min(f)), float(np.max(f)))
            else:
                line.set_data([], [])
            fig.canvas.draw_idle()
        return (line, status_text)

    ani = animation.FuncAnimation(
        fig, update, interval=PLOT_INTERVAL_MS, blit=False,
        cache_frame_data=False)

    plt.show()

    # --- Cleanup ---
    _state["stop"] = True
    if _state["synth"] is not None:
        _state["synth"].close()
    if _state["dmm"] is not None:
        _state["dmm"].close()
    log.info("Done.")


if __name__ == "__main__":
    main()
