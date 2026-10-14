import math
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

st.set_page_config(page_title="GestureHome", page_icon="🏠", layout="wide")

# ----------------------------------------------------------------------------
# CONFIG - every threshold lives here with a one-line comment
# ----------------------------------------------------------------------------
CONFIG = {
    "arm_hold_s": 1.0,         # open palm must be held this long to arm
    "arm_timeout_s": 8.0,      # armed state expires if no selection starts
    "select_timeout_s": 5.0,   # pointing must finish selecting within this time
    "select_hold_s": 0.8,      # point-and-hold time to select a device
    "adjust_timeout_s": 10.0,  # adjust state expires after this much inactivity
    "confirm_timeout_s": 6.0,  # risky change is dropped if not confirmed in time
    "confirm_hold_s": 0.5,     # thumbs-up hold time to confirm
    "max_changes_per_s": 5,    # rate limit for value changes
    "pinch_min": 0.15,         # pinch ratio mapped to 0 %
    "pinch_max": 1.20,         # pinch ratio mapped to 100 %
    "refractory_s": 0.8,       # lockout after a disarm so one motion = one action
    "cancel_block_s": 1.0,     # ignore pinch right after a cancelled risky change
    "readback_timeout_s": 2.0, # no state report within this time = no response
    "device_delay_s": (0.08, 0.25),  # simulated device network latency range
    "tick_s": 0.3,             # simulation / UI refresh period
    "oe_mincutoff": 1.2,       # One Euro filter: baseline smoothing
    "oe_beta": 0.4,            # One Euro filter: speed responsiveness
    "oe_dcutoff": 1.0,         # One Euro filter: derivative smoothing
}

DEVICES = {
    "lamp":    {"label": "Lamp",    "icon": "💡", "max": 100, "step": 5, "unit": "%", "action": "set_level",  "risk_at": None},
    "fan":     {"label": "Fan",     "icon": "🌀", "max": 5,   "step": 1, "unit": "",  "action": "set_speed",  "risk_at": 5},
    "speaker": {"label": "Speaker", "icon": "🔊", "max": 100, "step": 5, "unit": "%", "action": "set_volume", "risk_at": 80},
}
INITIAL = {"lamp": 80, "fan": 2, "speaker": 30}

IDLE, ARMED, SELECTING, ADJUSTING, CONFIRMING = "IDLE", "ARMED", "SELECTING", "ADJUSTING", "CONFIRMING"
STATE_COLORS = {IDLE: "#6b7280", ARMED: "#3b82f6", SELECTING: "#eab308", ADJUSTING: "#22c55e", CONFIRMING: "#f97316"}

# Scripted demo (autopilot). pose: none / open_palm / point / pinch / thumbs_up / fist
SCRIPT = [
    dict(label="Hand enters the view", dur=1.2, pose="none"),
    dict(label="Open palm held 1 s -> ARM", dur=1.8, pose="open_palm"),
    dict(label="Point at the Lamp and hold -> SELECT", dur=1.6, pose="point", target="lamp"),
    dict(label="Pinch closer -> dim the lamp", dur=3.0, pose="pinch", p0=0.95, p1=0.50),
    dict(label="Point at the Fan and hold -> SELECT", dur=1.6, pose="point", target="fan"),
    dict(label="Pinch wide -> fan to MAX (needs confirmation)", dur=3.0, pose="pinch", p0=0.40, p1=1.20),
    dict(label="Thumbs-up -> CONFIRM", dur=1.2, pose="thumbs_up"),
    dict(label="Read-back: device state vs command", dur=1.8, pose="none"),
    dict(label="Fault injected: Speaker will ignore commands", dur=0.4, pose="none", fault=("speaker", True)),
    dict(label="Point at the Speaker and hold -> SELECT", dur=1.6, pose="point", target="speaker"),
    dict(label="Pinch -> speaker 60 % (device fails -> MISMATCH)", dur=2.5, pose="pinch", p0=0.40, p1=0.78),
    dict(label="Mismatch shown on dashboard", dur=2.2, pose="none"),
    dict(label="Fault cleared, fist -> DISARM", dur=1.2, pose="fist", fault=("speaker", False)),
    dict(label="Idle", dur=2.5, pose="none"),
]


# ----------------------------------------------------------------------------
# Math helpers (landmarks are normalized image coordinates)
# ----------------------------------------------------------------------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def palm_width(lm):
    return dist(lm[5], lm[17])


def palm_center(lm):
    pts = [lm[i] for i in (0, 5, 9, 13, 17)]
    return (sum(p[0] for p in pts) / 5, sum(p[1] for p in pts) / 5)


def pinch_ratio(lm):
    return dist(lm[4], lm[8]) / palm_width(lm)


def synth_landmarks(ratio, rng):
    base = {0: (.50, .90), 5: (.42, .60), 9: (.50, .57), 13: (.58, .60), 17: (.65, .65), 8: (.42, .30)}
    pw = dist(base[5], base[17])
    base[4] = (base[8][0] + ratio * pw, base[8][1])
    return {i: (x + rng.gauss(0, .002), y + rng.gauss(0, .002)) for i, (x, y) in base.items()}


def quantize(device, fraction):
    spec = DEVICES[device]
    fraction = max(0.0, min(1.0, fraction))
    steps = round(fraction * spec["max"] / spec["step"])
    return int(steps * spec["step"])


def is_risky(device, value):
    r = DEVICES[device]["risk_at"]
    return r is not None and value >= r


class OneEuro:
    def __init__(self, mincutoff, beta, dcutoff):
        self.mc, self.beta, self.dc = mincutoff, beta, dcutoff
        self.t = self.x = None
        self.dx = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t is None:
            self.t, self.x = t, x
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        ad = self._alpha(self.dc, dt)
        self.dx = ad * dx + (1 - ad) * self.dx
        a = self._alpha(self.mc + self.beta * abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


class RateLimiter:
    def __init__(self, max_per_s):
        self.max, self.stamps = max_per_s, deque()

    def allow(self, now):
        while self.stamps and now - self.stamps[0] >= 1.0:
            self.stamps.popleft()
        if len(self.stamps) < self.max:
            self.stamps.append(now)
            return True
        return False


@dataclass
class GestureEvent:
    type: str
    hand: str
    t_ms: int
    confidence: float
    params: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Message bus (in-memory MQTT-style topics) and simulated devices
# ----------------------------------------------------------------------------
class Bus:
    def __init__(self):
        self.subs, self.messages = {}, deque(maxlen=200)

    def subscribe(self, topic, cb):
        self.subs.setdefault(topic, []).append(cb)

    def publish(self, topic, payload):
        self.messages.append({"time": time.strftime("%H:%M:%S"), "topic": topic, "payload": str(payload)})
        for cb in self.subs.get(topic, []):
            cb(payload)


class Device:
    def __init__(self, name, bus, value):
        self.name, self.bus, self.value = name, bus, value
        self.fault, self.queue = False, []
        bus.subscribe(f"home/{name}/set", self.on_set)
        self.publish(None)

    def on_set(self, msg):
        lo, hi = CONFIG["device_delay_s"]
        self.queue.append((time.time() + random.uniform(lo, hi), msg))

    def tick(self, now):
        due = [q for q in self.queue if q[0] <= now]
        self.queue = [q for q in self.queue if q[0] > now]
        for _, msg in due:
            if not self.fault:
                self.value = int(max(0, min(DEVICES[self.name]["max"], msg["value"])))
            self.publish(msg["request_id"])

    def publish(self, request_id):
        self.bus.publish(f"home/{self.name}/state",
                         {"device": self.name, "value": self.value, "request_id": request_id})


class Controller:
    def __init__(self, bus):
        self.bus, self.pending, self.log = bus, {}, []
        self.readback, self.sent, self.ok, self.bad = None, 0, 0, 0
        self.latencies = deque(maxlen=50)
        for name in DEVICES:
            bus.subscribe(f"home/{name}/state", self.on_state)

    def send(self, device, value):
        req = uuid.uuid4().hex[:6]
        cmd = {"device": device, "action": DEVICES[device]["action"], "value": value, "request_id": req}
        self.pending[req] = {"cmd": cmd, "t": time.perf_counter(), "wall": time.time()}
        self.log.append({"time": time.strftime("%H:%M:%S"), "device": device, "action": cmd["action"],
                         "value": value, "request_id": req, "result": "pending...", "latency_ms": None})
        self.sent += 1
        self.bus.publish(f"home/{device}/set", cmd)

    def _update_log(self, req, result, lat):
        for row in reversed(self.log):
            if row["request_id"] == req:
                row["result"], row["latency_ms"] = result, lat
                return

    def on_state(self, payload):
        req = payload.get("request_id")
        if req not in self.pending:
            return
        p = self.pending.pop(req)
        lat = round((time.perf_counter() - p["t"]) * 1000)
        self.latencies.append(lat)
        want, got, dev = p["cmd"]["value"], payload["value"], payload["device"]
        if want == got:
            self.ok += 1
            self._update_log(req, "OK ✓", lat)
            self.readback = {"ok": True, "t": time.time(), "msg": f"{dev}: commanded {want}, device reports {got} - read-back OK"}
        else:
            self.bad += 1
            self._update_log(req, f"MISMATCH ✗ (device reports {got})", lat)
            self.readback = {"ok": False, "t": time.time(), "msg": f"{dev}: commanded {want}, device reports {got} - MISMATCH"}

    def check_timeouts(self, now):
        for req in [r for r, p in self.pending.items() if now - p["wall"] > CONFIG["readback_timeout_s"]]:
            p = self.pending.pop(req)
            self.bad += 1
            self._update_log(req, "NO RESPONSE ✗", None)
            self.readback = {"ok": False, "t": now, "msg": f"{p['cmd']['device']}: no state report - NO RESPONSE"}


# ----------------------------------------------------------------------------
# Grammar state machine: arm -> select -> adjust -> confirm (with timeouts)
# ----------------------------------------------------------------------------
class Grammar:
    def __init__(self, cfg):
        self.cfg = cfg
        self.limiter = RateLimiter(cfg["max_changes_per_s"])
        self.last_pose = self.last_target = None
        self.pose_since, self.fired = 0.0, False
        self.last_cmd = {}
        self.reset(0.0)

    def reset(self, now, reason=""):
        self.state, self.entered = IDLE, now
        self.selected, self.pending = None, None
        self.refractory_until = now + self.cfg["refractory_s"]
        self.block_until = 0.0
        self.notice = reason

    def goto(self, state, now):
        self.state, self.entered = state, now

    def timeout(self):
        c = self.cfg
        return {ARMED: c["arm_timeout_s"], SELECTING: c["select_timeout_s"],
                ADJUSTING: c["adjust_timeout_s"], CONFIRMING: c["confirm_timeout_s"]}.get(self.state)

    def remaining(self, now):
        t = self.timeout()
        return None if t is None else max(0.0, t - (now - self.entered))

    def held(self, now):
        return now - self.pose_since

    def step(self, now, pose, target, ratio, current):
        c, ev, cmds = self.cfg, [], []

        def emit(kind, **params):
            ev.append(GestureEvent(kind, "right", int(now * 1000), 0.95, params))

        if pose != self.last_pose or target != self.last_target:
            self.last_pose, self.last_target, self.pose_since, self.fired = pose, target, now, False
        held = now - self.pose_since

        rem = self.remaining(now)
        if rem is not None and rem <= 0:
            emit("timeout", state=self.state)
            self.reset(now, f"{self.state} timed out - disarmed")
            return ev, cmds

        if self.state == IDLE:
            if pose == "open_palm" and held >= c["arm_hold_s"] and not self.fired and now >= self.refractory_until:
                self.fired = True
                self.goto(ARMED, now)
                self.notice = "Armed - point at a device"
                emit("arm")

        elif self.state == ARMED:
            if pose == "fist":
                self.reset(now, "Disarmed")
                emit("disarm")
            elif pose == "point" and target:
                self.goto(SELECTING, now)

        elif self.state == SELECTING:
            if pose == "fist":
                self.reset(now, "Disarmed")
                emit("disarm")
            elif pose != "point" or not target:
                self.goto(ARMED, now)
            elif held >= c["select_hold_s"]:
                self.selected = target
                self.goto(ADJUSTING, now)
                self.notice = f"{DEVICES[target]['label']} selected - pinch to adjust"
                emit("select", device=target)

        elif self.state == ADJUSTING:
            if pose == "fist":
                self.reset(now, "Disarmed")
                emit("disarm")
            elif pose == "point" and target and target != self.selected:
                self.goto(SELECTING, now)
            elif pose == "pinch" and now >= self.block_until:
                dev = self.selected
                frac = (ratio - c["pinch_min"]) / (c["pinch_max"] - c["pinch_min"])
                val = quantize(dev, frac)
                ref = self.last_cmd.get(dev, current[dev])
                if val != ref:
                    if not self.limiter.allow(now):
                        emit("rate_limited", device=dev)
                    elif is_risky(dev, val):
                        self.pending = (dev, val)
                        self.goto(CONFIRMING, now)
                        self.notice = f"High-risk: {dev} -> {val}. Thumbs-up to confirm, fist to cancel"
                        emit("confirm_required", device=dev, value=val)
                    else:
                        cmds.append((dev, val))
                        self.last_cmd[dev] = val
                        self.entered = now
                        emit("adjust", device=dev, value=val, pinch_ratio=round(ratio, 2))

        elif self.state == CONFIRMING:
            if pose == "fist":
                self.pending = None
                self.block_until = now + c["cancel_block_s"]
                self.goto(ADJUSTING, now)
                self.notice = "Cancelled - nothing was sent"
                emit("cancel")
            elif pose == "thumbs_up" and held >= c["confirm_hold_s"] and not self.fired:
                self.fired = True
                dev, val = self.pending
                cmds.append((dev, val))
                self.last_cmd[dev] = val
                self.pending = None
                self.goto(ADJUSTING, now)
                self.notice = "Confirmed - command sent"
                emit("confirm", device=dev, value=val)
        return ev, cmds


# ----------------------------------------------------------------------------
# Session setup
# ----------------------------------------------------------------------------
def seed_log(ctl):
    now = time.time()
    seeds = [("lamp", 60, 118), ("speaker", 25, 142), ("fan", 3, 96), ("lamp", 80, 131)]
    for i, (dev, val, lat) in enumerate(seeds):
        t = time.strftime("%H:%M:%S", time.localtime(now - 600 + i * 90))
        ctl.log.append({"time": t, "device": dev, "action": DEVICES[dev]["action"], "value": val,
                        "request_id": f"seed{i}", "result": "OK ✓ (demo history)", "latency_ms": lat})


def init():
    S = st.session_state
    if "ready" in S:
        return
    S.bus = Bus()
    S.devices = {n: Device(n, S.bus, v) for n, v in INITIAL.items()}
    S.ctl = Controller(S.bus)
    seed_log(S.ctl)
    S.grammar = Grammar(CONFIG)
    S.filter = OneEuro(CONFIG["oe_mincutoff"], CONFIG["oe_beta"], CONFIG["oe_dcutoff"])
    S.rng = random.Random(7)
    S.feed = deque(maxlen=12)
    S.script_fault = {n: False for n in DEVICES}
    S.idx, S.step_t0 = 0, time.time()
    S.last_tick, S.hz = time.time(), 3.0
    S.lm, S.ratio, S.pose, S.label = synth_landmarks(0.9, S.rng), 0.9, "none", SCRIPT[0]["label"]
    S.ready = True


def wipe():
    S = st.session_state
    S.ctl.log.clear()
    S.ctl.pending.clear()
    S.ctl.readback = None
    S.ctl.sent = S.ctl.ok = S.ctl.bad = 0
    S.ctl.latencies.clear()
    S.bus.messages.clear()
    S.feed.clear()


def restart_demo():
    S = st.session_state
    now = time.time()
    S.idx, S.step_t0 = 0, now
    S.grammar.reset(now)
    S.grammar.last_cmd.clear()
    S.script_fault = {n: False for n in DEVICES}
    for n, v in INITIAL.items():
        S.devices[n].value = v
        S.devices[n].publish(None)


def autopilot(now):
    S = st.session_state
    step = SCRIPT[S.idx]
    if now - S.step_t0 >= step["dur"]:
        S.idx = (S.idx + 1) % len(SCRIPT)
        S.step_t0, step = now, SCRIPT[S.idx]
        if "fault" in step:
            S.script_fault[step["fault"][0]] = step["fault"][1]
    frac = min(1.0, (now - S.step_t0) / step["dur"])
    p0 = step.get("p0", 0.9)
    p = p0 + (step.get("p1", p0) - p0) * frac
    return step["pose"], step.get("target"), p, step["label"]


def tick():
    S = st.session_state
    now = time.time()
    dt = max(now - S.last_tick, 1e-3)
    S.last_tick = now
    S.hz = 0.8 * S.hz + 0.2 * (1.0 / dt)

    if S.get("autopilot", True):
        pose, target, p, label = autopilot(now)
    else:
        pose, target, p = S.get("m_pose", "none"), S.get("m_target", "lamp"), S.get("m_pinch", 0.8)
        target = target if pose in ("point",) else None
        label = "Manual control"
    S.pose, S.label = pose, label

    S.lm = synth_landmarks(p, S.rng)
    S.ratio = S.filter(pinch_ratio(S.lm), now)

    for n, d in S.devices.items():
        d.fault = S.script_fault[n] or S.get(f"fault_{n}", False)
        d.tick(now)

    current = {n: d.value for n, d in S.devices.items()}
    events, cmds = S.grammar.step(now, pose, target, S.ratio, current)
    for e in events:
        extra = " ".join(f"{k}={v}" for k, v in e.params.items())
        S.feed.appendleft(f"{time.strftime('%H:%M:%S')}  {e.type.upper():<17} conf={e.confidence}  {extra}")
    for dev, val in cmds:
        S.ctl.send(dev, val)
    S.ctl.check_timeouts(now)


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
CSS = """
<style>
.tile{border:2px solid rgba(128,128,128,.35);border-radius:14px;padding:14px;text-align:center;
      background:rgba(128,128,128,.08);transition:all .25s}
.tile .icon{font-size:42px}.tile .name{font-weight:700;font-size:18px}.tile .val{font-size:26px;font-weight:800}
.bar{height:10px;border-radius:6px;background:rgba(128,128,128,.25);overflow:hidden;margin-top:8px}
.fill{height:100%;background:#f59e0b;transition:width .3s}
.tag{display:inline-block;margin-top:8px;padding:2px 8px;border-radius:8px;font-size:12px;color:#fff}
.badge{display:inline-block;padding:6px 14px;border-radius:999px;color:#fff;font-weight:800;font-size:18px}
</style>
"""


def tile_html(name, dev, grammar):
    spec = DEVICES[name]
    v = dev.value
    pct = 100 * v / spec["max"]
    border = "rgba(128,128,128,.35)"
    tag = ""
    if grammar.state in (ADJUSTING, CONFIRMING) and grammar.selected == name:
        border = "#f97316" if grammar.state == CONFIRMING else "#22c55e"
        tag = '<span class="tag" style="background:%s">%s</span>' % (border, "CONFIRM?" if grammar.state == CONFIRMING else "SELECTED")
    elif grammar.state == SELECTING and grammar.last_target == name:
        border = "#eab308"
        tag = '<span class="tag" style="background:#eab308">POINTING...</span>'
    if dev.fault:
        tag += ' <span class="tag" style="background:#ef4444">FAULT</span>'
    text = ("OFF" if v == 0 else f"Speed {v}/5") if name == "fan" else f"{v}{spec['unit']}"
    opacity = 0.3 + 0.7 * pct / 100 if name == "lamp" else 1
    return (f'<div class="tile" style="border-color:{border}"><div class="icon" style="opacity:{opacity:.2f}">{spec["icon"]}</div>'
            f'<div class="name">{spec["label"]}</div><div class="val">{text}</div>'
            f'<div class="bar"><div class="fill" style="width:{pct:.0f}%"></div></div>{tag}</div>')


def hand_svg(lm, ratio, pose):
    def P(i):
        return f"{lm[i][0] * 100:.1f},{lm[i][1] * 100:.1f}"
    palm = " ".join(P(i) for i in (0, 5, 9, 13, 17))
    col = "#22c55e" if ratio < 0.35 else "#60a5fa"
    return f"""<svg viewBox="30 20 50 80" width="100%" height="230" style="background:rgba(128,128,128,.08);border-radius:12px">
<polygon points="{palm}" fill="rgba(96,165,250,.18)" stroke="#60a5fa" stroke-width=".8"/>
<line x1="{lm[5][0]*100:.1f}" y1="{lm[5][1]*100:.1f}" x2="{lm[8][0]*100:.1f}" y2="{lm[8][1]*100:.1f}" stroke="#94a3b8" stroke-width=".8"/>
<line x1="{lm[0][0]*100:.1f}" y1="{lm[0][1]*100:.1f}" x2="{lm[4][0]*100:.1f}" y2="{lm[4][1]*100:.1f}" stroke="#94a3b8" stroke-width=".8"/>
<line x1="{lm[4][0]*100:.1f}" y1="{lm[4][1]*100:.1f}" x2="{lm[8][0]*100:.1f}" y2="{lm[8][1]*100:.1f}" stroke="{col}" stroke-width="1.2" stroke-dasharray="1.5 1"/>
{''.join(f'<circle cx="{lm[i][0]*100:.1f}" cy="{lm[i][1]*100:.1f}" r="1.1" fill="#e2e8f0"/>' for i in (0,5,9,13,17))}
<circle cx="{lm[4][0]*100:.1f}" cy="{lm[4][1]*100:.1f}" r="2" fill="{col}"/>
<circle cx="{lm[8][0]*100:.1f}" cy="{lm[8][1]*100:.1f}" r="2" fill="{col}"/>
<text x="31" y="26" font-size="3.6" fill="#94a3b8">pose: {pose}  |  pinch_ratio: {ratio:.2f}</text></svg>"""


GRAMMAR_DIAGRAM = """IDLE --open palm held 1 s--> ARMED (timeout 8 s)
ARMED --point at device--> SELECTING (timeout 5 s, hold 0.8 s)
SELECTING --hold complete--> ADJUSTING (timeout 10 s, pinch = value, max 5 changes/s)
ADJUSTING --risky value (fan max, volume >= 80)--> CONFIRMING (timeout 6 s)
CONFIRMING --thumbs-up--> send command --> ADJUSTING
CONFIRMING --fist--> cancel (nothing sent) --> ADJUSTING
ANY --fist or timeout--> IDLE (refractory lockout)
After every command: read-back compares device state with command -> OK / MISMATCH"""


def render():
    S = st.session_state
    g, ctl = S.grammar, S.ctl
    now = time.time()

    avg = f"{sum(ctl.latencies) / len(ctl.latencies):.0f} ms" if ctl.latencies else "-"
    m = st.columns(5)
    m[0].metric("Commands sent", ctl.sent)
    m[1].metric("Read-back OK", ctl.ok)
    m[2].metric("Mismatches", ctl.bad)
    m[3].metric("Avg latency", avg)
    m[4].metric("Loop rate", f"{S.hz:.1f} Hz")

    if S.get("autopilot", True):
        st.info(f"▶ Live demo running - {S.label}")

    left, right = st.columns([2, 1])
    with left:
        cols = st.columns(3)
        for col, name in zip(cols, DEVICES):
            col.markdown(tile_html(name, S.devices[name], g), unsafe_allow_html=True)
        rem = g.remaining(now)
        cd = f" - {rem:.1f}s left" if rem is not None else ""
        st.markdown(f'<br><span class="badge" style="background:{STATE_COLORS[g.state]}">{g.state}{cd}</span>',
                    unsafe_allow_html=True)
        if g.notice:
            st.caption(g.notice)
        rb = ctl.readback
        if rb and now - rb["t"] < 6:
            (st.success if rb["ok"] else st.error)(("✅ " if rb["ok"] else "❌ ") + rb["msg"])
    with right:
        st.markdown(hand_svg(S.lm, S.ratio, S.pose), unsafe_allow_html=True)
        st.caption("Simulated hand landmarks - no camera is used.")

    t1, t2, t3, t4 = st.tabs(["Command log", "Gesture events", "MQTT-style bus", "Grammar"])
    with t1:
        if ctl.log:
            st.dataframe(pd.DataFrame(ctl.log[-12:][::-1]), hide_index=True)
    with t2:
        st.code("\n".join(S.feed) or "waiting for gestures...", language="text")
    with t3:
        msgs = list(S.bus.messages)[-8:][::-1]
        if msgs:
            st.dataframe(pd.DataFrame(msgs), hide_index=True)
    with t4:
        st.code(GRAMMAR_DIAGRAM, language="text")


@st.fragment(run_every=CONFIG["tick_s"])
def live():
    tick()
    render()


# ----------------------------------------------------------------------------
# Self-tests (run from the sidebar)
# ----------------------------------------------------------------------------
def run_self_tests():
    res = []
    lm = synth_landmarks(0.5, random.Random(1))
    res.append(("pinch_ratio close to 0.5", abs(pinch_ratio(lm) - 0.5) < 0.1))
    res.append(("palm_width positive", palm_width(lm) > 0))
    res.append(("quantize fan 1.0 -> 5", quantize("fan", 1.0) == 5))
    rl = RateLimiter(5)
    first = [rl.allow(i * 0.01) for i in range(6)]
    res.append(("rate limiter: 5 pass, 6th blocked", first == [True] * 5 + [False]))
    res.append(("rate limiter: allows again after 1 s", rl.allow(1.5)))
    cur = {"lamp": 50, "fan": 2, "speaker": 30}
    g = Grammar(CONFIG)
    g.step(0.0, "open_palm", None, .5, cur)
    g.step(1.2, "open_palm", None, .5, cur)
    res.append(("grammar: arm after 1 s palm", g.state == ARMED))
    g.step(10.0, "none", None, .5, cur)
    res.append(("grammar: arm timeout -> IDLE", g.state == IDLE))
    g = Grammar(CONFIG)
    for t, pose, tgt, r in [(0, "open_palm", None, .5), (1.1, "open_palm", None, .5), (1.2, "point", "fan", .5),
                            (2.2, "point", "fan", .5)]:
        g.step(t, pose, tgt, r, cur)
    _, cmds = g.step(2.3, "pinch", None, 1.2, cur)
    res.append(("grammar: risky change waits for confirmation", g.state == CONFIRMING and cmds == []))
    _, cmds = g.step(3.0, "thumbs_up", None, 1.2, cur)
    _, cmds = g.step(3.6, "thumbs_up", None, 1.2, cur)
    res.append(("grammar: confirm sends command", cmds == [("fan", 5)]))
    bus = Bus()
    ctl = Controller(bus)
    dev = Device("fan", bus, 2)
    dev.fault = True
    ctl.send("fan", 4)
    dev.tick(time.time() + 1)
    res.append(("read-back mismatch detected", ctl.readback is not None and not ctl.readback["ok"]))
    return res


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
init()
st.markdown(CSS, unsafe_allow_html=True)
st.title("🏠 GestureHome")
st.caption("Simulated smart-room controller: arm → select → adjust → confirm → read-back. "
           "Runs entirely on your computer. A demo is already running below.")

with st.sidebar:
    st.header("Controls")
    st.toggle("Demo autopilot", value=True, key="autopilot",
              help="Turn off to control the simulated hand yourself.")
    manual = not st.session_state.get("autopilot", True)
    st.radio("Hand pose", ["none", "open_palm", "point", "pinch", "thumbs_up", "fist"],
             key="m_pose", disabled=not manual)
    st.selectbox("Point at", list(DEVICES), key="m_target", disabled=not manual)
    st.slider("Pinch ratio", 0.1, 1.3, 0.8, 0.05, key="m_pinch", disabled=not manual)
    st.caption("Manual: hold open_palm 1 s to arm → point 1 s → pinch → thumbs_up to confirm risky changes → fist to disarm.")
    st.divider()
    st.subheader("Force a device failure")
    for n in DEVICES:
        st.checkbox(f"{DEVICES[n]['label']} ignores commands", key=f"fault_{n}")
    st.divider()
    if st.button("🔄 Restart demo"):
        restart_demo()
    if st.session_state.ctl.log:
        df = pd.DataFrame(st.session_state.ctl.log)
        st.download_button("⬇ Download command log (CSV)", df.to_csv(index=False), "gesturehome_log.csv", "text/csv")
    if st.button("🗑 Delete all data"):
        wipe()
        st.success("Log, bus history and counters deleted.")
    if st.button("🧪 Run self-tests"):
        for name, ok in run_self_tests():
            st.write(("✅ " if ok else "❌ ") + name)
    st.divider()
    st.caption("Privacy: no camera, no network upload, no identification of people. Data lives in memory only.")

live()