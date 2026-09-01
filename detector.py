"""
MNQ live setup detector — runs on a schedule via GitHub Actions.

Pattern: large clean impulse candle -> opposite-color next candle begins forming
-> lower-timeframe leading confirmation within the first N minutes of that next
candle -> alert sent via Telegram (target = the impulse candle's open price).

State is persisted in state.json (committed back to the repo by the workflow)
so the same candle never triggers more than one alert.
"""

import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def to_ist_str(iso_time_str):
    """Convert a UTC ISO timestamp to a readable IST string,
    e.g. '23 Aug 2026, 07:45 PM IST'."""
    dt_utc = datetime.fromisoformat(iso_time_str)
    dt_ist = dt_utc.astimezone(IST)
    return dt_ist.strftime("%d %b %Y, %I:%M %p IST")


def candle_close_ist_str(iso_start_time_str, tf_minutes):
    """Bars are timestamped by their START time. This computes the actual
    CLOSE time (start + duration) and formats it in IST — since alerts should
    reflect when the candle actually finished forming, not when it started."""
    dt_start_utc = datetime.fromisoformat(iso_start_time_str)
    dt_close_utc = dt_start_utc + timedelta(minutes=tf_minutes)
    return to_ist_str(dt_close_utc.isoformat())

BASE_URL = "https://api.topstepx.com"
STATE_FILE = "state.json"

MIN_BODY_POINTS = 50
MAX_WICK_PCT = 0.30
MAX_PROCESSED_PER_TF = 2000  # generous cap — state now only stores real candidates, not every candle

# Timeframe -> API params + own duration in minutes
TIMEFRAMES = {
    "5m":  {"unit": 2, "unitNumber": 5,  "minutes": 5,  "lookback_days": 5},
    "15m": {"unit": 2, "unitNumber": 15, "minutes": 15, "lookback_days": 10},
    "30m": {"unit": 2, "unitNumber": 30, "minutes": 30, "lookback_days": 15},
    "1h":  {"unit": 3, "unitNumber": 1,  "minutes": 60, "lookback_days": 30},
}

# Confirmation config per TF: which lower granularity + how many minutes into
# the next candle's formation to check for an opposite-direction move
CONFIRM_CONFIG = {
    "5m":  {"lower_minutes": 1, "confirm_minutes": 3},
    "15m": {"lower_minutes": 5, "confirm_minutes": 10},
    "30m": {"lower_minutes": 5, "confirm_minutes": 15},
    "1h":  {"lower_minutes": 5, "confirm_minutes": 15},
}

LOWER_GRAN_LOOKBACK_DAYS = {1: 5, 5: 15}


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            state.setdefault("processed", {tf: [] for tf in TIMEFRAMES})
            state.setdefault("raw_processed", {tf: [] for tf in TIMEFRAMES})
            state.setdefault("sweep_processed", {tf: [] for tf in TIMEFRAMES})
            return state
    return {
        "processed": {tf: [] for tf in TIMEFRAMES},
        "raw_processed": {tf: [] for tf in TIMEFRAMES},
        "sweep_processed": {tf: [] for tf in TIMEFRAMES},
    }


def save_state(state):
    for tf in state["processed"]:
        state["processed"][tf] = state["processed"][tf][-MAX_PROCESSED_PER_TF:]
    for tf in state["raw_processed"]:
        state["raw_processed"][tf] = state["raw_processed"][tf][-MAX_PROCESSED_PER_TF:]
    for tf in state["sweep_processed"]:
        state["sweep_processed"][tf] = state["sweep_processed"][tf][-MAX_PROCESSED_PER_TF:]
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def authenticate():
    username = os.environ["TOPSTEPX_USERNAME"]
    api_key = os.environ["TOPSTEPX_API_KEY"]
    session = requests.Session()
    resp = session.post(
        f"{BASE_URL}/api/Auth/loginKey",
        headers={"accept": "text/plain", "Content-Type": "application/json"},
        json={"userName": username, "apiKey": api_key},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        print(f"❌ Auth failed: {data}")
        sys.exit(1)
    token = data["token"]
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    return session, headers


def get_mnq_contract_id(session, headers):
    resp = session.post(
        f"{BASE_URL}/api/Contract/search",
        headers=headers,
        json={"live": False, "searchText": "MNQ"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        print(f"❌ Contract search failed: {data}")
        sys.exit(1)
    for c in data.get("contracts", []):
        if c.get("symbolId") == "F.US.MNQ" and c.get("activeContract"):
            return c["id"]
    print(f"❌ No active MNQ contract found: {data}")
    sys.exit(1)


def fetch_bars(session, headers, contract_id, unit, unit_number, lookback_days):
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)
    resp = session.post(
        f"{BASE_URL}/api/History/retrieveBars",
        headers=headers,
        json={
            "contractId": contract_id,
            "live": False,
            "startTime": start_dt.isoformat(),
            "endTime": end_dt.isoformat(),
            "unit": unit,
            "unitNumber": unit_number,
            "limit": 20000,
            "includePartialBar": False,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        print(f"⚠️  Bars fetch error ({unit}/{unit_number}): {data}")
        return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])
    bars = data.get("bars") or []
    if not bars:
        return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])
    df = pd.DataFrame(bars).drop_duplicates(subset="t")
    df["t"] = pd.to_datetime(df["t"], utc=True)
    return df.sort_values("t").reset_index(drop=True)


def send_telegram(text):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=15,
    )
    if not resp.ok:
        print(f"⚠️  Telegram send failed: {resp.status_code} {resp.text}")
    else:
        print("✅ Telegram alert sent.")


def evaluate_candidates(tf_name, df_big, df_low, tf_minutes, confirm_minutes,
                         processed_list, now_utc):
    alerts = []
    newly_processed = []
    tf_duration = timedelta(minutes=tf_minutes)
    confirm_duration = timedelta(minutes=confirm_minutes)

    for i in range(len(df_big)):
        row = df_big.iloc[i]
        ts_str = row["t"].isoformat()
        if ts_str in processed_list:
            continue

        body = row["c"] - row["o"]
        rng = row["h"] - row["l"]
        if rng <= 0 or abs(body) < MIN_BODY_POINTS:
            # Doesn't qualify as a big candle — no need to remember this one,
            # it's cheap to re-check next run and never needs revisiting.
            continue

        upper_wick = row["h"] - max(row["o"], row["c"])
        lower_wick = min(row["o"], row["c"]) - row["l"]
        total_wick = upper_wick + lower_wick
        if total_wick > MAX_WICK_PCT * rng:
            # Same reasoning — not clean, cheap to re-check, don't persist.
            continue

        color = "green" if body > 0 else "red"
        next_candle_start = row["t"] + tf_duration
        confirm_end = next_candle_start + confirm_duration

        if now_utc < confirm_end:
            # Not enough time has passed yet to fully evaluate the confirm
            # window — leave unprocessed so we re-check on a later run.
            continue

        low_bars = df_low[(df_low["t"] >= next_candle_start) & (df_low["t"] < confirm_end)]
        if len(low_bars) == 0:
            # Give up after 2 hours of no lower-TF data (likely a data gap)
            if (now_utc - confirm_end).total_seconds() > 7200:
                newly_processed.append(ts_str)
            continue

        sub_open = low_bars["o"].iloc[0]
        sub_close = low_bars["c"].iloc[-1]
        sub_color = "green" if sub_close > sub_open else ("red" if sub_close < sub_open else "doji")
        low_confirms = (color == "green" and sub_color == "red") or \
                        (color == "red" and sub_color == "green")

        newly_processed.append(ts_str)

        if low_confirms:
            alerts.append({
                "tf": tf_name,
                "time": ts_str,
                "color": color,
                "body_points": round(abs(body), 2),
                "open": row["o"],
                "close": row["c"],
            })

    return alerts, newly_processed


def evaluate_raw_impulses(tf_name, df_big, raw_processed_list):
    """
    Instant, unconfirmed alert path: fires the moment a qualifying impulse
    candle is seen as closed — no waiting for the lower-TF confirmation
    window. Separate state list from the confirmed-setup path, so the two
    alert types are tracked independently and don't interfere with each other.
    """
    raw_alerts = []
    newly_processed = []

    for i in range(len(df_big)):
        row = df_big.iloc[i]
        ts_str = row["t"].isoformat()
        if ts_str in raw_processed_list:
            continue

        body = row["c"] - row["o"]
        rng = row["h"] - row["l"]
        if rng <= 0 or abs(body) < MIN_BODY_POINTS:
            continue

        upper_wick = row["h"] - max(row["o"], row["c"])
        lower_wick = min(row["o"], row["c"]) - row["l"]
        total_wick = upper_wick + lower_wick
        if total_wick > MAX_WICK_PCT * rng:
            continue

        color = "green" if body > 0 else "red"
        newly_processed.append(ts_str)
        raw_alerts.append({
            "tf": tf_name,
            "time": ts_str,
            "color": color,
            "body_points": round(abs(body), 2),
            "open": row["o"],
            "close": row["c"],
        })

    return raw_alerts, newly_processed


# ---- Third alert type: opposite + sweep + shallow (0-30%) close ----
# Uses a DIFFERENT impulse definition than the raw/confirmed alerts above,
# matching exactly the validated backtest: body >= 50pts, body/range >= 60%,
# and the wick on the impulse's OWN side <= 15 points (not a % of range).
SWEEP_MIN_BODY_POINTS = 50.0
SWEEP_MIN_BODY_TO_RANGE = 0.60
SWEEP_MAX_WICK_POINTS = 15.0


def is_sweep_impulse(row):
    o, h, l, c = row["o"], row["h"], row["l"], row["c"]
    body = abs(c - o)
    rng = h - l
    if rng <= 0 or body < SWEEP_MIN_BODY_POINTS:
        return False, None
    if (body / rng) < SWEEP_MIN_BODY_TO_RANGE:
        return False, None

    if c > o:
        upper_wick = h - max(o, c)
        if upper_wick > SWEEP_MAX_WICK_POINTS:
            return False, None
        return True, "green"
    elif c < o:
        lower_wick = min(o, c) - l
        if lower_wick > SWEEP_MAX_WICK_POINTS:
            return False, None
        return True, "red"
    return False, None


def retracement_percent(impulse_open, impulse_close, price):
    """0% = impulse close, 100% = impulse open."""
    body = abs(impulse_close - impulse_open)
    if body <= 0:
        return 0.0
    if impulse_close > impulse_open:
        return ((impulse_close - price) / body) * 100.0
    return ((price - impulse_close) / body) * 100.0


def evaluate_sweep_setups(tf_name, df_big, sweep_processed_list):
    """
    Third, independent alert path: fires once the IMMEDIATE next same-TF
    candle has closed and satisfies three conditions together:
      1. Opposite color from the impulse
      2. Sweeps beyond the impulse's own extreme before closing
      3. Closes back in the shallow 0-30% retracement zone
    Requires a full next-candle close (not just a partial lower-TF window),
    so this is typically SLOWER to fire than the confirmed 🚨 alert, but
    doesn't depend on lower-timeframe data at all.
    """
    sweep_alerts = []
    newly_processed = []

    for i in range(len(df_big) - 1):
        row = df_big.iloc[i]
        ts_str = row["t"].isoformat()
        if ts_str in sweep_processed_list:
            continue

        is_impulse, color = is_sweep_impulse(row)
        if not is_impulse:
            continue

        nxt = df_big.iloc[i + 1]  # guaranteed closed (includePartialBar=False)

        opposite = (nxt["c"] < nxt["o"]) if color == "green" else (nxt["c"] > nxt["o"])
        sweep = (nxt["h"] > row["h"]) if color == "green" else (nxt["l"] < row["l"])
        next_close_retr = retracement_percent(row["o"], row["c"], nxt["c"])
        close_0_30 = 0.0 <= next_close_retr <= 30.0

        newly_processed.append(ts_str)

        if opposite and sweep and close_0_30:
            sweep_alerts.append({
                "tf": tf_name,
                "time": ts_str,
                "color": color,
                "body_points": round(abs(row["c"] - row["o"]), 2),
                "open": row["o"],
                "close": row["c"],
                "next_close": nxt["c"],
                "next_close_retracement_pct": round(next_close_retr, 1),
            })

    return sweep_alerts, newly_processed


def main():
    state = load_state()
    session, headers = authenticate()
    contract_id = get_mnq_contract_id(session, headers)
    now_utc = datetime.now(timezone.utc)

    granularity_cache = {}

    def get_granularity(minutes, unit, unit_number, lookback_days):
        key = minutes
        if key not in granularity_cache:
            df = fetch_bars(session, headers, contract_id, unit, unit_number, lookback_days)
            granularity_cache[key] = df
            print(f"  Fetched {len(df)} bars @ {minutes}m")
        return granularity_cache[key]

    all_alerts = []
    all_raw_alerts = []
    all_sweep_alerts = []

    for tf_name, cfg in TIMEFRAMES.items():
        df_big = get_granularity(cfg["minutes"], cfg["unit"], cfg["unitNumber"], cfg["lookback_days"])
        conf_cfg = CONFIRM_CONFIG[tf_name]
        lb_days = LOWER_GRAN_LOOKBACK_DAYS[conf_cfg["lower_minutes"]]
        df_low = get_granularity(conf_cfg["lower_minutes"], 2, conf_cfg["lower_minutes"], lb_days)

        processed_list = state["processed"].setdefault(tf_name, [])
        alerts, newly_processed = evaluate_candidates(
            tf_name, df_big, df_low, cfg["minutes"], conf_cfg["confirm_minutes"],
            processed_list, now_utc,
        )
        processed_list.extend(newly_processed)
        all_alerts.extend(alerts)

        raw_processed_list = state["raw_processed"].setdefault(tf_name, [])
        raw_alerts, raw_newly_processed = evaluate_raw_impulses(tf_name, df_big, raw_processed_list)
        raw_processed_list.extend(raw_newly_processed)
        all_raw_alerts.extend(raw_alerts)

        sweep_processed_list = state["sweep_processed"].setdefault(tf_name, [])
        sweep_alerts, sweep_newly_processed = evaluate_sweep_setups(tf_name, df_big, sweep_processed_list)
        sweep_processed_list.extend(sweep_newly_processed)
        all_sweep_alerts.extend(sweep_alerts)

    # Instant, unconfirmed alerts — fire first since they're the faster signal
    for alert in all_raw_alerts:
        direction = "possible SHORT setup forming" if alert["color"] == "green" else "possible LONG setup forming"
        tf_minutes = TIMEFRAMES[alert["tf"]]["minutes"]
        msg = (
            f"⚡ RAW IMPULSE — {alert['tf']} timeframe (NOT confirmed yet)\n"
            f"Impulse candle: {alert['color'].upper()} ({alert['body_points']} pts)\n"
            f"Candle closed: {candle_close_ist_str(alert['time'], tf_minutes)}\n"
            f"{direction}\n"
            f"Watch for target (candle open): {alert['open']}\n"
            f"Impulse close: {alert['close']}\n"
            f"(A confirmed 🚨 alert will follow separately if the lower-timeframe reversal confirms.)"
        )
        print(msg)
        send_telegram(msg)

    for alert in all_sweep_alerts:
        direction = "SHORT (fade back down)" if alert["color"] == "green" else "LONG (fade back up)"
        tf_minutes = TIMEFRAMES[alert["tf"]]["minutes"]
        msg = (
            f"🎯 SWEEP + SHALLOW CLOSE — {alert['tf']} timeframe\n"
            f"Impulse candle: {alert['color'].upper()} ({alert['body_points']} pts)\n"
            f"Candle closed: {candle_close_ist_str(alert['time'], tf_minutes)}\n"
            f"Next candle swept the extreme, then closed back at only "
            f"{alert['next_close_retracement_pct']}% retracement\n"
            f"Bias: {direction}\n"
            f"Target (candle open): {alert['open']}\n"
            f"Impulse close: {alert['close']}"
        )
        print(msg)
        send_telegram(msg)

    for alert in all_alerts:
        direction = "SHORT (fade back down)" if alert["color"] == "green" else "LONG (fade back up)"
        tf_minutes = TIMEFRAMES[alert["tf"]]["minutes"]
        msg = (
            f"🚨 MNQ Setup — {alert['tf']} timeframe\n"
            f"Impulse candle: {alert['color'].upper()} ({alert['body_points']} pts)\n"
            f"Candle closed: {candle_close_ist_str(alert['time'], tf_minutes)}\n"
            f"Bias: {direction}\n"
            f"Target (candle open): {alert['open']}\n"
            f"Impulse close: {alert['close']}"
        )
        print(msg)
        send_telegram(msg)

    if not all_alerts and not all_raw_alerts and not all_sweep_alerts:
        print("No new setups this run.")

    save_state(state)


if __name__ == "__main__":
    main()
