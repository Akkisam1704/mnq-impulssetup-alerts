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

BASE_URL = "https://api.topstepx.com"
STATE_FILE = "state.json"

MIN_BODY_POINTS = 50
MAX_WICK_PCT = 0.25
MAX_PROCESSED_PER_TF = 300  # cap state file growth

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
            return json.load(f)
    return {"processed": {tf: [] for tf in TIMEFRAMES}}


def save_state(state):
    for tf in state["processed"]:
        state["processed"][tf] = state["processed"][tf][-MAX_PROCESSED_PER_TF:]
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
            newly_processed.append(ts_str)
            continue

        upper_wick = row["h"] - max(row["o"], row["c"])
        lower_wick = min(row["o"], row["c"]) - row["l"]
        total_wick = upper_wick + lower_wick
        if total_wick > MAX_WICK_PCT * rng:
            newly_processed.append(ts_str)
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

    for alert in all_alerts:
        direction = "SHORT (fade back down)" if alert["color"] == "green" else "LONG (fade back up)"
        msg = (
            f"🚨 MNQ Setup — {alert['tf']} timeframe\n"
            f"Impulse candle: {alert['color'].upper()} ({alert['body_points']} pts)\n"
            f"Time: {alert['time']}\n"
            f"Bias: {direction}\n"
            f"Target (candle open): {alert['open']}\n"
            f"Impulse close: {alert['close']}"
        )
        print(msg)
        send_telegram(msg)

    if not all_alerts:
        print("No new setups this run.")

    save_state(state)


if __name__ == "__main__":
    main()
