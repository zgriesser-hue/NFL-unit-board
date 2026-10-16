"""Game-by-game weather report for the NFL games still to be played this week, emailed through Gmail.

Reads boards/unit_board_latest.csv (this week's games) and, if present, boards/totals_board_latest.csv (roof and the model's weather
adjustment). Forecasts come from the free Open-Meteo service (no key): temperature, wind, gusts and rain chance from kickoff to about
three hours after. Games in a dome are listed as indoors. Flags: wind 15+ mph or gusts 25+, temperature 32 F or colder, rain/snow 50%+.

Email settings (GitHub secrets): GMAIL_USER (the sending Gmail address), GMAIL_APP_PASSWORD (a Google "app password", not the normal
password), WEATHER_EMAIL_TO (optional; defaults to GMAIL_USER, comma-separate several). With no secrets it just prints the report and
saves boards/weather_report_latest.md. Set WEATHER_DRY_RUN=1 to never send.
"""
import html
import os
import smtplib
import sys
from email.message import EmailMessage

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NFL_ROOT", HERE)
BOARD = f"{ROOT}/boards"
ET = "America/New_York"
STADIUMS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4009), "BAL": (39.2780, -76.6227), "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167), "CIN": (39.0955, -84.5161), "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945), "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373), "KC": (39.0489, -94.4839),
    "LV": (36.0909, -115.1833), "LAC": (33.9535, -118.3392), "LA": (33.9535, -118.3392), "MIA": (25.9580, -80.2389),
    "MIN": (44.9738, -93.2577), "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158), "SF": (37.4030, -121.9697),
    "SEA": (47.5952, -122.3316), "TB": (27.9759, -82.5033), "TEN": (36.1665, -86.7713), "WAS": (38.9076, -76.8645)}
DOMES = {"DET", "LV", "MIN", "NO", "LA", "LAC"}            # fixed roof or canopy: weather does not matter
RETRACTABLE = {"ARI", "ATL", "DAL", "HOU", "IND"}                          # roof is usually closed in bad weather; check before kickoff
SKY = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog", 51: "light drizzle", 53: "drizzle",
       55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
       66: "freezing rain", 67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
       80: "rain showers", 81: "rain showers", 82: "heavy showers", 85: "snow showers", 86: "heavy snow showers",
       95: "thunderstorms", 96: "thunderstorms, hail", 99: "thunderstorms, hail"}
WIND_FLAG, GUST_FLAG, COLD_FLAG, RAIN_FLAG = 15.0, 25.0, 32.0, 50.0


def log(m):
    print(m, flush=True)


def forecast(lat, lon):
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=30, params={
        "latitude": lat, "longitude": lon, "timezone": "UTC", "forecast_days": 10, "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph", "precipitation_unit": "inch",
        "hourly": "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability,precipitation,weather_code"})
    r.raise_for_status()
    h = r.json()["hourly"]
    d = pd.DataFrame(h)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    return d.set_index("time")


def games():
    b = pd.read_csv(f"{BOARD}/unit_board_latest.csv")
    b["kick"] = pd.to_datetime(b["kickoff_utc"], utc=True)
    b = b[b["kick"] > pd.Timestamp.now(tz="UTC")].sort_values("kick")
    extra = {}
    tp = f"{BOARD}/totals_board_latest.csv"
    if os.path.exists(tp):
        t = pd.read_csv(tp)
        extra = t.set_index("game_id")[["weather_src", "Weather", "market_total"]].to_dict("index") if "Weather" in t else {}
    return b, extra


def build():
    b, extra = games()
    cache, rows = {}, []
    for g in b.itertuples():
        home, away = g.home_team, g.away_team
        kick_et = g.kick.tz_convert(ET)
        row = {"game": f"{away} at {home}", "when": kick_et.strftime("%a %-I:%M %p ET"), "flags": [], "model": ""}
        ex = extra.get(g.game_id, {})
        if pd.notna(ex.get("Weather", float("nan"))):
            row["model"] = f"{ex['Weather']:+.1f} pts on the total"
        if home in DOMES or ex.get("weather_src") == "dome":
            row.update(kind="dome", summary="Indoors (dome)")
            rows.append(row)
            continue
        if home not in STADIUMS:
            continue
        try:
            if home not in cache:
                cache[home] = forecast(*STADIUMS[home])
            f = cache[home]
            w = f.loc[g.kick.floor("h"):g.kick.floor("h") + pd.Timedelta(hours=3)]
            if w.empty:
                row.update(kind="none", summary="No forecast yet (more than 10 days out)")
                rows.append(row)
                continue
            t0, t1 = float(w["temperature_2m"].iloc[0]), float(w["temperature_2m"].iloc[-1])
            wind, gust = float(w["wind_speed_10m"].max()), float(w["wind_gusts_10m"].max())
            pp, rain = float(w["precipitation_probability"].max()), float(w["precipitation"].sum())
            code = int(w["weather_code"].iloc[min(1, len(w) - 1)])
            row.update(kind="outdoor", temp=f"{t0:.0f} to {t1:.0f} F" if round(t0) != round(t1) else f"{t0:.0f} F", wind=f"{wind:.0f} mph (gusts {gust:.0f})",
                       rain=f"{pp:.0f}% ({rain:.2f} in)", sky=SKY.get(code, "n/a"))
            if wind >= WIND_FLAG or gust >= GUST_FLAG:
                row["flags"].append("WIND")
            if min(t0, t1) <= COLD_FLAG:
                row["flags"].append("COLD")
            if pp >= RAIN_FLAG:
                row["flags"].append("RAIN/SNOW")
            row["summary"] = ""
            if home in RETRACTABLE:
                row["summary"] = "Retractable roof: usually closed in bad weather, check before kickoff"
        except Exception as e:
            row.update(kind="none", summary=f"Forecast unavailable ({str(e)[:60]})")
        rows.append(row)
    return rows


def render(rows):
    now = pd.Timestamp.now(tz=ET).strftime("%a %b %-d, %-I:%M %p ET")
    flagged = [r for r in rows if r["flags"]]
    lines = [f"NFL weather report, {now}", ""]
    lines.append(("Games to watch: " + "; ".join(f"{r['game']} ({', '.join(r['flags'])})" for r in flagged)) if flagged else "No weather flags: nothing here should move a game.")
    lines.append("")
    for r in rows:
        if r.get("kind") == "outdoor":
            s = f"{r['game']}  {r['when']}\n   {r['temp']}, wind {r['wind']}, rain chance {r['rain']}, {r['sky']}"
            if r["flags"]:
                s += "\n   FLAG: " + ", ".join(r["flags"])
        else:
            s = f"{r['game']}  {r['when']}\n   {r.get('summary', '')}"
        if r.get("summary") and r.get("kind") == "outdoor":
            s += "\n   " + r["summary"]
        if r["model"]:
            s += f"\n   Our totals model: {r['model']}"
        lines.append(s)
    lines += ["", "Forecast is for kickoff to about 3 hours later (Open-Meteo). Flags: wind 15+ mph or gusts 25+, 32 F or colder, 50%+ rain/snow.",
              "International and neutral-site games use the listed home team's stadium: check those yourself."]
    text = "\n".join(lines)
    td = "style='padding:6px 8px;border-bottom:1px solid #ddd;vertical-align:top'"
    body = ""
    for r in rows:
        bg = " style='background:#fff3cd'" if r["flags"] else ""
        if r.get("kind") == "outdoor":
            detail = f"{html.escape(r['temp'])}<br>wind {html.escape(r['wind'])}<br>rain {html.escape(r['rain'])}<br>{html.escape(r['sky'])}"
            if r.get("summary"):
                detail += f"<br><i>{html.escape(r['summary'])}</i>"
        else:
            detail = html.escape(r.get("summary", ""))
        flag = "<b>" + ", ".join(r["flags"]) + "</b>" if r["flags"] else ""
        body += (f"<tr{bg}><td {td}><b>{html.escape(r['game'])}</b><br>{html.escape(r['when'])}</td><td {td}>{detail}</td>"
                 f"<td {td}>{flag}</td><td {td}>{html.escape(r['model'])}</td></tr>")
    head = "<tr>" + "".join(f"<th style='text-align:left;padding:6px 8px;border-bottom:2px solid #999'>{h}</th>" for h in ("Game", "Forecast at kickoff", "Flag", "Model effect")) + "</tr>"
    summ = html.escape(lines[2])
    page = (f"<div style='font-family:Arial,sans-serif'><h3>NFL weather report</h3><p>{html.escape(now)}</p><p><b>{summ}</b></p>"
            f"<table style='border-collapse:collapse;font-size:14px'>{head}{body}</table>"
            f"<p style='color:#666;font-size:12px'>Forecast covers kickoff to about 3 hours later (Open-Meteo). Flags: wind 15+ mph or gusts 25+, 32 F or colder, 50%+ rain/snow. "
            f"Neutral-site games use the listed home stadium: check those yourself.</p></div>")
    return text, page, ("; ".join(r["game"] for r in flagged))


def send(text, page, flagged):
    user, pw = os.environ.get("GMAIL_USER", "").strip(), os.environ.get("GMAIL_APP_PASSWORD", "").strip()
    to = os.environ.get("WEATHER_EMAIL_TO", "").strip() or user
    if os.environ.get("WEATHER_DRY_RUN") == "1" or not (user and pw and to):
        log("Not emailing (no GMAIL_USER / GMAIL_APP_PASSWORD secrets, or dry run). Report saved to boards/weather_report_latest.md")
        return False
    m = EmailMessage()
    m["Subject"] = "NFL weather report" + (f": {len(flagged.split('; '))} game(s) flagged" if flagged else ": no flags")
    m["From"], m["To"] = user, to
    m.set_content(text)
    m.add_alternative(page, subtype="html")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=60) as s:
        s.login(user, pw)
        s.send_message(m)
    log(f"Emailed the report to {to}")
    return True


def main():
    rows = build()
    if not rows:
        log("No games left to report on this week.")
        return 0
    text, page, flagged = render(rows)
    print(text)
    os.makedirs(BOARD, exist_ok=True)
    with open(f"{BOARD}/weather_report_latest.md", "w") as f:
        f.write(text + "\n")
    send(text, page, flagged)
    return 0


if __name__ == "__main__":
    sys.exit(main())
