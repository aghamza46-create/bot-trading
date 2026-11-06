"""
BOT DE SIGNAUX POUR METATRADER 5 MOBILE (100 % gratuit)
=======================================================
Le bot analyse le forex et l'or (données Yahoo Finance), calcule le risque
(EWMA, ATR, VaR/CVaR, probabilité TP-avant-SL, Kelly) et vous envoie sur
Telegram : sens, entrée, stop, take profit et taille de lot conseillée.
Vous passez ensuite l'ordre à la main dans MetaTrader 5.

    python signals_bot.py          -> analyse réelle (+ Telegram si configuré)
    python signals_bot.py --demo   -> test avec données synthétiques, sans internet

Variables d'environnement (toutes optionnelles) :
    CAPITAL (défaut 1000, en USD), LEVERAGE (défaut 100),
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""

import math
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# =============================== CONFIG ===============================
CAPITAL = float(os.getenv("CAPITAL", "1000"))       # capital du compte (USD)
LEVERAGE = float(os.getenv("LEVERAGE", "100"))      # levier du compte
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

INTERVALLE = "15m"
PERIODE = "5d"
FRAICHEUR_MAX_MIN = 45       # ignore les données plus vieilles (marché fermé)

SEUIL_SCORE = 0.60
RSI_MAX = 75
K_STOP = 2.0                 # stop = K x ATR
RATIO_RR = 2.0               # TP = RR x stop
RISQUE_PAR_TRADE = 0.01      # 1 % du capital
FRACTION_KELLY = 0.25
VAR_MAX_PCT = 0.02
MARGE_MAX_PCT = 0.20
HORIZON = 10
LAMBDA_EWMA = 0.94
RETRECISSEMENT_DERIVE = 0.5
LOT_MIN, LOT_PAS = 0.01, 0.01

# contract = taille du contrat, base_usd = True si la devise de base est le USD
# cost = coût aller-retour estimé (spread + commission) en fraction du prix
# conv = "usd" (cotation en USD) ou "div" (cotation en JPY/CAD/CHF : on divise par le prix)
INSTRUMENTS = {
    "EURUSD": dict(ticker="EURUSD=X", contract=100000, base_usd=False, conv="usd", cost=0.00015, pip=0.0001),
    "GBPUSD": dict(ticker="GBPUSD=X", contract=100000, base_usd=False, conv="usd", cost=0.00020, pip=0.0001),
    "AUDUSD": dict(ticker="AUDUSD=X", contract=100000, base_usd=False, conv="usd", cost=0.00025, pip=0.0001),
    "NZDUSD": dict(ticker="NZDUSD=X", contract=100000, base_usd=False, conv="usd", cost=0.00030, pip=0.0001),
    "USDJPY": dict(ticker="JPY=X",    contract=100000, base_usd=True,  conv="div", cost=0.00015, pip=0.01),
    "USDCAD": dict(ticker="CAD=X",    contract=100000, base_usd=True,  conv="div", cost=0.00025, pip=0.0001),
    "USDCHF": dict(ticker="CHF=X",    contract=100000, base_usd=True,  conv="div", cost=0.00025, pip=0.0001),
    "XAUUSD": dict(ticker="GC=F",     contract=100,    base_usd=False, conv="usd", cost=0.00040, pip=0.1),
}
# ======================================================================


# ----------------------------- INDICATEURS -----------------------------
def calculer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    df["ret"] = np.log(c).diff()
    df["ema_f"] = c.ewm(span=12, adjust=False).mean()
    df["ema_s"] = c.ewm(span=48, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    perte = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / perte.replace(0, np.nan))
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - c.shift()).abs(),
                    (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    vol = df["volume"].replace(0, np.nan)
    df["vol_ratio"] = (vol / vol.rolling(20).mean()).fillna(1.0)
    return df


def score_signal(df: pd.DataFrame) -> float:
    """Score dans [-1 ; 1] : positif = achat, négatif = vente."""
    d = df.iloc[-1]
    sigma = df["ret"].tail(50).std()
    mom = math.log(d.close / df["close"].iloc[-21])
    mom_z = mom / (sigma * math.sqrt(20) + 1e-12)
    tend = 1.0 if d.ema_f > d.ema_s else -1.0
    rsi_c = (d.rsi - 50) / 50
    boost = math.tanh(max(d.vol_ratio - 1, 0))
    return 0.40 * tend + 0.35 * math.tanh(mom_z) + 0.15 * rsi_c + 0.10 * boost * tend


# ----------------------------- MOTEUR DE RISQUE -----------------------------
def volatilite_ewma(rets):
    return math.sqrt((rets ** 2).ewm(alpha=1 - LAMBDA_EWMA, adjust=False).mean().iloc[-1])


def var_cvar(rets, niveau=0.95):
    r = rets.dropna().tail(200).values
    q = np.percentile(r, (1 - niveau) * 100)
    queue = r[r <= q]
    cvar = -queue.mean() if len(queue) else -q
    ech = math.sqrt(HORIZON)
    return -q * ech, cvar * ech


def proba_tp_avant_sl(mu, sigma, a, b):
    """P(toucher +b avant -a) pour un brownien avec dérive."""
    if sigma <= 0:
        return 0.5
    k = 2 * mu / sigma ** 2
    if abs(k * max(a, b)) < 1e-9:
        return a / (a + b)
    ka, kb = max(min(k * a, 50), -50), max(min(k * b, 50), -50)
    den = math.exp(-kb) - math.exp(ka)
    if den == 0:
        return 0.5
    return min(max((1 - math.exp(ka)) / den, 0.0), 1.0)


def evaluer(nom, cfg, df):
    """Retourne (signal | None, raison, score)."""
    d = df.iloc[-1]
    prix, atr = float(d.close), float(d.atr)
    s = score_signal(df)
    sens = 1 if s > 0 else -1

    if abs(s) < SEUIL_SCORE:
        return None, "score insuffisant", s
    if (sens > 0 and d.rsi > RSI_MAX) or (sens < 0 and d.rsi < 100 - RSI_MAX):
        return None, "RSI en zone extrême", s

    dist = K_STOP * atr
    a, b = dist / prix, RATIO_RR * dist / prix
    rets = df["ret"].dropna()
    sigma = volatilite_ewma(rets)
    mu = sens * RETRECISSEMENT_DERIVE * rets.tail(50).mean()
    p = proba_tp_avant_sl(mu, sigma, a, b)

    cout = cfg["cost"]
    esperance = p * b - (1 - p) * a - cout
    if esperance <= 0:
        return None, f"espérance nette négative ({esperance:+.3%})", s
    R = (b - cout) / (a + cout)
    kelly = p - (1 - p) / R if R > 0 else -1
    if kelly <= 0:
        return None, f"Kelly <= 0 ({kelly:+.2f})", s

    # --- taille de position (lots) ---
    conv = 1.0 if cfg["conv"] == "usd" else 1.0 / prix
    perte_par_lot = dist * cfg["contract"] * conv                   # USD perdus au stop pour 1 lot
    risque_usd = CAPITAL * min(RISQUE_PAR_TRADE, kelly * FRACTION_KELLY)
    lots = risque_usd / perte_par_lot

    var, cvar = var_cvar(rets)
    var_par_lot = var * prix * cfg["contract"] * conv
    if var_par_lot > 0:
        lots = min(lots, CAPITAL * VAR_MAX_PCT / var_par_lot)

    notionnel_lot = cfg["contract"] * (1.0 if cfg["base_usd"] else prix)
    marge_lot = notionnel_lot / LEVERAGE
    lots = min(lots, CAPITAL * MARGE_MAX_PCT / marge_lot)

    lots = math.floor(lots / LOT_PAS) * LOT_PAS
    if lots < LOT_MIN:
        return None, "capital insuffisant pour le lot minimum avec ces limites de risque", s
    lots = round(lots, 2)

    return dict(
        nom=nom, sens=sens, prix=prix, stop=prix - sens * dist, tp=prix + sens * RATIO_RR * dist,
        dist=dist, pips=dist / cfg["pip"], lots=lots, risque_reel=lots * perte_par_lot,
        p=p, esperance=esperance, kelly=kelly, var=var, cvar=cvar, score=s,
    ), "OK", s


# ----------------------------- DONNÉES -----------------------------
def telecharger(cfg):
    import yfinance as yf
    df = yf.download(cfg["ticker"], period=PERIODE, interval=INTERVALLE,
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    return df


def donnees_demo(seed):
    """Marché synthétique en tendance, pour tester sans internet."""
    rng = np.random.default_rng(seed)
    n = 300
    rets = rng.normal(0.0004, 0.0006, n)
    close = 1.10 * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0003, n)))
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="15min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.integers(100, 1000, n)}, index=idx)


# ----------------------------- TELEGRAM -----------------------------
def formater(sig):
    sens = "🟢 ACHAT" if sig["sens"] > 0 else "🔴 VENTE"
    dec = 2 if sig["prix"] > 50 else 5
    return (
        f"{sens} {sig['nom']}\n"
        f"Entrée ≈ {sig['prix']:.{dec}f}\n"
        f"Stop loss : {sig['stop']:.{dec}f} (≈ {sig['pips']:.0f} pips)\n"
        f"Take profit : {sig['tp']:.{dec}f}\n"
        f"Lot conseillé : {sig['lots']:.2f} (risque ≈ {sig['risque_reel']:.2f} $)\n"
        f"P(TP avant SL) : {sig['p']:.0%} | Espérance : {sig['esperance']:+.2%}\n"
        f"Kelly : {sig['kelly']:.2f} | VaR95 : {sig['var']:.2%} | CVaR95 : {sig['cvar']:.2%}\n"
        f"Score : {sig['score']:+.2f}\n"
        f"⚠️ Prix indicatif (Yahoo) : vérifiez le prix de votre courtier et "
        f"gardez la même distance pour le SL et le TP."
    )


def envoyer_telegram(texte):
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": TELEGRAM_CHAT_ID, "text": texte}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=20)
    except Exception as e:
        print(f"Erreur Telegram : {e}")


# ----------------------------- PROGRAMME -----------------------------
def main():
    demo = "--demo" in sys.argv
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC] Scan de {len(INSTRUMENTS)} marchés "
          f"| capital={CAPITAL:.0f} $ | {'DÉMO' if demo else 'réel'}")
    nb = 0

    for i, (nom, cfg) in enumerate(INSTRUMENTS.items()):
        try:
            df = donnees_demo(i + 1) if demo else telecharger(cfg)
            if df is None or len(df) < 80:
                print(f"{nom}: données insuffisantes")
                continue

            if not demo:
                df = df.iloc[:-1]                                  # bougie clôturée uniquement
                age = (datetime.now(timezone.utc) - df.index[-1].tz_convert("UTC")).total_seconds() / 60
                if age > FRAICHEUR_MAX_MIN + 15:
                    print(f"{nom}: marché fermé ou données trop anciennes ({age:.0f} min)")
                    continue

            df = calculer(df)
            sig, raison, s = evaluer(nom, cfg, df)
            if sig is None:
                print(f"{nom}: pas de trade ({raison}, score {s:+.2f})")
                continue

            # Anti-doublon sans mémoire : on n'alerte que si le signal vient d'apparaître
            if not demo:
                s_prec = score_signal(df.iloc[:-1])
                if abs(s_prec) >= SEUIL_SCORE and (s_prec > 0) == (s > 0):
                    print(f"{nom}: signal déjà envoyé précédemment")
                    continue

            msg = formater(sig)
            print("\n" + msg + "\n")
            envoyer_telegram(msg)
            nb += 1
        except Exception as e:
            print(f"{nom}: erreur ({e})")

    print(f"Terminé : {nb} signal(s).")


if __name__ == "__main__":
    main()
