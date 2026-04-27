"""
Trade Plan HTML Report Generator
==================================
Converts a trade_plan_{TICKER}.json file into a self-contained HTML report with:

  • Summary cards — strategy, regime, account balance, approval status
  • Spread structure diagram — visual strike ladder (puts left, calls right)
  • Legs table — symbol, strike, action, delta, bid/ask
  • Risk check breakdown — colour-coded pass/fail list
  • History timeline — all cycles with expand/collapse
  • Chart.js charts — net credit trend, ratio trend, account balance trend
  • Trade thesis — why / why_now / exit_plan (if present)
  • Order details — order ID, fill status (if submitted)

Usage
-----
  # From the command line (generate HTML next to the JSON):
  python -m trading_agent.trade_plan_report trade_plans/trade_plan_IWM.json

  # Generate for all tickers in a directory:
  python -m trading_agent.trade_plan_report trade_plans/

  # Programmatically (called from OrderExecutor after each save):
  from trading_agent.trade_plan_report import generate_report
  generate_report("trade_plans/trade_plan_IWM.json")
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_report(json_path: str) -> str:
    """
    Read *json_path* and write a companion HTML file next to it.
    Returns the path to the generated HTML file.
    """
    with open(json_path) as fh:
        data = json.load(fh)

    html = _build_html(data, json_path)
    html_path = json_path.replace(".json", ".html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return html_path


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def _build_html(data: Dict, source_path: str) -> str:
    ticker = data.get("ticker", "UNKNOWN")
    created = _fmt_ts(data.get("created", ""))
    last_updated = _fmt_ts(data.get("last_updated", ""))
    history: List[Dict] = data.get("state_history", [])

    latest = history[-1] if history else {}
    latest_plan = latest.get("trade_plan", {})
    latest_verdict = latest.get("risk_verdict", {})
    latest_order = latest.get("order_result", {})
    approved = latest_verdict.get("approved", False)
    mode = latest.get("mode", "—")

    # Chart data — walk history for trend lines
    chart_labels, chart_credits, chart_ratios, chart_balances = [], [], [], []
    for entry in history:
        tp = entry.get("trade_plan", {})
        rv = entry.get("risk_verdict", {})
        if tp.get("net_credit") is not None:
            chart_labels.append(_fmt_ts(entry.get("timestamp", ""), short=True))
            chart_credits.append(tp.get("net_credit", 0))
            chart_ratios.append(round(tp.get("credit_to_width_ratio", 0) * 100, 2))
            chart_balances.append(rv.get("account_balance", 0))

    status_class = "approved" if approved else "rejected"
    status_label = "APPROVED" if approved else "REJECTED"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{ticker} Trade Plan</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --surface2: #21262d;
    --border: #30363d; --text: #e6edf3; --muted: #8b949e;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
    --blue: #58a6ff; --purple: #bc8cff; --orange: #ffa657;
    --approved: #1f6feb; --card-radius: 10px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace; font-size: 14px; line-height: 1.6; }}
  a {{ color: var(--blue); text-decoration: none; }}

  /* Layout */
  .page {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px 60px; }}
  .header {{ display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 28px; border-bottom: 1px solid var(--border); padding-bottom: 16px; }}
  .header-left h1 {{ font-size: 28px; font-weight: 700; letter-spacing: -0.5px; }}
  .header-left h1 span {{ color: var(--blue); }}
  .header-meta {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
  .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; letter-spacing: 0.5px; }}
  .badge.approved {{ background: rgba(63,185,80,0.15); color: var(--green); border: 1px solid var(--green); }}
  .badge.rejected {{ background: rgba(248,81,73,0.15); color: var(--red); border: 1px solid var(--red); }}
  .badge.dry_run {{ background: rgba(88,166,255,0.15); color: var(--blue); border: 1px solid var(--blue); }}
  .badge.live {{ background: rgba(255,166,87,0.15); color: var(--orange); border: 1px solid var(--orange); }}
  .badge.submitted {{ background: rgba(63,185,80,0.15); color: var(--green); border: 1px solid var(--green); }}

  /* Cards */
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin-bottom: 24px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--card-radius); padding: 16px; }}
  .card-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px; }}
  .card-value {{ font-size: 20px; font-weight: 700; }}
  .card-value.green {{ color: var(--green); }}
  .card-value.red {{ color: var(--red); }}
  .card-value.blue {{ color: var(--blue); }}
  .card-value.yellow {{ color: var(--yellow); }}
  .card-sub {{ color: var(--muted); font-size: 11px; margin-top: 4px; }}

  /* Section */
  .section {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--card-radius); margin-bottom: 20px; overflow: hidden; }}
  .section-header {{ padding: 14px 18px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 10px; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); }}
  .section-header .icon {{ font-size: 16px; }}
  .section-body {{ padding: 18px; }}

  /* Table */
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }}
  td {{ padding: 10px; border-bottom: 1px solid var(--border); font-size: 13px; vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: var(--surface2); }}
  .tag-sell {{ background: rgba(248,81,73,0.15); color: var(--red); padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }}
  .tag-buy {{ background: rgba(63,185,80,0.15); color: var(--green); padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }}
  .tag-put {{ background: rgba(188,140,255,0.15); color: var(--purple); padding: 2px 8px; border-radius: 4px; font-size: 11px; }}
  .tag-call {{ background: rgba(255,166,87,0.15); color: var(--orange); padding: 2px 8px; border-radius: 4px; font-size: 11px; }}
  .mono {{ font-family: monospace; font-size: 12px; color: var(--blue); }}

  /* Risk checks */
  .checks {{ display: flex; flex-direction: column; gap: 8px; }}
  .check {{ display: flex; align-items: flex-start; gap: 10px; padding: 8px 10px; border-radius: 6px; font-size: 13px; }}
  .check.pass {{ background: rgba(63,185,80,0.08); }}
  .check.fail {{ background: rgba(248,81,73,0.08); }}
  .check .icon {{ font-size: 14px; margin-top: 1px; flex-shrink: 0; }}
  .check.pass .icon {{ color: var(--green); }}
  .check.fail .icon {{ color: var(--red); }}

  /* Spread diagram */
  .spread-diagram {{ display: flex; flex-direction: column; gap: 16px; }}
  .spread-side {{ border: 1px solid var(--border); border-radius: 8px; padding: 14px; }}
  .spread-side h4 {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); margin-bottom: 12px; }}
  .strike-ladder {{ display: flex; flex-direction: column; gap: 8px; }}
  .strike-row {{ display: flex; align-items: center; gap: 12px; }}
  .strike-bar-wrap {{ flex: 1; background: var(--surface2); border-radius: 4px; height: 28px; position: relative; overflow: hidden; }}
  .strike-bar {{ height: 100%; border-radius: 4px; display: flex; align-items: center; padding: 0 10px; font-size: 12px; font-weight: 600; white-space: nowrap; }}
  .strike-bar.sell-put {{ background: rgba(248,81,73,0.25); color: var(--red); }}
  .strike-bar.buy-put  {{ background: rgba(188,140,255,0.20); color: var(--purple); }}
  .strike-bar.sell-call {{ background: rgba(248,81,73,0.25); color: var(--red); }}
  .strike-bar.buy-call  {{ background: rgba(255,166,87,0.20); color: var(--orange); }}
  .strike-label {{ width: 60px; text-align: right; font-weight: 700; font-size: 13px; }}
  .strike-meta {{ width: 120px; text-align: right; color: var(--muted); font-size: 12px; }}

  /* History */
  details {{ border: 1px solid var(--border); border-radius: 8px; margin-bottom: 10px; overflow: hidden; }}
  summary {{ padding: 12px 16px; cursor: pointer; display: flex; align-items: center; gap: 10px; background: var(--surface2); user-select: none; font-size: 13px; list-style: none; }}
  summary::-webkit-details-marker {{ display: none; }}
  summary::before {{ content: "▶"; font-size: 10px; color: var(--muted); transition: transform 0.2s; }}
  details[open] summary::before {{ transform: rotate(90deg); }}
  summary:hover {{ background: var(--border); }}
  .hist-body {{ padding: 16px; background: var(--surface); border-top: 1px solid var(--border); }}
  .hist-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
  @media (max-width: 640px) {{ .hist-grid {{ grid-template-columns: 1fr; }} }}
  .hist-block {{ background: var(--surface2); border-radius: 6px; padding: 12px; }}
  .hist-block h5 {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); margin-bottom: 8px; }}
  .hist-kv {{ display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid var(--border); font-size: 12px; }}
  .hist-kv:last-child {{ border-bottom: none; }}
  .hist-kv .k {{ color: var(--muted); }}
  .hist-kv .v {{ font-weight: 600; }}

  /* Thesis */
  .thesis-block {{ display: flex; flex-direction: column; gap: 12px; }}
  .thesis-item {{ border-left: 3px solid var(--blue); padding: 10px 14px; background: var(--surface2); border-radius: 0 6px 6px 0; }}
  .thesis-item h5 {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--blue); margin-bottom: 4px; }}
  .thesis-item p {{ font-size: 13px; color: var(--text); line-height: 1.5; }}

  /* Charts */
  .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }}
  .chart-wrap {{ background: var(--surface2); border-radius: 8px; padding: 16px; }}
  .chart-wrap h5 {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }}

  /* Order detail */
  .order-detail {{ background: var(--surface2); border-radius: 8px; padding: 14px; }}
  .order-kv {{ display: flex; gap: 12px; padding: 5px 0; border-bottom: 1px solid var(--border); font-size: 13px; }}
  .order-kv:last-child {{ border-bottom: none; }}
  .order-kv .k {{ color: var(--muted); min-width: 140px; }}
  .order-kv .v {{ font-weight: 600; font-family: monospace; }}

  /* Footer */
  .footer {{ color: var(--muted); font-size: 11px; text-align: center; margin-top: 40px; }}
</style>
</head>
<body>
<div class="page">

  <!-- HEADER -->
  <div class="header">
    <div class="header-left">
      <h1>&#9679; <span>{ticker}</span> Trade Plan</h1>
      <div class="header-meta">
        Created {created} &nbsp;·&nbsp; Last updated {last_updated}
        &nbsp;·&nbsp; {len(history)} cycle(s) recorded
        &nbsp;·&nbsp; Source: {os.path.basename(source_path)}
      </div>
    </div>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <span class="badge {status_class}">{status_label}</span>
      <span class="badge {mode}">{mode.upper()}</span>
      {_order_badge(latest_order)}
    </div>
  </div>

  <!-- SUMMARY CARDS -->
  {_cards(latest_plan, latest_verdict, latest_order, history)}

  <!-- SPREAD STRUCTURE -->
  {_spread_section(latest_plan)}

  <!-- LEGS TABLE -->
  {_legs_section(latest_plan)}

  <!-- RISK CHECKS -->
  {_risk_section(latest_verdict)}

  <!-- TRADE THESIS -->
  {_thesis_section(latest)}

  <!-- TREND CHARTS -->
  {_charts_section(chart_labels, chart_credits, chart_ratios, chart_balances)}

  <!-- ORDER DETAILS -->
  {_order_section(latest_order)}

  <!-- HISTORY -->
  {_history_section(history)}

  <div class="footer">
    Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    &nbsp;·&nbsp; Autonomous Options Trading Agent
  </div>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _cards(plan: Dict, verdict: Dict, order: Dict, history: List[Dict]) -> str:
    strategy = plan.get("strategy", "—")
    regime = plan.get("regime", "—").upper()
    credit = plan.get("net_credit", 0)
    max_loss = plan.get("max_loss", 0)
    ratio = plan.get("credit_to_width_ratio", 0)
    width = plan.get("spread_width", 0)
    expiry = plan.get("expiration", "—")
    balance = verdict.get("account_balance", 0)
    max_allowed = verdict.get("max_allowed_loss", 0)

    # Days to expiry
    dte_str = "—"
    if expiry and expiry != "—":
        try:
            exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
            dte = (exp_dt - datetime.now()).days
            dte_str = f"{dte}d"
        except Exception:
            dte_str = "—"

    # Win rate across history
    approved_count = sum(1 for e in history if e.get("risk_verdict", {}).get("approved"))
    win_pct = f"{approved_count}/{len(history)}" if history else "—"

    credit_color = "green" if credit > 0 else "red"
    ratio_color = "green" if ratio >= 0.33 else ("yellow" if ratio >= 0.25 else "red")
    balance_color = "blue"

    return f"""
<div class="cards">
  <div class="card">
    <div class="card-label">Strategy</div>
    <div class="card-value blue" style="font-size:16px">{strategy}</div>
    <div class="card-sub">Regime: {regime}</div>
  </div>
  <div class="card">
    <div class="card-label">Net Credit</div>
    <div class="card-value {credit_color}">${credit:.2f}</div>
    <div class="card-sub">Per contract × 100</div>
  </div>
  <div class="card">
    <div class="card-label">Max Loss</div>
    <div class="card-value red">${max_loss:.0f}</div>
    <div class="card-sub">Allowed: ${max_allowed:.0f}</div>
  </div>
  <div class="card">
    <div class="card-label">Credit / Width</div>
    <div class="card-value {ratio_color}">{ratio*100:.1f}%</div>
    <div class="card-sub">Width: ${width:.0f}</div>
  </div>
  <div class="card">
    <div class="card-label">Expiration</div>
    <div class="card-value" style="font-size:16px">{expiry}</div>
    <div class="card-sub">DTE: {dte_str}</div>
  </div>
  <div class="card">
    <div class="card-label">Account Equity</div>
    <div class="card-value {balance_color}">${balance:,.0f}</div>
    <div class="card-sub">Paper trading</div>
  </div>
  <div class="card">
    <div class="card-label">Cycles Approved</div>
    <div class="card-value blue">{win_pct}</div>
    <div class="card-sub">Approval rate</div>
  </div>
</div>"""


def _spread_section(plan: Dict) -> str:
    legs = plan.get("legs", [])
    if not legs:
        return ""

    puts = [l for l in legs if l.get("type") == "put"]
    calls = [l for l in legs if l.get("type") == "call"]

    def _ladder(side_legs, side):
        rows = []
        sorted_legs = sorted(side_legs, key=lambda x: x.get("strike", 0), reverse=True)
        for leg in sorted_legs:
            strike = leg.get("strike", 0)
            action = leg.get("action", "")
            delta = leg.get("delta", 0)
            bid = leg.get("bid", 0)
            ask = leg.get("ask", 0)
            bar_class = f"{action}-{side}"
            label = "SELL" if action == "sell" else "BUY"
            rows.append(f"""
        <div class="strike-row">
          <div class="strike-label">${strike:.0f}</div>
          <div class="strike-bar-wrap">
            <div class="strike-bar {bar_class}">{label} — Δ {delta:+.3f} &nbsp;|&nbsp; bid ${bid:.2f} / ask ${ask:.2f}</div>
          </div>
          <div class="strike-meta">mid ${(bid+ask)/2:.2f}</div>
        </div>""")
        return "\n".join(rows)

    put_html = f"""<div class="spread-side">
      <h4>📉 Put Side</h4>
      <div class="strike-ladder">{_ladder(puts, 'put')}</div>
    </div>""" if puts else ""

    call_html = f"""<div class="spread-side">
      <h4>📈 Call Side</h4>
      <div class="strike-ladder">{_ladder(calls, 'call')}</div>
    </div>""" if calls else ""

    reasoning = plan.get("reasoning", "")

    return f"""
<div class="section">
  <div class="section-header"><span class="icon">⚖️</span> Spread Structure</div>
  <div class="section-body">
    <div class="spread-diagram">
      {put_html}
      {call_html}
    </div>
    {f'<p style="margin-top:14px;color:var(--muted);font-size:12px;line-height:1.5">{reasoning}</p>' if reasoning else ''}
  </div>
</div>"""


def _legs_section(plan: Dict) -> str:
    legs = plan.get("legs", [])
    if not legs:
        return ""

    rows = []
    for leg in legs:
        symbol = leg.get("symbol", "—")
        strike = leg.get("strike", 0)
        action = leg.get("action", "")
        opt_type = leg.get("type", "")
        delta = leg.get("delta", 0)
        bid = leg.get("bid", 0)
        ask = leg.get("ask", 0)
        mid = (bid + ask) / 2

        rows.append(f"""<tr>
      <td class="mono">{symbol}</td>
      <td style="font-weight:700">${strike:.0f}</td>
      <td><span class="tag-{action}">{action.upper()}</span></td>
      <td><span class="tag-{opt_type}">{opt_type.upper()}</span></td>
      <td style="color:{'var(--red)' if delta < 0 else 'var(--green)'}">{delta:+.4f}</td>
      <td>${bid:.2f}</td>
      <td>${ask:.2f}</td>
      <td style="font-weight:600">${mid:.2f}</td>
    </tr>""")

    return f"""
<div class="section">
  <div class="section-header"><span class="icon">📋</span> Option Legs</div>
  <div class="section-body" style="padding:0">
    <table>
      <thead><tr>
        <th>Symbol</th><th>Strike</th><th>Action</th><th>Type</th>
        <th>Delta</th><th>Bid</th><th>Ask</th><th>Mid</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>
</div>"""


def _risk_section(verdict: Dict) -> str:
    passed = verdict.get("checks_passed", [])
    failed = verdict.get("checks_failed", [])
    summary = verdict.get("summary", "")

    items = []
    for msg in passed:
        items.append(f'<div class="check pass"><span class="icon">✅</span><span>{msg}</span></div>')
    for msg in failed:
        items.append(f'<div class="check fail"><span class="icon">❌</span><span>{msg}</span></div>')

    return f"""
<div class="section">
  <div class="section-header"><span class="icon">🛡️</span> Risk Checks &nbsp;
    <span style="font-size:12px;font-weight:400;color:var(--text)">{summary}</span>
  </div>
  <div class="section-body">
    <div class="checks">{"".join(items)}</div>
  </div>
</div>"""


def _thesis_section(entry: Dict) -> str:
    # Thesis can be in top-level entry (from agent _log_signal raw_signal)
    # or inside the trade_plan dict — check both places gracefully
    tp = entry.get("trade_plan", {})
    thesis = tp.get("thesis") or entry.get("thesis") or {}

    if not thesis:
        return ""

    why = thesis.get("why", "")
    why_now = thesis.get("why_now", "")
    exit_plan = thesis.get("exit_plan", "")

    items = []
    if why:
        items.append(f'<div class="thesis-item"><h5>Why this market?</h5><p>{why}</p></div>')
    if why_now:
        items.append(f'<div class="thesis-item"><h5>Why now?</h5><p>{why_now}</p></div>')
    if exit_plan:
        items.append(f'<div class="thesis-item"><h5>Exit plan</h5><p>{exit_plan}</p></div>')

    if not items:
        return ""

    return f"""
<div class="section">
  <div class="section-header"><span class="icon">💡</span> Trade Thesis</div>
  <div class="section-body">
    <div class="thesis-block">{"".join(items)}</div>
  </div>
</div>"""


def _charts_section(labels, credits, ratios, balances) -> str:
    if len(labels) < 2:
        return ""

    lbl_js = json.dumps(labels)
    cr_js = json.dumps(credits)
    ra_js = json.dumps(ratios)
    ba_js = json.dumps(balances)

    return f"""
<div class="section">
  <div class="section-header"><span class="icon">📊</span> Historical Trends</div>
  <div class="section-body">
    <div class="charts-grid">
      <div class="chart-wrap">
        <h5>Net Credit per Cycle ($)</h5>
        <canvas id="chartCredit" height="160"></canvas>
      </div>
      <div class="chart-wrap">
        <h5>Credit / Width Ratio (%)</h5>
        <canvas id="chartRatio" height="160"></canvas>
      </div>
      <div class="chart-wrap">
        <h5>Account Balance ($)</h5>
        <canvas id="chartBalance" height="160"></canvas>
      </div>
    </div>
  </div>
</div>
<script>
const chartDefaults = {{
  responsive: true,
  plugins: {{ legend: {{ display: false }}, tooltip: {{ mode: 'index', intersect: false }} }},
  scales: {{
    x: {{ ticks: {{ color: '#8b949e', font: {{ size: 10 }} }}, grid: {{ color: '#21262d' }} }},
    y: {{ ticks: {{ color: '#8b949e', font: {{ size: 10 }} }}, grid: {{ color: '#21262d' }} }}
  }}
}};

new Chart(document.getElementById('chartCredit'), {{
  type: 'line',
  data: {{
    labels: {lbl_js},
    datasets: [{{ data: {cr_js}, borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,0.1)',
      fill: true, tension: 0.3, pointRadius: 4, pointBackgroundColor: '#3fb950' }}]
  }},
  options: {{ ...chartDefaults }}
}});

new Chart(document.getElementById('chartRatio'), {{
  type: 'line',
  data: {{
    labels: {lbl_js},
    datasets: [
      {{ data: {ra_js}, borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.1)',
        fill: true, tension: 0.3, pointRadius: 4, pointBackgroundColor: '#58a6ff' }},
      {{ data: Array({len(labels)}).fill(33), borderColor: '#d29922',
        borderDash: [6,3], pointRadius: 0, label: 'Min 33%' }}
    ]
  }},
  options: {{ ...chartDefaults, plugins: {{ ...chartDefaults.plugins,
    legend: {{ display: true, labels: {{ color: '#8b949e', font: {{ size: 10 }} }} }} }} }}
}});

new Chart(document.getElementById('chartBalance'), {{
  type: 'line',
  data: {{
    labels: {lbl_js},
    datasets: [{{ data: {ba_js}, borderColor: '#bc8cff', backgroundColor: 'rgba(188,140,255,0.1)',
      fill: true, tension: 0.3, pointRadius: 4, pointBackgroundColor: '#bc8cff' }}]
  }},
  options: {{ ...chartDefaults }}
}});
</script>"""


def _order_section(order: Dict) -> str:
    if not order:
        return ""

    alpaca = order.get("alpaca_response", {})
    order_id = alpaca.get("id") or order.get("order_id", "—")
    status = alpaca.get("status") or order.get("status", "—")
    limit_price = alpaca.get("limit_price", "—")
    submitted_at = _fmt_ts(alpaca.get("submitted_at", ""))
    filled_at = _fmt_ts(alpaca.get("filled_at", "") or "") or "—"
    filled_qty = alpaca.get("filled_qty", "—")
    tif = alpaca.get("time_in_force", "—")
    legs = alpaca.get("legs", [])

    leg_rows = ""
    for lg in legs:
        sym = lg.get("symbol", "—")
        side = lg.get("side", "—")
        lg_status = lg.get("status", "—")
        filled = lg.get("filled_qty", "0")
        intent = lg.get("position_intent", "—")
        leg_rows += f"""<tr>
      <td class="mono">{sym}</td>
      <td><span class="tag-{side}">{side.upper()}</span></td>
      <td style="font-size:11px;color:var(--muted)">{intent}</td>
      <td>{filled}</td>
      <td style="color:var(--muted);font-size:12px">{lg_status}</td>
    </tr>"""

    legs_table = f"""
    <table style="margin-top:14px">
      <thead><tr><th>Symbol</th><th>Side</th><th>Intent</th><th>Filled Qty</th><th>Status</th></tr></thead>
      <tbody>{leg_rows}</tbody>
    </table>""" if leg_rows else ""

    return f"""
<div class="section">
  <div class="section-header"><span class="icon">🚀</span> Order Submission</div>
  <div class="section-body">
    <div class="order-detail">
      <div class="order-kv"><span class="k">Order ID</span><span class="v">{order_id}</span></div>
      <div class="order-kv"><span class="k">Status</span>
        <span class="v" style="color:{'var(--green)' if 'fill' in str(status) else 'var(--yellow)'}">{status}</span>
      </div>
      <div class="order-kv"><span class="k">Limit Price</span><span class="v">{limit_price}</span></div>
      <div class="order-kv"><span class="k">Time in Force</span><span class="v">{tif}</span></div>
      <div class="order-kv"><span class="k">Submitted At</span><span class="v">{submitted_at}</span></div>
      <div class="order-kv"><span class="k">Filled At</span><span class="v">{filled_at}</span></div>
      <div class="order-kv"><span class="k">Filled Qty</span><span class="v">{filled_qty}</span></div>
    </div>
    {legs_table}
  </div>
</div>"""


def _history_section(history: List[Dict]) -> str:
    if not history:
        return ""

    items = []
    # Show newest first
    for entry in reversed(history):
        tp = entry.get("trade_plan", {})
        rv = entry.get("risk_verdict", {})
        order = entry.get("order_result", {})
        run_id = entry.get("run_id", "—")
        ts = _fmt_ts(entry.get("timestamp", ""))
        approved = rv.get("approved", False)
        mode = entry.get("mode", "")
        strategy = tp.get("strategy", "—")
        regime = tp.get("regime", "—")
        credit = tp.get("net_credit", 0)
        ratio = tp.get("credit_to_width_ratio", 0)
        valid = tp.get("valid", False)
        rejection = tp.get("rejection_reason", "")
        summary = rv.get("summary", "")
        balance = rv.get("account_balance", 0)

        status_dot = "🟢" if approved else "🔴"
        mode_badge = f'<span class="badge {mode}" style="font-size:10px;padding:2px 8px">{mode.upper()}</span>'

        # Build collapsed detail blocks
        kv_plan = f"""
          <div class="hist-block">
            <h5>Trade Plan</h5>
            <div class="hist-kv"><span class="k">Strategy</span><span class="v">{strategy}</span></div>
            <div class="hist-kv"><span class="k">Regime</span><span class="v">{regime}</span></div>
            <div class="hist-kv"><span class="k">Net Credit</span><span class="v" style="color:var(--{'green' if credit>0 else 'red'})">${credit:.2f}</span></div>
            <div class="hist-kv"><span class="k">Credit/Width</span><span class="v">{ratio*100:.1f}%</span></div>
            <div class="hist-kv"><span class="k">Spread Width</span><span class="v">${tp.get('spread_width',0):.0f}</span></div>
            <div class="hist-kv"><span class="k">Max Loss</span><span class="v">${tp.get('max_loss',0):.0f}</span></div>
            <div class="hist-kv"><span class="k">Expiration</span><span class="v">{tp.get('expiration','—')}</span></div>
            <div class="hist-kv"><span class="k">Plan Valid</span><span class="v" style="color:var(--{'green' if valid else 'red'})">{valid}</span></div>
            {f'<div class="hist-kv"><span class="k">Rejection</span><span class="v" style="color:var(--red);font-size:11px">{rejection}</span></div>' if rejection else ''}
          </div>"""

        passed_html = "".join(f'<div style="color:var(--green);font-size:12px;padding:2px 0">✅ {p}</div>' for p in rv.get("checks_passed", []))
        failed_html = "".join(f'<div style="color:var(--red);font-size:12px;padding:2px 0">❌ {f}</div>' for f in rv.get("checks_failed", []))

        kv_risk = f"""
          <div class="hist-block">
            <h5>Risk Verdict — {summary}</h5>
            <div style="margin-bottom:4px">{passed_html}{failed_html}</div>
            <div class="hist-kv" style="margin-top:8px"><span class="k">Account Balance</span><span class="v">${balance:,.2f}</span></div>
            <div class="hist-kv"><span class="k">Max Allowed Loss</span><span class="v">${rv.get('max_allowed_loss',0):,.2f}</span></div>
          </div>"""

        order_html = ""
        if order:
            alpaca = order.get("alpaca_response", {})
            order_html = f"""
          <div class="hist-block" style="grid-column:1/-1">
            <h5>Order Result</h5>
            <div class="hist-kv"><span class="k">Status</span><span class="v" style="color:var(--green)">{order.get('status','—')}</span></div>
            <div class="hist-kv"><span class="k">Order ID</span><span class="v" style="font-family:monospace;font-size:11px">{order.get('order_id','—')}</span></div>
            <div class="hist-kv"><span class="k">Limit Price</span><span class="v">{alpaca.get('limit_price','—')}</span></div>
            <div class="hist-kv"><span class="k">Submitted At</span><span class="v">{_fmt_ts(alpaca.get('submitted_at',''))}</span></div>
          </div>"""

        # Legs mini-table
        legs = tp.get("legs", [])
        legs_html = ""
        if legs:
            rows = "".join(f'<tr><td class="mono" style="font-size:11px">{l.get("symbol","")}</td>'
                           f'<td><span class="tag-{l.get("action","")}">{l.get("action","").upper()}</span></td>'
                           f'<td><span class="tag-{l.get("type","")}">{l.get("type","").upper()}</span></td>'
                           f'<td style="font-weight:700">${l.get("strike",0):.0f}</td>'
                           f'<td style="color:var(--muted)">{l.get("delta",0):+.3f}</td>'
                           f'<td>${l.get("bid",0):.2f} / ${l.get("ask",0):.2f}</td></tr>'
                           for l in legs)
            legs_html = f"""
          <div class="hist-block" style="grid-column:1/-1">
            <h5>Legs ({len(legs)})</h5>
            <table><thead><tr><th>Symbol</th><th>Action</th><th>Type</th><th>Strike</th><th>Delta</th><th>Bid/Ask</th></tr></thead>
            <tbody>{rows}</tbody></table>
          </div>"""

        items.append(f"""
<details {'open' if entry == list(reversed(history))[0] else ''}>
  <summary>
    {status_dot} &nbsp; <strong>{ts}</strong> &nbsp; · &nbsp; {strategy} &nbsp; · &nbsp;
    <span style="color:var(--{'green' if credit > 0 else 'muted'})">${credit:.2f} credit</span>
    &nbsp; · &nbsp; {mode_badge}
    &nbsp; · &nbsp; <span style="color:var(--muted);font-size:11px">run_id: {run_id}</span>
  </summary>
  <div class="hist-body">
    <div class="hist-grid">
      {kv_plan}
      {kv_risk}
      {order_html}
      {legs_html}
    </div>
  </div>
</details>""")

    return f"""
<div class="section">
  <div class="section-header"><span class="icon">🕰️</span> Cycle History ({len(history)} runs)</div>
  <div class="section-body">
    {"".join(items)}
  </div>
</div>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_ts(ts: str, short: bool = False) -> str:
    if not ts:
        return "—"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if short:
            return dt.strftime("%m/%d %H:%M")
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return ts[:19] if ts else "—"


def _order_badge(order: Dict) -> str:
    if not order:
        return ""
    status = order.get("status", "")
    if status == "submitted":
        return '<span class="badge submitted">SUBMITTED</span>'
    return ""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m trading_agent.trade_plan_report <path_to_json_or_dir>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        files = [os.path.join(target, f) for f in os.listdir(target)
                 if f.startswith("trade_plan_") and f.endswith(".json")]
    else:
        files = [target]

    for f in sorted(files):
        try:
            out = generate_report(f)
            print(f"✓  {out}")
        except Exception as exc:
            print(f"✗  {f}: {exc}")
