"""
Generate project summary PDF — v5 as main model.
Run: python generate_report.py
Output: Saranga_Regime_Strategy_Report.pdf
"""

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
OUT_PDF = ROOT / "Saranga_Regime_Strategy_Report.pdf"

IMG_V5 = ROOT / "output_v5" / "equity_curves.png"
IMG_V6 = ROOT / "output_v6" / "equity_curves.png"
IMG_V2 = ROOT / "output_v2" / "equity_curves.png"


class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, "Regime-Based SPY Strategy - Project Summary", align="R")
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str):
        self.ln(4)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(20, 60, 100)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 60, 100)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 11)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def bullet(self, text: str):
        self.set_font("Helvetica", "", 11)
        self.multi_cell(0, 6, f"  -  {text}")
        self.ln(1)

    def add_image_safe(self, path: Path, caption: str, w: float = 190):
        if not path.exists():
            self.set_font("Helvetica", "I", 10)
            self.set_text_color(180, 0, 0)
            self.multi_cell(0, 6, f"[Missing image: {path.name}]")
            self.set_text_color(0, 0, 0)
            return
        self.ln(2)
        self.image(str(path), x=10, w=w)
        self.ln(2)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 5, caption, align="C")
        self.set_text_color(0, 0, 0)
        self.ln(4)


def build_pdf():
    pdf = Report()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(20, 60, 100)
    pdf.cell(0, 14, "Regime-Based SPY Strategy", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, "Statistical Jump Model + Absorption Ratio", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 8, "Main model: v5 Hybrid  |  Evaluation: Jul 2022 - Dec 2025", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)
    pdf.set_text_color(0, 0, 0)

    pdf.section_title("1. What We Did")
    pdf.body_text(
        "We built a regime-switching trading strategy on SPY. The model reads daily market "
        "signals, labels each day as Bull (invested) or Bear (cash), and compares strategy "
        "performance to Buy & Hold. Several versions were tested; v5 is the chosen main model. "
        "v2 and v6 are alternative designs kept for comparison."
    )

    pdf.section_title("2. Data Used")
    pdf.bullet("Primary file: Data/etf_ohlcv_20160301_20251231.csv - daily OHLCV, Mar 2016 to Dec 2025.")
    pdf.bullet(
        "13 ETFs: SPY (traded asset), QQQ, RSP (breadth/momentum), plus 10 sector ETFs "
        "(XLB, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY) for Absorption Ratio."
    )
    pdf.bullet("Returns use Adj_Close (split/dividend adjusted). Log returns are computed day-over-day.")
    pdf.bullet(
        "Out-of-sample (OOS) evaluation window: 1 Jul 2022 to 31 Dec 2025 (src/config.py). "
        "All versions are clipped to this window so metrics are comparable."
    )
    pdf.bullet(
        "S&P 500/400/600 PIT stock files exist in Data/ but were not used in the final pipeline - "
        "sector ETFs drive the Absorption Ratio instead."
    )

    pdf.section_title("3. Step-by-Step Procedure")
    steps = [
        "Load ETF prices and compute log returns for all 13 tickers.",
        "Compute Absorption Ratio (AR): rolling PCA on 10 sector return correlations "
        "(126-day window, 2 components). Higher AR = more correlated sectors = fragile market.",
        "Compute AR z-score: fast (15-day) vs slow (126-day) rolling mean/std of AR.",
        "Build feature matrix for the Jump Model (JM): SPY downside vol, Sortino ratios, "
        "vol ratio, breadth (RSP vs SPY), and optionally AR columns depending on version.",
        "Walk-forward cross-validation: each month, refit JM on the prior 3 years (~756 days), "
        "pick the best lambda from the candidate grid, predict Bull/Bear for that month.",
        "Backtest: Bull -> hold SPY (full or graded exposure); Bear -> cash (0% SPY). "
        "Signals are lagged 1 day - no look-ahead.",
        "Report metrics: CAGR, volatility, Sharpe, max drawdown, Calmar, % time invested. "
        "Save equity curve plots and CSV outputs.",
    ]
    for i, step in enumerate(steps, 1):
        pdf.bullet(f"Step {i}: {step}")

    pdf.section_title("4. Jump Model (JM)")
    pdf.body_text(
        "The Statistical Jump Model (Shu, Yu & Mulvey, 2024) splits the market into 2 states "
        "(Bull and Bear). Each day it assigns the day to the state whose centroid best fits "
        "that day's feature vector - but switching states costs a penalty lambda."
    )
    pdf.body_text(
        "Objective: minimise sum of squared fit errors + lambda x (number of regime switches).\n"
        "Higher lambda -> stickier regimes, fewer switches, slower to detect a new Bear.\n"
        "Lower lambda  -> more switches, faster reaction, but can flicker and overfit."
    )
    pdf.body_text(
        "Each month, walk-forward CV picks the lambda that best separates Bull vs Bear days "
        "on the prior 3-year training window (using SPY return separation as the score)."
    )

    pdf.section_title("5. Main Model - v5 Hybrid")
    pdf.body_text(
        "v5 combines two ideas:\n"
        "(A) v1-style JM regime - trained on 7 features including AR inside the feature matrix. "
        "This gives a stable, sticky Bull/Bear label (~7 regime switches over OOS).\n"
        "(B) v4-style AR overlay - applied separately at backtest time. Even when JM still says "
        "Bull, high/rising AR z-score can cut exposure or force a fast exit (handles late Bear "
        "detection, e.g. 2025-style lag)."
    )
    pdf.body_text(
        "Position sizing (v5 hybrid):\n"
        "  - JM Bear -> 0% (cash)\n"
        "  - JM Bull + AR calm -> 100% SPY\n"
        "  - JM Bull + AR fragile -> 75% floor (bull_floor), can scale down further on exit rules\n"
        "  - AR exit: z-score > 1.25, or high + rising for 3 days -> drop to cash"
    )
    pdf.body_text(
        "v5 OOS results (Jul 2022 - Dec 2025):\n"
        "  JM baseline alone:  CAGR 10.4%  |  Sharpe 0.96  |  Max DD -12.5%  |  Invested 72.5%\n"
        "  v5 Hybrid:          CAGR  6.7%  |  Sharpe 0.70  |  Max DD -12.5%  |  Invested 66.0%\n"
        "  Buy & Hold:         CAGR 18.5%  |  Sharpe 1.10  |  Max DD -19.2%  |  Invested 99.8%\n"
        "The hybrid trades some return for faster AR-driven exits and lower time in market."
    )
    pdf.add_image_safe(IMG_V5, "Figure 1 - v5: JM Strategy (blue), v5 Hybrid (orange), Buy & Hold (grey)")

    pdf.add_page()
    pdf.section_title("6. Alternative Models")

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "v6 - AR-first Cascade", new_x="LMARGIN", new_y="NEXT")
    pdf.body_text(
        "JM is trained on 5 SPY microstructure features only (no AR in JM). AR acts as a "
        "pre-filter at backtest time - checked before JM:\n"
        "  1. AR veto (z > 1.0, or high + rising for 3 days) -> cash (0%)\n"
        "  2. Else JM Bear -> cash\n"
        "  3. Else JM Bull and no AR veto -> 100% SPY\n"
        "Strict binary 0/1 sizing - no partial positions."
    )
    pdf.body_text(
        "v6 OOS: JM CAGR 8.4%, Sharpe 0.82, Max DD -11.8%  |  "
        "Cascade CAGR 4.9%, Sharpe 0.56, Max DD -10.4%, Invested 58.7%"
    )
    pdf.add_image_safe(IMG_V6, "Figure 2 - v6: JM Strategy vs AR-first Cascade vs Buy & Hold")

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "v2 - 2x2 Grid (JM trend x AR fragility)", new_x="LMARGIN", new_y="NEXT")
    pdf.body_text(
        "JM and AR are kept on separate axes and combined in a 2x2 table:\n"
        "                    AR stable    AR fragile\n"
        "  JM Bull              100%          50%\n"
        "  JM Bear                0%           0%\n"
        "JM uses 5 microstructure features (no AR). AR z-score > 1.0 marks 'fragile'. "
        "Bull + fragile days get half exposure instead of full."
    )
    pdf.body_text(
        "v2 OOS: JM CAGR 8.4%, Sharpe 0.82  |  "
        "2x2 Grid CAGR 6.6%, Sharpe 0.72, Max DD -10.4%, Invested 71.8%"
    )
    pdf.add_image_safe(IMG_V2, "Figure 3 - v2: JM Strategy vs 2x2 Grid vs Buy & Hold")

    pdf.section_title("7. Lambda Tuning")
    pdf.body_text("Lambda is the jump penalty in the JM. It controls how sticky regimes are.")
    pdf.bullet("Early runs used candidates starting at 5 and above only: [5, 10, 15, 20, 25, 30, 40, 50].")
    pdf.bullet("Later, lambda = 3 was added to the grid: [3, 5, 10, 15, 20, 25, 30, 40, 50].")
    pdf.bullet(
        "Walk-forward CV picks one lambda per month. In practice, lambda = 3 was chosen "
        "most often once it was allowed - especially when the model needed faster regime switches."
    )
    pdf.bullet(
        "Meaning: pushing lambda as low as the grid allows makes the model switch regimes "
        "more easily. That fixes late Bear detection but can cause flickering if taken too far "
        "(values 1-2 were tested in lambda experiments and rejected as too noisy)."
    )

    pdf.section_title("8. Issues Faced")
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Issue 1 - Sticky regimes / late Bear detection", new_x="LMARGIN", new_y="NEXT")
    pdf.body_text(
        "When AR was included inside JM features (v1), regimes became very sticky - only ~7 switches "
        "over the full OOS window. The model stayed Bull too long during drawdowns (visible in the "
        "2022 crash: market fell before the red Bear shading started). AR inside features reinforced "
        "stickiness because AR itself moves slowly."
    )
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Issue 2 - Lambda gravitating to the minimum", new_x="LMARGIN", new_y="NEXT")
    pdf.body_text(
        "Once lambda = 3 was allowed, monthly CV often picked the lowest available value. "
        "That is expected: lower lambda = cheaper to switch = more reactive Bear calls. "
        "The trade-off is return vs responsiveness - very low lambda (1-2) over-switches and "
        "hurts out-of-sample stability. Lambda = 3 became the practical floor."
    )

    pdf.section_title("9. Summary")
    pdf.body_text(
        "v5 is the main model: sticky JM regime (7 features) plus a separate AR overlay for "
        "fast exits and graded re-entry. It reduces max drawdown vs Buy & Hold (-12.5% vs -19.2%) "
        "but gives up absolute return. v6 (AR pre-filter cascade) and v2 (2x2 grid) are cleaner "
        "alternative architectures explored along the way. Earlier attempts (v1, v3, v4, lambda "
        "experiments) are archived in Scrapped_Tries/."
    )

    pdf.output(str(OUT_PDF))
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    for path in (IMG_V5, IMG_V6, IMG_V2):
        if not path.exists():
            print(f"WARNING: missing {path} - run main5.py / main6.py / main2.py first")
    build_pdf()
