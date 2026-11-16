r"""Generate the final model card and a 10-page fraud business strategy PDF.

Place in your existing project's src/ folder, beside final_test_evaluation.py.
Run from project root:
    .\.venv\Scripts\python.exe src\governance_reports.py

New dependency (install only once if absent): reportlab
    .\.venv\Scripts\python.exe -m pip install reportlab

Read-only regarding the dataset, trained pipelines, split, and thresholds.
All quantitative claims come from your previously saved output files, not
hard-coded example results. This is reporting, NEVER model re-evaluation.
"""
from __future__ import annotations

import html
import json
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / 'reports'
FINAL = REPORTS / 'final_evaluation'
OUT_CARD = ROOT / '06_model_card.pdf'
OUT_STRATEGY = ROOT / '07_business_fraud_strategy.pdf'

NAVY = colors.HexColor('#14263D')
TEAL = colors.HexColor('#147E82')
INK = colors.HexColor('#253447')
SLATE = colors.HexColor('#516174')
LIGHT = colors.HexColor('#EDF3F5')
BORDER = colors.HexColor('#DCE5E9')
RED = colors.HexColor('#9B4338')
WHITE = colors.white
W, H = A4

STYLES = {
    'cover_kicker': ParagraphStyle('CoverKicker', fontName='Helvetica-Bold', fontSize=10, leading=14, textColor=TEAL, spaceAfter=10),
    'cover': ParagraphStyle('Cover', fontName='Helvetica-Bold', fontSize=23, leading=28, textColor=NAVY, spaceAfter=16),
    'h1': ParagraphStyle('H1', fontName='Helvetica-Bold', fontSize=16.2, leading=20, textColor=NAVY, spaceBefore=4, spaceAfter=10),
    'h2': ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=10.7, leading=14, textColor=TEAL, spaceBefore=10, spaceAfter=6),
    'body': ParagraphStyle('Body', fontName='Helvetica', fontSize=9.1, leading=13.5, textColor=INK, spaceAfter=7),
    'small': ParagraphStyle('Small', fontName='Helvetica', fontSize=8.1, leading=11.5, textColor=INK, spaceAfter=5),
    'muted': ParagraphStyle('Muted', fontName='Helvetica', fontSize=8.1, leading=11.5, textColor=SLATE, spaceAfter=6),
    'warning': ParagraphStyle('Warning', fontName='Helvetica-Bold', fontSize=9.1, leading=13.5, textColor=RED, spaceAfter=9),
    'cell': ParagraphStyle('Cell', fontName='Helvetica', fontSize=8, leading=10.5, textColor=INK),
    'cell_header': ParagraphStyle('CellHeader', fontName='Helvetica-Bold', fontSize=8, leading=11, textColor=WHITE),
}


def safe(s):
    return html.escape(str(s), quote=False)


def P(s, kind='body'):
    return Paragraph(s, STYLES[kind])


def paragraph(s, kind='body'):
    return P(safe(s), kind)


def bullet(s):
    return P('<font color="#147E82"><b>&bull;</b></font>  ' + safe(s), 'body')


def title(s):
    return P(safe(s), 'h1')


def sub(s):
    return P(safe(s), 'h2')


def fmt(v, nd=3):
    return f'{float(v):,.{nd}f}'


def pct(v, nd=1):
    return f'{float(v)*100:.{nd}f}%'


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(f'Missing previous project output: {path.relative_to(ROOT)}')
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv_if_exists(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def tiny_table(headers, rows, widths=None, max_rows=None):
    if max_rows is not None:
        rows = list(rows)[:max_rows]
    matrix = [[P(safe(x), 'cell_header') for x in headers]]
    matrix += [[P(safe(x), 'cell') for x in row] for row in rows]
    opts = {'colWidths': widths} if widths else {}
    table = Table(matrix, hAlign='LEFT', repeatRows=1, **opts)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, LIGHT]),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LINEBELOW', (0,-1), (-1,-1), .5, BORDER),
    ]))
    return table


def chart(path, width=164*mm, max_height=80*mm):
    if not path.exists():
        return paragraph(f'Chart not available: {path.relative_to(ROOT)}.', 'muted')
    img = Image(str(path))
    scale = min(width / img.imageWidth, max_height / img.imageHeight)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    img.hAlign = 'CENTER'
    return img


def footer(c, doc):
    c.saveState()
    c.setStrokeColor(BORDER)
    c.line(19*mm, 18*mm, W-19*mm, 18*mm)
    c.setFont('Helvetica', 8)
    c.setFillColor(SLATE)
    c.drawString(19*mm, 13.6*mm, 'FRAUD RISK INTELLIGENCE  |  EXPERIMENTAL / NOT FOR LIVE DECISIONS')
    c.drawRightString(W-19*mm, 13.6*mm, str(doc.page))
    c.restoreState()


def create_doc(out_path):
    return SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=19*mm, rightMargin=19*mm,
                             topMargin=19*mm, bottomMargin=23*mm, title=out_path.stem.replace('_',' '),
                             author='Fraud Risk Internship Project', pageCompression=1)


def section(story, heading, *items):
    if story:
        story.append(PageBreak())
    story.append(title(heading))
    story.extend(items)


def prep_data():
    m = read_json(FINAL/'10_final_test_metrics.json')
    policy = read_json(ROOT/'models'/'fraud_decision_policy.json')
    train = read_json(ROOT/'models'/'model_training_metadata.json')
    cal = read_json(ROOT/'models'/'calibration_metadata.json')
    compare = read_csv_if_exists(FINAL/'03_policy_comparison.csv')
    sensitivity = read_csv_if_exists(FINAL/'04_frozen_policy_sensitivity.csv')
    perf = read_csv_if_exists(REPORTS/'model_comparison.csv')
    stress = read_csv_if_exists(REPORTS/'robustness'/'04_scenario_comparison.csv')
    cv = read_csv_if_exists(REPORTS/'robustness'/'01_cv_fold_results.csv')
    importance = read_csv_if_exists(REPORTS/'explainability'/'01_global_feature_importance.csv')
    feature_catalog = read_csv_if_exists(REPORTS/'feature_catalog.csv')
    calibration_cmp = read_csv_if_exists(REPORTS/'calibration_comparison.csv')
    imbalance_cmp = read_csv_if_exists(REPORTS/'imbalance_comparison.csv')
    for k in ['ranking_and_calibration','frozen_3_way_policy','binary_at_0_50','selected_thresholds_from_policy_only']:
        if k not in m:
            raise ValueError(f'Final test metrics JSON missing required key: {k}')
    if m.get('frozen_model_and_policy_unchanged') is not True:
        raise ValueError('The final test file does not confirm a frozen model/policy.')
    if not compare.empty and abs(float(compare.iloc[0]['total_cost_cu']) - float(m['frozen_3_way_policy']['total_cost_cu'])) > 1e-5:
        raise ValueError('Saved policy cost CSV and final metrics JSON disagree.')
    return locals()


def model_card(d):
    m, pol, tm, cal = d['m'], d['policy'], d['train'], d['cal']
    rank = m['ranking_and_calibration']; op = m['frozen_3_way_policy']; t = m['selected_thresholds_from_policy_only']
    s=[]
    section(s, 'MODEL CARD  /  Fraud Risk Decision System',
            P('MODEL GOVERNANCE  |  INTERNAL REVIEW', 'cover_kicker'),
            paragraph('Generated ' + date.today().isoformat() + ' from saved model and final test artifacts.', 'muted'),
            sub('Intended use'),
            paragraph('Experimental transaction-risk scoring for an online/card transaction service. The output is a predicted fraud probability plus a three-way operational recommendation: Approve, Manual Review, or Block. It is for offline assessment and simulated workflow design only.'),
            sub('Model and data'),
            tiny_table(['Attribute', 'Value'], [
                ['Model family', str(m.get('model_family','Not recorded'))],
                ['Feature mode', str(m.get('feature_mode','Not recorded'))],
                ['Calibrator', str(m.get('calibration_method','Not recorded'))],
                ['Input fields', str(len(tm.get('features',[])))],
                ['Held-out test records', str(m['test_rows'])],
                ['Test fraud prevalence', pct(rank['fraud_prevalence'],2)],
                ['Validation approach', 'Stratified random split; separate training, calibration, policy, and test sets'],
            ], widths=[48*mm, 124*mm]),
            sub('Exclusions and restrictions'),
            bullet('No evidence of chronological or customer-level generalization: timestamps and customer IDs are unavailable.'),
            bullet('No real financial cost, manual-review efficacy, or production fairness validation was provided.'),
            bullet('No automated live blocking or unreviewed use with customer funds is authorized by this experiment.'),
            bullet('The final held-out test was evaluated once; changing the pipeline or thresholds now requires a fresh future holdout.'),
    )
    section(s, 'EVALUATION  /  Model and policy evidence',
            sub('Held-out predictive metrics'),
            tiny_table(['Metric', 'Observed'], [
                ['ROC-AUC', fmt(rank['roc_auc'],4)],
                ['PR-AUC / Average Precision', fmt(rank['pr_auc_average_precision'],4)],
                ['Brier score (lower is better)', fmt(rank['brier_score'],5)],
                ['Precision / Recall / F1 at 0.50', ' / '.join(fmt(m['binary_at_0_50'][x],3) for x in ['precision','recall','f1'])],
            ], widths=[88*mm, 84*mm]),
            sub('Frozen operating policy'),
            paragraph(f"Approve for p &lt; {t['review']:.4f}; review for {t['review']:.4f} &lt;= p &lt; {t['block']:.4f}; block for p &gt;= {t['block']:.4f}."),
            tiny_table(['Measure', 'Result'], [
                ['Approve / Review / Block', f"{op['approved']} / {op['reviewed']} / {op['blocked']}"],
                ['Fraud flagged (review or block)', f"{op['fraud_flagged']} of {m['fraud_rows']} ({pct(op['flagged_fraud_recall'])})"],
                ['Legitimate reviewed / blocked', f"{op['legitimate_reviewed']} / {op['legitimate_blocked']}"],
                ['Review share', pct(op['review_share'])],
                ['Simulated policy cost', fmt(op['total_cost_cu'],2) + ' CU'],
                ['Approve-all simulated cost', fmt(m['approve_all_cost_cu'],2) + ' CU'],
            ], widths=[88*mm,84*mm]),
            paragraph('Cost is expressed in hypothetical currency units (CU). Lower modeled costs on a retrospective sample do not guarantee operational savings.', 'muted'),
            sub('Calibration and model-selection caution'),
            paragraph('The model was sigmoid-calibrated on reserved calibration data. In the earlier method-selection subset, the raw probabilities scored better on Brier loss than either tested calibration method. The selected calibrator should therefore not be described as a demonstrated calibration improvement.'),
    )
    section(s, 'GOVERNANCE  /  Monitoring, controls and sign-off',
            sub('Minimum production monitoring dashboard'),
            *[bullet(x) for x in [
                'Transaction volume, missingness and input schema failures by channel.',
                'Fraud-label prevalence and label-arrival delay (matured labels only).',
                'PR-AUC, ROC-AUC, precision, recall and calibrated Brier score as labels mature.',
                'Approve / review / block shares, review queue, and review capacity utilization.',
                'Legitimate block rate, disputes and customer friction incidents.',
                'Feature drift, score drift and within-segment fraud-rate changes.',
                'Manual-review turnaround time, appeal reversals and investigation outcomes.',
            ]],
            sub('Proposed escalation triggers - require owner approval'),
            paragraph('Investigate substantial score-distribution or feature drift, any schema break or rising missingness, sustained review share above the assumed 25% capacity, legitimate block rate above the assumed 2% guardrail, or material deterioration in fraud capture after label maturity. These are illustrative controls, not verified company service-level agreements.'),
            sub('Retraining / temporary disable policy'),
            paragraph('Retrain only with newly collected, labeled data under a freshly defined temporal/group-aware holdout and a documented approval process. Disable automatic Block and route high-risk cases to manual review if critical features fail, output probabilities become invalid, or adverse-customer-outcome monitoring triggers a material incident.'),
            sub('Governance checklist'),
            *[bullet(x) for x in [
                'Confirm every field is available before the transaction decision.',
                'Obtain real cost and review-capacity inputs; repeat cost sensitivity analysis.',
                'Review fairness, privacy and access controls with the business owner.',
                'Document audit trails, versioned thresholds and rollback procedures.',
                'Obtain risk/compliance sign-off before any production trial.',
            ]],
            paragraph('Approval status: NOT APPROVED FOR PRODUCTION. Owner and reviewer sign-off: pending.', 'warning'),
    )
    create_doc(OUT_CARD).build(s, onFirstPage=footer, onLaterPages=footer)


def strategy(d):
    m = d['m']; r=m['ranking_and_calibration']; op=m['frozen_3_way_policy']; t=m['selected_thresholds_from_policy_only']
    s=[]
    section(s,'Business Fraud Strategy  /  Executive summary',
        P('FRAUD RISK INTELLIGENCE', 'cover_kicker'),
        P('Decision intelligence for transaction fraud', 'cover'),
        paragraph('Evidence-based internship case study | ' + date.today().isoformat(), 'muted'),
        sub('Recommendation'),
        paragraph('Use a governed, human-in-the-loop pilot design as a research prototype - not an automatic production blocking tool. The measured ranking performance is moderate, and the operating-cost numbers rely on hypothetical assumptions.'),
        sub('Test-set snapshot'),
        tiny_table(['Indicator','Measured result'],[
            ['Held-out test',f"{m['test_rows']} transactions, {m['fraud_rows']} fraud cases"],
            ['ROC-AUC / PR-AUC',f"{fmt(r['roc_auc'],4)} / {fmt(r['pr_auc_average_precision'],4)}"],
            ['Approve / Review / Block', f"{op['approved']} / {op['reviewed']} / {op['blocked']}"],
            ['Fraud flagged', f"{op['fraud_flagged']} of {m['fraud_rows']} ({pct(op['flagged_fraud_recall'])})"],
            ['Modeled cost vs approve all',f"{fmt(op['total_cost_cu'],2)} CU vs {fmt(m['approve_all_cost_cu'],2)} CU"],
            ['Relative modeled cost reduction',pct(m['base_policy_cost_saving_fraction'])],
        ], widths=[75*mm,97*mm]),
        sub('Decision caveat'),
        paragraph(f"On the held-out sample, {m['fraud_rows']-op['fraud_flagged']} of {m['fraud_rows']} fraud cases were not flagged. Reported cost reduction is a scenario result, not measured bank savings.", 'warning'),
        paragraph('Prepared from persisted outputs of the user-run pipeline: source dataset, feature catalog, validation results, calibrated model metadata, frozen policy, final test metrics and stress-testing reports.', 'muted'),
    )
    section(s,'02  /  Business problem and objectives',
        sub('The fraud operations trade-off'),
        paragraph('High-risk transactions may lead to direct losses when approved; incorrectly blocked legitimate transactions create customer friction, support demand and reputational risk. Manual review can reduce risk but consumes operational capacity. A binary classifier alone cannot express this multi-option business trade-off.'),
        sub('Pipeline output'),
        *[bullet(x) for x in [
            'A calibrated fraud-probability estimate for each transaction.',
            'A three-way decision: Approve, Manual Review or Block.',
            'Explanations at global and individual-transaction levels.',
            'Cost and capacity diagnostics for risk and operations teams.',
            'A model card, stress tests, and post-deployment control proposals.',
        ]],
        sub('Success criteria and failure modes'),
        paragraph('Ranking quality is measured using PR-AUC/average precision and ROC-AUC. Review capacity, fraud capture, legitimate blocking, calibrated probability quality and simulated total business cost jointly determine operational usefulness. Strong ROC-AUC alone does not guarantee the lowest business cost.'),
        sub('Scope of this assignment'),
        paragraph('The experiment is offline and retrospective, using a provided synthetic or case-study transaction dataset. It cannot verify real-time eligibility of every feature, actual economics, true adversarial adaptation, or production monitoring. A future pilot would need additional data and formal approval.'),
    )
    section(s,'03  /  Data forensics and engineered features',
        sub('Data quality and missingness'),
        paragraph('The provided dataset has 3,000 rows and 25 original columns, including a binary fraud label. Nulls appear in numeric amount, distance, risk score and elapsed-time fields. The literal category "None" in authentication_level means no authentication and must not be interpreted as a missing value.'),
        paragraph('Initial auditing covers schema, duplicates, missingness, cardinality, impossible values, and extremes. Missing-value imputation and learned preprocessing occur inside training pipelines, preventing leakage from validation or test partitions.'),
        sub('Feature engineering discipline'),
        paragraph('The project created 28 additional descriptive and interaction features. Derived fields include amount normalization, transaction velocity ratios, night-time flags, device-distance combinations and risk interactions. Any derived feature dependent on information unavailable at transaction authorization should be excluded from a future production scorer.'),
        tiny_table(['Engineered example','Analytical purpose'],[
            ['log(1+amount)', 'Reduce skew while preserving amount ranking'],
            ['amount / (1 + recent count)', 'Behavior-normalized spending intensity'],
            ['new device x distance', 'Check joined behavior risk'],
            ['risk score interactions', 'Represent combined risk signals'],
            ['time-window flags', 'Represent transaction timing context'],
        ], widths=[72*mm,100*mm]),
        sub('Leakage limitations'),
        paragraph('The source dataset has no full timestamps or customer identifiers; a robust chronological split and customer-independent validation cannot be established. Pretransaction availability must be assessed with the data owner before any live deployment.'),
    )
    perf=d['perf']
    rows=[]
    if not perf.empty:
        cols=perf.columns
        fam=next((c for c in ['model','model_family','algorithm'] if c in cols),None)
        mode=next((c for c in ['feature_set','feature_mode','features'] if c in cols),None)
        score=next((c for c in ['pr_auc_mean','cv_pr_auc','pr_auc','mean_pr_auc'] if c in cols),None)
        if fam and score:
            rows=[[str(row[fam])+(f" ({row[mode]})" if mode else ''),fmt(row[score],3)] for _,row in perf.head(6).iterrows()]
    section(s,'04  /  Validation methodology and model selection',
        sub('Leakage-aware validation architecture'),
        paragraph('A fixed seed and saved split manifest separate training, calibration, policy optimization and final test data. Cross-validation and hyperparameter search remain within training only. Calibration is fitted using its reserved partition; decision thresholds are tuned on a separate policy partition.'),
        paragraph(f"Final held-out test: {m['test_rows']} rows ({m['fraud_rows']} fraudulent). The calibrated pipeline and two policy thresholds were frozen before the one-time test evaluation. No test observations were reused for tuning."),
        sub('Model comparison'),
        paragraph('The candidate families include Logistic Regression, Random Forest and Histogram Gradient Boosting, with raw and engineered features. The training-stage ranking chose Logistic Regression using raw features; a randomized parameter search improved its training-only PR-AUC.'),
        *( [tiny_table(['Training candidate','CV PR-AUC'],rows, widths=[120*mm,52*mm])] if rows else [paragraph('The saved comparison table is available in reports/model_comparison.csv; consult it for the full training comparison.', 'muted')]),
        sub('Interpretation of holdout metrics'),
        paragraph(f"The final ROC-AUC of {fmt(r['roc_auc'],4)} reflects ranking discrimination, while PR-AUC {fmt(r['pr_auc_average_precision'],4)} is more directly informative for the minority fraud class. These values should be contextualized against fraud prevalence {pct(r['fraud_prevalence'],2)}."),
        paragraph('A retrospective random split tends to be more optimistic than a validated forward-in-time deployment evaluation. This weakness must be resolved in any operational study.'),
    )
    section(s,'05  /  Probability calibration and class imbalance',
        sub('Why probability quality matters'),
        paragraph('A score of 0.30 should have an interpretable relationship to eventual fraud frequency across similarly scored transactions. Poor calibration can lead to wrong cost-based choices even when ranking metrics are reasonable.'),
        tiny_table(['Final test metric','Value'],[
            ['ROC-AUC',fmt(r['roc_auc'],4)],
            ['Average Precision / PR-AUC',fmt(r['pr_auc_average_precision'],4)],
            ['Brier score',fmt(r['brier_score'],5)],
            ['Precision at p >= 0.50',fmt(m['binary_at_0_50']['precision'],3)],
            ['Recall at p >= 0.50',fmt(m['binary_at_0_50']['recall'],3)],
        ], widths=[92*mm,80*mm]),
        chart(FINAL/'08_calibration_curve.png',max_height=80*mm),
        paragraph('Imbalance strategies evaluated included unweighted fitting, balanced class weights, and training-fold-only oversampling. Sigmoid and isotonic calibration were compared. Neither calibration method improved Brier loss versus the raw model on the small calibration-selection subset; this is a documented limitation.', 'small'),
    )
    section(s,'06  /  Frozen three-way fraud decision policy',
        sub('Thresholds selected on policy validation only'),
        paragraph(f"Approve below {t['review']:.4f}; Manual Review from {t['review']:.4f} to below {t['block']:.4f}; Block at {t['block']:.4f} and above."),
        chart(FINAL/'09_decision_counts.png',max_height=81*mm),
        tiny_table(['Held-out outcome','Count'],[
            ['Approved',op['approved']],['Manual review',op['reviewed']],['Blocked',op['blocked']],
            ['Fraud reviewed',op['fraud_reviewed']],['Fraud blocked',op['fraud_blocked']],
            ['Legitimate reviewed',op['legitimate_reviewed']],['Legitimate blocked',op['legitimate_blocked']],
        ], widths=[112*mm,60*mm]),
        paragraph(f"The frozen policy flagged {op['fraud_flagged']} / {m['fraud_rows']} fraud transactions ({pct(op['flagged_fraud_recall'])}). It also incurred {op['legitimate_reviewed']} legitimate manual reviews and {op['legitimate_blocked']} legitimate blocks.", 'small'),
    )
    compare=d['compare']
    policy_rows=[]
    if not compare.empty:
        policy_rows=[[x['policy'],fmt(x['total_cost_cu'],2),int(x['reviewed']),int(x['blocked'])] for _,x in compare.iterrows()]
    section(s,'07  /  Hypothetical financial impact and sensitivity',
        sub('Cost model assumptions'),
        paragraph('Cost scenarios account for fraud losses, manual-review cost, legitimate-customer friction, false blocking and review success. All values are hypothetical currency units (CU) and an assumed review capture rate. The analysis does not establish actual financial returns.'),
        tiny_table(['Policy', 'Total cost (CU)', 'Review', 'Block'],policy_rows or [['Frozen policy',fmt(op['total_cost_cu'],2),op['reviewed'],op['blocked']]],widths=[79*mm,47*mm,23*mm,23*mm]),
        sub('Financial interpretation'),
        paragraph(f"Compared with approve-all cost {fmt(m['approve_all_cost_cu'],2)} CU, the frozen three-way policy costs {fmt(op['total_cost_cu'],2)} CU, giving a modeled difference of {fmt(m['base_policy_cost_difference_vs_approve_all_cu'],2)} CU ({pct(m['base_policy_cost_saving_fraction'])})."),
        sub('Sensitivity on unchanged decisions'),
        *([tiny_table(['Cost scenario','Cost (CU)'],[[r['scenario'],fmt(r['total_cost_cu'],2)] for _,r in d['sensitivity'].head(6).iterrows()],widths=[110*mm,62*mm])] if not d['sensitivity'].empty else [paragraph('No sensitivity CSV available; see saved policy cost summary.', 'muted')]),
        paragraph('Sensitivity tables recompute outcomes under alternative cost assumptions without retuning the held-out test thresholds. The same decisions need not remain optimal under different economics. Therefore, business approval requires validation of assumed cost components.', 'small'),
    )
    importance=d['importance']
    top=[]
    if not importance.empty and {'feature','permutation_pr_auc_decrease_mean'}.issubset(importance.columns):
        imp=importance.sort_values('permutation_pr_auc_decrease_mean',ascending=False).head(5)
        top=[[x['feature'],fmt(x['permutation_pr_auc_decrease_mean'],4)] for _,x in imp.iterrows()]
    section(s,'08  /  Explainable AI and individual cases',
        sub('Global feature influence'),
        paragraph('For the selected Logistic Regression model, exact additive contributions in log-odds explain how transformed features shift the base-model score. Permutation importance estimates how much PR-AUC decreases when a feature is disrupted. The calibrated probability uses an additional mapping and need not equal a direct sum of local contributions.'),
        *( [tiny_table(['Top feature','Mean permutation PR-AUC decrease'],top,widths=[90*mm,82*mm])] if top else [paragraph('See reports/explainability/ for recorded feature ranking.', 'muted')]),
        chart(REPORTS/'explainability'/'02_global_feature_importance.png',max_height=73*mm),
        sub('Local explanations and non-causality'),
        paragraph('Five high-risk policy-set transactions received local explanations. Feature dependence views cover device-distance and amount-merchant-risk combinations. Explanation correlations are not proof that changing a feature will change fraud in the real world.'),
        paragraph('Local explanations can be unstable for correlated inputs or small sample segments; monitor reproducibility of feature rankings and investigate counterintuitive attributions.', 'small'),
    )
    stress=d['stress']
    scen=[]
    if not stress.empty and {'scenario','decision_change_count','review_count','block_count'}.issubset(stress.columns):
        scen=[[str(x['scenario']).replace('_',' '),int(x['decision_change_count']),int(x['review_count']),int(x['block_count'])] for _,x in stress.head(8).iterrows()]
    section(s,'09  /  Robustness, drift and failure modes',
        sub('Stress testing on policy validation'),
        paragraph('Saved stress tests simulate increased amounts, extreme transactions, missing merchant-risk inputs, higher velocity and channel-mix changes. The tests quantify changes in scores and decision mix; they do not prove accuracy under true real-world drift.'),
        *([tiny_table(['Scenario','Decision changes','Review','Block'],scen,widths=[91*mm,31*mm,25*mm,25*mm])] if scen else [paragraph('Stress scenario details are recorded in reports/robustness/04_scenario_comparison.csv.', 'muted')]),
        sub('Operational interpretation'),
        paragraph('A surge in transaction velocity may greatly increase review and block volume; it can breach an assumed staffing limit even while the model continues to run. Drift is therefore a queue-management and customer-impact concern as well as a predictive concern.'),
        sub('Cross-validation variation'),
        paragraph('The model was assessed on multiple training folds. Compare fold-by-fold PR-AUC, ROC-AUC and feature stability in the robustness outputs; volatile attribution or large fold-to-fold swings should trigger further diagnostic investigation.'),
        chart(REPORTS/'robustness'/'05_stress_decision_chart.png',max_height=69*mm),
    )
    section(s,'10  /  Model governance and decision roadmap',
        sub('Operational monitoring: minimum scorecard'),
        *[bullet(x) for x in [
            'Input data quality: schema violations, missingness and numeric extremes by channel.',
            'Feature and score drift: important risk variables, fraud probability distribution and channel mix.',
            'Label-based outcomes: fraud prevalence, PR-AUC, ROC-AUC, calibration and confidence intervals.',
            'Decision impact: review share, queue length, review turnaround, block share and appeal rates.',
            'Customer harm: legitimate blocks, false-positive reviews, churn and complaint indicators.',
            'Economic outcomes: observed review expense, fraud losses avoided and realized net cost.',
        ]],
        sub('Triggers and response'),
        paragraph('Investigate sustained review share over the assumed 25% review capacity, legitimate block rate over the assumed 2% constraint, critical feature outages, invalid scores, or material feature/score distribution changes. Escalate to manual-only review if safety guardrails fail; restore automatic blocking only after human approval and targeted validation.'),
        sub('Recommended sequence for a real pilot'),
        *[bullet(x) for x in [
            'Acquire chronological event data and reliable customer identifiers; validate prospectively.',
            'Validate feature latency, data provenance and privacy/legal eligibility.',
            'Replace hypothetical cost and review-efficacy values with measured business parameters.',
            'Run a monitored shadow-mode pilot with humans reviewing high-risk decisions.',
            'Obtain risk-owner sign-off and formal model governance review before live use.',
        ]],
        sub('Evidence and source register'),
        paragraph('Primary sources: transaction_fraud_risk_advanced_dataset.csv; reports/model_comparison.csv; reports/calibration_comparison.csv; reports/explainability/*; reports/robustness/*; reports/final_evaluation/10_final_test_metrics.json; reports/final_evaluation/03_policy_comparison.csv; models/fraud_decision_policy.json; and the DS 02 assignment brief.', 'small'),
        paragraph('Final recommendation: continue as a documented offline model and controlled research prototype. NOT APPROVED FOR PRODUCTION.', 'warning'),
    )
    create_doc(OUT_STRATEGY).build(s,onFirstPage=footer,onLaterPages=footer)


def main():
    data=prep_data()
    model_card(data)
    strategy(data)
    print('GOVERNANCE + BUSINESS REPORTS COMPLETE')
    print(f' - {OUT_CARD.relative_to(ROOT)}')
    print(f' - {OUT_STRATEGY.relative_to(ROOT)}')
    print('All metrics read from saved project reports. No retraining or TEST scoring.')
    print('Costs remain hypothetical; model is NOT production approved.')

if __name__ == '__main__':
    main()
