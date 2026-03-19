"""
DiabetaGuard — Clinical Inference Engine
Python/Flask Backend · Rule-Based Expert System
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import math
import os

app = Flask(__name__, static_folder="static")
CORS(app)

# ─────────────────────────────────────────────
#  RULE-BASED INFERENCE ENGINE
# ─────────────────────────────────────────────

class InferenceRule:
    """A single clinical inference rule."""
    def __init__(self, condition_fn, score, finding, category):
        self.condition_fn = condition_fn
        self.score = score
        self.finding = finding
        self.category = category  # 'critical' | 'major' | 'minor'

    def evaluate(self, d):
        try:
            triggered = self.condition_fn(d)
        except Exception:
            triggered = False
        if triggered:
            return self.score, self.finding
        return 0, None


class ComplicationDomain:
    """A diabetic complication domain with its own ruleset."""
    def __init__(self, key, name, icon, description, rules):
        self.key = key
        self.name = name
        self.icon = icon
        self.description = description
        self.rules = rules

    def evaluate(self, d):
        total_score = 0
        findings = []
        triggered_rules = []
        for rule in self.rules:
            pts, finding = rule.evaluate(d)
            if finding:
                total_score += pts
                findings.append({"text": finding, "score": pts, "category": rule.category})
                triggered_rules.append(rule.finding[:40])
        score = min(round(total_score), 100)
        return {
            "key": self.key,
            "name": self.name,
            "icon": self.icon,
            "description": self.description,
            "score": score,
            "findings": findings,
            "rules_triggered": len(findings)
        }


# ─── DOMAIN 1: NEPHROPATHY ────────────────────
nephropathy_rules = [
    InferenceRule(lambda d: d["uacr"] >= 300,   35, "Macroalbuminuria (UACR ≥300 mg/g) — advanced nephropathy marker; immediate nephrology referral required", "critical"),
    InferenceRule(lambda d: 30 <= d["uacr"] < 300, 20, "Microalbuminuria (UACR 30–300 mg/g) — earliest and most reversible nephropathy sign", "major"),
    InferenceRule(lambda d: d["egfr"] < 30,     40, "Severely reduced eGFR (<30 mL/min) — CKD Stage 4; urgent nephrology referral", "critical"),
    InferenceRule(lambda d: 30 <= d["egfr"] < 60, 25, "Reduced eGFR (30–59) — CKD Stage 3; nephropathy established", "major"),
    InferenceRule(lambda d: 60 <= d["egfr"] < 90, 8,  "Mildly reduced eGFR (60–89) — early renal decline; monitor closely", "minor"),
    InferenceRule(lambda d: d["sex"] == "female" and d["creatinine"] > 1.0, 15, "Elevated serum creatinine (>1.0 F) — impaired glomerular filtration", "major"),
    InferenceRule(lambda d: d["sex"] == "male"   and d["creatinine"] > 1.2, 15, "Elevated serum creatinine (>1.2 M) — impaired glomerular filtration", "major"),
    InferenceRule(lambda d: d["bun"] > 20,       8,  "Elevated BUN (>20 mg/dL) — reduced urinary nitrogen clearance", "minor"),
    InferenceRule(lambda d: d["sbp"] >= 140,    10,  "Hypertension accelerates nephropathy — intraglomerular pressure elevation", "major"),
    InferenceRule(lambda d: d["diabetes_duration"] > 10, 8, "Diabetes duration >10 years — cumulative glomerular basement membrane damage", "minor"),
    InferenceRule(lambda d: d["hba1c"] >= 9,    10,  "Poor glycaemic control (HbA1c ≥9%) — glycation damages podocytes and mesangium", "major"),
    InferenceRule(lambda d: d["hba1c"] >= 7 and d["hba1c"] < 9, 5, "HbA1c above target — sustained hyperglycaemia strains renal filtration", "minor"),
]

# ─── DOMAIN 2: RETINOPATHY ───────────────────
retinopathy_rules = [
    InferenceRule(lambda d: d["vision"] == "severe",   40, "Severe vision loss — possible proliferative retinopathy or diabetic macular oedema; urgent ophthalmology", "critical"),
    InferenceRule(lambda d: d["vision"] == "moderate", 25, "Moderate visual changes (floaters/blurring) — possible pre-proliferative retinopathy", "major"),
    InferenceRule(lambda d: d["vision"] == "mild",     12, "Mild visual blurring — possible early non-proliferative retinopathy or cataract", "minor"),
    InferenceRule(lambda d: d["hba1c"] >= 9,           20, "HbA1c ≥9% — sustained hyperglycaemia damages retinal capillary pericytes", "major"),
    InferenceRule(lambda d: 7 <= d["hba1c"] < 9,      10, "HbA1c above 7% — increased risk of retinal microaneurysms and hard exudates", "minor"),
    InferenceRule(lambda d: d["diabetes_duration"] > 15, 20, "Diabetes >15 years — nearly all Type 1 and 60% of Type 2 develop retinopathy", "major"),
    InferenceRule(lambda d: 5 < d["diabetes_duration"] <= 15, 10, "Duration 5–15 years — annual dilated fundus exam strongly recommended", "minor"),
    InferenceRule(lambda d: d["sbp"] >= 160,           15, "Severe hypertension (SBP ≥160) — dramatically worsens retinal haemorrhage risk", "major"),
    InferenceRule(lambda d: 140 <= d["sbp"] < 160,    8,  "Hypertension — major accelerant of retinal vessel damage", "minor"),
    InferenceRule(lambda d: d["diabetes_type"] == "type1", 10, "Type 1 diabetes carries 90% retinopathy risk at 25 years duration", "minor"),
    InferenceRule(lambda d: d["ldl"] >= 160,           8,  "High LDL associated with hard exudates and retinal lipid deposition", "minor"),
]

# ─── DOMAIN 3: NEUROPATHY ────────────────────
neuropathy_rules = [
    InferenceRule(lambda d: d["tingling"] == "severe",   40, "Severe tingling/numbness — significant distal symmetric polyneuropathy", "critical"),
    InferenceRule(lambda d: d["tingling"] == "moderate", 25, "Moderate paresthesias — moderate peripheral nerve axonal damage", "major"),
    InferenceRule(lambda d: d["tingling"] == "mild",     12, "Mild tingling — early small-fibre sensory neuropathy possible", "minor"),
    InferenceRule(lambda d: d["foot_ulcers"] == "yes",   35, "Foot ulcers present — severe neuropathy with loss of protective sensation", "critical"),
    InferenceRule(lambda d: d["hba1c"] >= 9,             15, "Poor glycaemic control — sorbitol accumulation damages myelin sheaths", "major"),
    InferenceRule(lambda d: d["diabetes_duration"] > 10, 15, "Duration >10 years — 50% of diabetic patients develop clinical neuropathy", "major"),
    InferenceRule(lambda d: d["smoking"] == "current",   12, "Active smoking impairs peripheral microcirculation, accelerating nerve ischaemia", "major"),
    InferenceRule(lambda d: d["fbg"] >= 200,             10, "Severely elevated fasting glucose — acute osmotic nerve injury", "major"),
    InferenceRule(lambda d: d["hr"] > 100 and d["diabetes_duration"] > 5, 8, "Resting tachycardia may indicate autonomic neuropathy (cardiac vagal denervation)", "minor"),
    InferenceRule(lambda d: d["bmi"] >= 30,              5,  "Obesity — additional metabolic stress on peripheral nerve function", "minor"),
]

# ─── DOMAIN 4: CARDIOVASCULAR ────────────────
cardiovascular_rules = [
    InferenceRule(lambda d: d["sbp"] >= 180,     30, "Hypertensive crisis (SBP ≥180 mmHg) — immediate cardiovascular emergency", "critical"),
    InferenceRule(lambda d: 160 <= d["sbp"] < 180, 20, "Stage 2 hypertension — high short-term CVD event probability", "major"),
    InferenceRule(lambda d: 140 <= d["sbp"] < 160, 12, "Stage 1 hypertension — diabetes doubles CVD risk at this BP level", "major"),
    InferenceRule(lambda d: d["ldl"] >= 160,      20, "High LDL (≥160 mg/dL) — atherosclerotic plaque formation and rupture risk", "major"),
    InferenceRule(lambda d: 100 <= d["ldl"] < 160, 10, "LDL above optimal — statin therapy consideration warranted for diabetic patients", "minor"),
    InferenceRule(lambda d: d["hdl"] < 40,        15, "Low HDL (<40 mg/dL) — impaired reverse cholesterol transport; atherogenic", "major"),
    InferenceRule(lambda d: d["trig"] >= 500,      20, "Very high triglycerides — pancreatitis risk and severe atherogenicity", "critical"),
    InferenceRule(lambda d: 200 <= d["trig"] < 500, 12, "High triglycerides — metabolic syndrome, VLDL overproduction", "major"),
    InferenceRule(lambda d: d["smoking"] == "current", 20, "Active smoking — 2–4× higher CVD event risk in diabetic patients", "critical"),
    InferenceRule(lambda d: d["smoking"] == "former",   8, "Former smoking history — residual cardiovascular risk persists for years", "minor"),
    InferenceRule(lambda d: d["bmi"] >= 35,        12, "Class II obesity — independent cardiovascular risk factor; increases cardiac workload", "major"),
    InferenceRule(lambda d: 30 <= d["bmi"] < 35,    7, "Obesity — insulin resistance and adipokines worsen cardiac outcomes", "minor"),
    InferenceRule(lambda d: d["age"] > 55 and d["sex"] == "male",   10, "Male age >55 — elevated Framingham 10-year CVD risk", "major"),
    InferenceRule(lambda d: d["age"] > 65 and d["sex"] == "female",  8, "Female age >65 — post-menopausal CVD risk elevation", "minor"),
    InferenceRule(lambda d: d["hba1c"] >= 10,      12, "HbA1c ≥10% — each 1% HbA1c rise adds approximately 14% relative CVD event risk", "major"),
    InferenceRule(lambda d: d["cholesterol"] >= 240, 10, "Total cholesterol ≥240 mg/dL — high cardiovascular risk threshold", "major"),
    InferenceRule(lambda d: d["dbp"] >= 90,          8, "Diastolic hypertension — increased left ventricular afterload", "minor"),
]

# ─── DOMAIN 5: FOOT DISEASE ──────────────────
foot_rules = [
    InferenceRule(lambda d: d["foot_ulcers"] == "yes",   50, "Active foot ulcer — immediate podiatric assessment and wound care required", "critical"),
    InferenceRule(lambda d: d["tingling"] == "severe",   20, "Severe neuropathy — complete loss of protective sensation in feet", "major"),
    InferenceRule(lambda d: d["tingling"] == "moderate", 12, "Moderate neuropathy — significantly impaired foot sensation", "major"),
    InferenceRule(lambda d: d["smoking"] == "current",   15, "Smoking causes peripheral arterial disease — foot ischaemia and gangrene risk", "major"),
    InferenceRule(lambda d: d["diabetes_duration"] > 15, 15, "Long diabetes duration — cumulative vascular and nerve damage in lower extremities", "major"),
    InferenceRule(lambda d: d["hba1c"] >= 9,             12, "Poor glycaemic control — impairs wound healing and infection response", "major"),
    InferenceRule(lambda d: d["egfr"] < 60,              10, "CKD co-occurrence — amplified lower-limb complication and healing risk", "minor"),
    InferenceRule(lambda d: d["sbp"] >= 160,              8, "Severe hypertension — peripheral vascular disease acceleration", "minor"),
    InferenceRule(lambda d: d["bmi"] >= 35,               8, "Severe obesity — increased plantar pressure and wound chronicity", "minor"),
]


DOMAINS = [
    ComplicationDomain("nephropathy",    "Diabetic Nephropathy",     "🫘", "Kidney damage from chronic hyperglycaemia and hypertension", nephropathy_rules),
    ComplicationDomain("retinopathy",   "Diabetic Retinopathy",     "👁️", "Microvascular damage to retinal blood vessels", retinopathy_rules),
    ComplicationDomain("neuropathy",    "Diabetic Neuropathy",      "⚡", "Peripheral and autonomic nerve fibre damage", neuropathy_rules),
    ComplicationDomain("cardiovascular","Cardiovascular Disease",   "❤️", "Macrovascular coronary and cerebrovascular risk", cardiovascular_rules),
    ComplicationDomain("foot_disease",  "Diabetic Foot Disease",    "🦶", "Ulceration, infection and amputation risk", foot_rules),
]


# ─────────────────────────────────────────────
#  RECOMMENDATION ENGINE
# ─────────────────────────────────────────────

def generate_recommendations(results, d, overall_score):
    recs = []

    def add(priority, text):
        recs.append({"priority": priority, "text": text})

    scores = {r["key"]: r["score"] for r in results}

    if overall_score >= 70:
        add("urgent", "Multidisciplinary specialist referral — multiple high-risk domains detected. Endocrinology coordination with nephrology, ophthalmology and cardiology within 1–2 weeks.")
    if scores.get("nephropathy", 0) >= 40:
        add("urgent", "Initiate or intensify renoprotective therapy — ACE inhibitor or ARB is first-line. Target UACR reduction ≥30%. Nephrology referral and repeat renal panel in 3 months.")
    if scores.get("retinopathy", 0) >= 30:
        add("high", "Dilated fundus examination — arrange with ophthalmologist within 4 weeks. Intravitreal anti-VEGF or laser photocoagulation if proliferative changes confirmed.")
    if scores.get("neuropathy", 0) >= 30:
        add("high", "Neurological assessment — monofilament foot exam, vibration threshold, nerve conduction studies. Pharmacotherapy: pregabalin, duloxetine or tricyclics for symptomatic relief.")
    if scores.get("cardiovascular", 0) >= 40:
        add("urgent", "Cardiovascular risk reduction programme — high-intensity statin, antihypertensive optimisation, low-dose aspirin assessment. 10-year ASCVD risk score calculation.")
    if scores.get("foot_disease", 0) >= 30:
        add("high", "Podiatric assessment — daily self-inspection protocol, therapeutic footwear prescription, debridement if ulcer present. Vascular surgery if arterial insufficiency suspected.")
    if d.get("hba1c", 0) >= 9 or d.get("fbg", 0) >= 200:
        add("urgent", "Intensify glycaemic management immediately — review current regimen. Consider basal insulin initiation, GLP-1 receptor agonist, or SGLT-2 inhibitor. Target HbA1c <7.0%.")
    if d.get("sbp", 0) >= 140 or d.get("dbp", 0) >= 90:
        add("high", "Blood pressure optimisation — target <130/80 mmHg in diabetics. ACE inhibitors preferred for renoprotection. Consider 24hr ambulatory BP monitoring.")
    if d.get("ldl", 0) >= 100:
        add("high", "Lipid management — high-intensity statin indicated. Target LDL <70 mg/dL in diabetic patients with CVD risk factors. Recheck fasting lipids in 6–8 weeks.")
    if d.get("bmi", 0) >= 30:
        add("medium", "Weight management programme — 5–10% body weight reduction significantly improves all metabolic parameters. GLP-1 agonists offer dual glycaemic and weight benefit.")
    if d.get("smoking") == "current":
        add("urgent", "Smoking cessation — immediate priority. Tobacco doubles amputation risk and triples CVD events in diabetics. Refer to cessation programme with pharmacotherapy support.")
    if d.get("diabetes_duration", 0) > 5 and scores.get("retinopathy", 0) < 30:
        add("routine", "Annual dilated eye exam — with ≥5 years diabetes duration, yearly ophthalmology screening is ADA guideline-mandated even without current symptoms.")
    add("routine", "Lifestyle intervention — 150 min/week moderate aerobic activity, Mediterranean dietary pattern, ≥7hr sleep, stress reduction. These independently reduce HbA1c by 0.5–1.0%.")
    add("routine", "Structured monitoring schedule — HbA1c every 3 months until at target, then every 6 months. Annual lipid panel, renal function, urine albumin, foot exam, eye exam.")

    return recs


# ─────────────────────────────────────────────
#  OVERALL RISK COMPUTATION
# ─────────────────────────────────────────────

def compute_overall_risk(results):
    scores = [r["score"] for r in results]
    peak   = max(scores)
    mean   = sum(scores) / len(scores)
    # Weighted: 50% peak + 50% mean
    overall = round(peak * 0.5 + mean * 0.5)
    return min(overall, 100)


def risk_level(score):
    if score >= 70: return "HIGH RISK",        "Significant complications detected — urgent review required"
    if score >= 50: return "MODERATE-HIGH",    "Multiple risk domains elevated — structured follow-up needed"
    if score >= 30: return "BORDERLINE",       "Early warning signs present — preventive action recommended"
    return          "LOW RISK",                "No major markers detected — continue monitoring schedule"


def build_ai_summary(results, d, overall_score):
    top = sorted(results, key=lambda r: r["score"], reverse=True)
    highest = top[0]
    age_str = f"{d['age']}-year-old " if d.get("age") else ""
    sex_str = d.get("sex", "")
    dur_str = f"{d['diabetes_duration']}-year " if d.get("diabetes_duration") else ""
    type_str = d.get("diabetes_type", "diabetes").replace("type1","Type 1").replace("type2","Type 2")
    level, _ = risk_level(overall_score)
    rules_total = sum(r["rules_triggered"] for r in results)
    domains_flagged = sum(1 for r in results if r["score"] > 0)

    summary = (
        f"The rule-based inference engine has completed multi-domain analysis of a "
        f"{age_str}{sex_str} patient with {dur_str}{type_str} history. "
        f"Overall complication risk is assessed as {level} with a composite score of {overall_score}/100. "
        f"A total of {rules_total} clinical rules were triggered across {domains_flagged} complication domains.\n\n"
        f"The highest-priority finding is in the {highest['name']} domain (score: {highest['score']}/100). "
    )
    if highest["findings"]:
        summary += f"Primary driver: {highest['findings'][0]['text'][:120]}.\n\n"

    if d.get("hba1c", 0) >= 9:
        summary += (f"Suboptimal glycaemic control (HbA1c {d['hba1c']}%) is a cross-cutting risk driver "
                    f"amplifying damage across all five complication domains simultaneously. "
                    f"This represents the highest-yield single intervention target.\n\n")

    neph_score = next((r["score"] for r in results if r["key"]=="nephropathy"), 0)
    retin_score = next((r["score"] for r in results if r["key"]=="retinopathy"), 0)
    if neph_score >= 30 and retin_score >= 30:
        summary += ("Co-occurrence of nephropathy and retinopathy signals systemic microvascular disease. "
                    "These share pathophysiological mechanisms (AGE accumulation, PKC activation, oxidative stress) "
                    "and typically progress in parallel — addressing one benefits the other.\n\n")

    summary += (f"Inference engine: Python rule-based expert system · {len(DOMAINS)} domains · "
                f"{sum(len(d.rules) for d in DOMAINS)} clinical rules evaluated.")
    return summary


# ─────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/analyse", methods=["POST"])
def analyse():
    data = request.get_json(force=True)

    # Sanitise + default missing fields
    d = {
        "age":               float(data.get("age") or 0),
        "sex":               data.get("sex", "male"),
        "diabetes_type":     data.get("diabetesType", "type2"),
        "diabetes_duration": float(data.get("diabetesDuration") or 0),
        "bmi":               float(data.get("bmi") or 0),
        "smoking":           data.get("smoking", "never"),
        "fbg":               float(data.get("fbg") or 0),
        "hba1c":             float(data.get("hba1c") or 0),
        "ppg":               float(data.get("ppg") or 0),
        "insulin":           float(data.get("insulin") or 0),
        "sbp":               float(data.get("sbp") or 0),
        "dbp":               float(data.get("dbp") or 0),
        "cholesterol":       float(data.get("cholesterol") or 0),
        "ldl":               float(data.get("ldl") or 0),
        "hdl":               float(data.get("hdl") or 0),
        "trig":              float(data.get("trig") or 0),
        "creatinine":        float(data.get("creatinine") or 0),
        "egfr":              float(data.get("egfr") or 0),
        "uacr":              float(data.get("uacr") or 0),
        "bun":               float(data.get("bun") or 0),
        "tingling":          data.get("tingling", "no"),
        "vision":            data.get("vision", "no"),
        "foot_ulcers":       data.get("footUlcers", "no"),
        "hr":                float(data.get("hr") or 0),
    }

    # Run inference
    results = [domain.evaluate(d) for domain in DOMAINS]
    overall_score = compute_overall_risk(results)
    level, description = risk_level(overall_score)
    recommendations = generate_recommendations(results, d, overall_score)
    ai_summary = build_ai_summary(results, d, overall_score)

    return jsonify({
        "overall_score": overall_score,
        "risk_level": level,
        "risk_description": description,
        "results": results,
        "recommendations": recommendations,
        "ai_summary": ai_summary,
        "rules_evaluated": sum(len(dom.rules) for dom in DOMAINS),
        "domains": len(DOMAINS),
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "engine": "DiabetaGuard Python Inference Engine",
                    "domains": len(DOMAINS),
                    "total_rules": sum(len(d.rules) for d in DOMAINS)})


if __name__ == "__main__":
    app.run(debug=True, port=5050)
