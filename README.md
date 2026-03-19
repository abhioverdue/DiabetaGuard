# DiabetaGuard — Python Flask Clinical Inference Engine

A rule-based AI expert system for early detection of diabetic complications.

## Architecture
```
Frontend (HTML/CSS/JS)  →  POST /api/analyse  →  Python Flask Backend
                        ←  JSON response      ←  Rule-Based Engine
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Flask server
```bash
python app.py
```

### 3. Open in browser
Navigate to: http://localhost:5050

## API Endpoints

### POST /api/analyse
Accepts patient data JSON, returns risk scores and recommendations.

**Request body:**
```json
{
  "age": 55, "sex": "male", "diabetesType": "type2",
  "diabetesDuration": 10, "bmi": 30.5, "smoking": "current",
  "fbg": 160, "hba1c": 9.2, "sbp": 145, "dbp": 92,
  "ldl": 145, "hdl": 38, "trig": 220,
  "egfr": 62, "uacr": 55, "creatinine": 1.3,
  "tingling": "moderate", "vision": "mild"
}
```

**Response:**
```json
{
  "overall_score": 68,
  "risk_level": "MODERATE-HIGH",
  "risk_description": "...",
  "results": [...],
  "recommendations": [...],
  "ai_summary": "...",
  "rules_evaluated": 47,
  "domains": 5
}
```

### GET /api/health
Returns engine status and loaded rules count.

## Inference Engine

- **5 Complication Domains**: Nephropathy, Retinopathy, Neuropathy, Cardiovascular, Foot Disease
- **47 Clinical Rules**: Each with weighted scoring and severity categorisation
- **Composite Scoring**: 50% peak domain score + 50% mean across domains
- **Recommendations Engine**: Priority-ranked clinical actions based on triggered rules

## File Structure
```
diabetaguard/
├── app.py              # Flask server + Python inference engine
├── requirements.txt    # Dependencies
├── README.md
└── static/
    └── index.html      # Frontend (served by Flask)
```
