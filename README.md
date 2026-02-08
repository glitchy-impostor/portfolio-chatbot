# Final Deployment Checklist - Merged Railway Backend

## Files You Need to Upload to Railway

Your Railway project should have this structure:

```
your-railway-project/
├── api.py                      ← REPLACE with api-merged-final.py
├── requirements.txt            ← ADD football dependencies
├── index.faiss                 ← Keep existing (portfolio)
├── chunks.pkl                  ← Keep existing (portfolio)
├── ingest.py                   ← Keep existing (portfolio)
│
└── football-chatbot/           ← ADD this entire folder
    ├── pipelines/
    │   ├── __init__.py
    │   ├── router.py           ← Use router-fixed.py
    │   └── executor.py
    ├── models/
    │   ├── __init__.py
    │   ├── epa_model.py
    │   ├── drive_simulator.py
    │   └── team_profiler.py
    ├── formatters/
    │   ├── __init__.py
    │   └── response_formatter.py
    ├── context/
    │   ├── __init__.py
    │   └── presets.py
    ├── llm/
    │   ├── __init__.py
    │   ├── client.py
    │   ├── handler.py
    │   └── prompts.py
    └── trained_models/
        ├── epa_model.joblib
        ├── team_profiles.json
        └── player_estimates.json
```

---

## Step-by-Step Instructions

### Step 1: Replace api.py
Replace your existing `api.py` with `api-merged-final.py` (rename it to `api.py`)

### Step 2: Update requirements.txt
Add these lines to your existing requirements.txt:
```
psycopg2-binary>=2.9.6
pandas>=2.0.0
pyarrow>=12.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

### Step 3: Add football-chatbot folder
Copy the entire `football-chatbot` folder from your local project to Railway.

**IMPORTANT:** Make sure to use `router-fixed.py` as `pipelines/router.py`

### Step 4: Set Environment Variables in Railway Dashboard
Add these variables:
```
FOOTBALL_DATABASE_URL=postgresql://your-db-connection-string
FOOTBALL_RATE_LIMIT_PER_DAY=100
FOOTBALL_CHATBOT_PATH=./football-chatbot
```

### Step 5: Deploy
```bash
git add .
git commit -m "Add football analytics"
git push
```

---

## Files Included in This Package

| File | What to do |
|------|------------|
| `api-merged-final.py` | Rename to `api.py` and replace existing |
| `router-fixed.py` | Put in `football-chatbot/pipelines/router.py` |
| `requirements-additional.txt` | Add contents to your `requirements.txt` |

---

## Verification After Deploy

Test these endpoints:

```bash
# Portfolio (existing)
curl https://web-production-12eeb.up.railway.app/health

# FlashRead (existing)  
curl https://web-production-12eeb.up.railway.app/flashread/health

# Football (new)
curl https://web-production-12eeb.up.railway.app/football/health

# Football chat (new)
curl -X POST https://web-production-12eeb.up.railway.app/football/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about the Chiefs"}'

# Test the router fix
curl -X POST https://web-production-12eeb.up.railway.app/football/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Should I pass or run at the 4th yard line on 4th down?"}'
```

---

## Troubleshooting

### "Football analytics not configured"
- Check FOOTBALL_DATABASE_URL is set
- Check football-chatbot folder exists with all subfolders
- Check Railway logs for import errors

### "Down and distance required" on natural language queries
- Make sure router-fixed.py is in football-chatbot/pipelines/router.py
- NOT the old router.py

### Import errors
- Make sure all __init__.py files are present in each folder
- Check that trained_models/ has the .joblib and .json files
