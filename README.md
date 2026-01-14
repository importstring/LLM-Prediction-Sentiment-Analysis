# LLM-Prediction-Sentiment-Analysis
Through some Ollama models and pure Python the code retrieves a variety of outputs from a diverse range of LLMs and outputs a score either for or against the ticker. I originally made this project a while back but I thought it was worth revisited, modernizing it and making it significantly more presentable.

Currrently I'm thinking of doing:
project-root/
├─ frontend/
│  └─ ... (existing TS app)
├─ algorithm/
│  ├─ pyproject.toml / setup.cfg (optional, but good)
│  ├─ main.py              # CLI / dev entrypoint only
│  └─ algopkg/             # Python package
│     ├─ __init__.py       # very minimal
│     ├─ config.py         # paths, settings
│     ├─ utils/
│     │  ├─ __init__.py
│     │  └─ io.py, logging.py, math_utils.py, ...
│     ├─ data_models/
│     │  ├─ __init__.py
│     │  ├─ portfolio.py
│     │  ├─ stock_data.py
│     │  └─ dataclasses for configs, results, etc.
│     ├─ agents/
│     │  ├─ __init__.py
│     │  ├─ agent.py
│     │  ├─ prompt_manager.py
│     │  ├─ action_history.py
│     │  └─ analysis.py     # technical/fundamental helpers
│     ├─ llm_clients/
│     │  ├─ __init__.py
│     │  ├─ chatgpt_client.py
│     │  ├─ perplexity_client.py
│     │  └─ ollama_client.py
│     ├─ api/
│     │  ├─ __init__.py
│     │  └─ service.py      # thin functions used by backend/TS
│     └─ progress_ui/
│        ├─ __init__.py
│        └─ rich_progress.py
├─ data/
│  ├─ conversations/
│  ├─ stock_data/
│  ├─ databases/
│  │  └─ portfolio/
│  ├─ info/
│  │  └─ tickers/
│  └─ rankings/
└─ scripts/
   └─ llm_insights/
      └─ api_keys/
