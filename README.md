# personal-chat-ai

## How to run

```bash

# first install packages and creates venv
uv sync

# isntall like a package so it can be imported
uv pip install -e .


#run api
./start.sh

#run tests
pytest tests/test_gemini_simple.py -v



source .venv/bin/activate
```