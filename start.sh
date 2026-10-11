#!/bin/sh
# Runs the whole app in one container: the API on an internal port, the web interface on the public one.
set -e

uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000 &

# Wait until the API has loaded the model and the index (at most 120 s).
i=0
until python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" 2>/dev/null; do
    i=$((i + 1))
    if [ "$i" -ge 120 ]; then echo "API did not start" >&2; exit 1; fi
    sleep 1
done

export API_URL=http://127.0.0.1:8000  # the web interface calls the API above
exec streamlit run src/app.py --server.address 0.0.0.0 --server.port 7860 --server.headless true