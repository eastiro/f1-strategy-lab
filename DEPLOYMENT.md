# Deployment Guide

The public V4 application is `app/streamlit_app.py`. It is a shareable,
interactive research case study rather than a live strategy service.

Current public deployment:
[f1-strategy-lab-dzqqe3ga9salsqhshfnsfl.streamlit.app](https://f1-strategy-lab-dzqqe3ga9salsqhshfnsfl.streamlit.app/)

## Recommended: Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app from the repository.
4. Select `app/streamlit_app.py` as the entry point.
5. Choose Python 3.12.
6. Choose a memorable free `streamlit.app` subdomain, for example
   `f1-strategy-lab.streamlit.app`, if it is available.
7. Set the app to public and verify the generated URL in a signed-out browser.

No API key or live OpenF1 request is required at startup. The required
processed data and audit outputs are bundled in the repository.

## Docker

```powershell
docker build -t f1-strategy-lab .
docker run --rm -p 8501:8501 f1-strategy-lab
```

Open `http://localhost:8501/`.

## Pre-Publish Checklist

- Run `python -m unittest discover -s tests -p "test_*.py"`.
- Open the four V4 views once on desktop.
- Check the top section and navigation at a narrow mobile width.
- Change one free-experiment control and confirm the chart updates.
- Open the three collapsed research-boundary evidence sections.
- Confirm the page contains no raw traceback or missing-file warning.
- Confirm the public URL works without signing in.
- Add the final public URL to this README and to the resume project entry.

A separately purchased domain is not required. The free custom
`*.streamlit.app` subdomain is the canonical project URL.
