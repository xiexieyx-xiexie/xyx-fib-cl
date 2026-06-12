# fib Chloride Ingress Reliability

This repository contains a browser-based Monte Carlo implementation of the fib chloride
ingress service-life model.

## Run locally

Serve the repository root with any static file server, then open `index.html`.

```bash
python -m http.server 8000
```

The calculations run in a Web Worker, so no backend or environment variables are required.

## Test

```bash
node --test tests/simulation.test.mjs
```

## Deploy

The project is configured as a static Vercel deployment. Import the GitHub repository in
Vercel or deploy the repository root with the Vercel CLI.

The original Streamlit implementation remains available in `app.py` for reference.
