# Waffle App

Waffle_app is a streamlit demo for easy delivery of [Waffle](https://snuailab.github.io/waffle/).

## Install

### pip

```bash
pip install -r requirements.txt
```

## Run

### streamlit

```bash
streamlit run main.py --server.runOnSave False --server.allowRunOnSave False --server.fileWatcherType none --server.port <PORT>
```

## Docker

### Build

```bash
./build_docker.sh
```

### Run Docker-compose

```bash
docker compose up -d
```
