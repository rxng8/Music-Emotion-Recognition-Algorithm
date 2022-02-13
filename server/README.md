# Server for serving music emotion recognition model

## Requirements:
1. System: Window, Linux, and Mac.
2. Python

## Running the server:
1. Export the model to `server/model/my_model`
2. Running
  ```
  cd server
  pip install -r requirements.txt
  python app.py
  ```
3. Or to start a deployment server, run:
```
uvicorn app:app --host 0.0.0.0 --port 80
```

