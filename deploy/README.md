В одном терминале 

cd /deploy/backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000


В другом терминале 


cd /deploy/frontend
streamlit run app.py