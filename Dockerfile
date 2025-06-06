# Python 3.10 기반
FROM python:3.10-slim

# 작업 디렉터리 설정
WORKDIR /app

# 필요한 파일 복사
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 포트 노출
EXPOSE 8501

# Streamlit 실행 명령
CMD ["streamlit", "run", "server.py", "--server.port=8501", "--server.enableCORS=false"]
