# Python 3.9 베이스 이미지 사용
FROM python:3.9-slim

# Streamlit에서 터미널 오류 방지용
ENV PYTHONUNBUFFERED=1

# 작업 디렉토리 생성
WORKDIR /app

# requirements.txt 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY . .

# 8501 포트 노출
EXPOSE 8501

# Streamlit 실행
CMD ["streamlit", "run", "server.py", "--server.port=8501", "--server.enableCORS=false"]
