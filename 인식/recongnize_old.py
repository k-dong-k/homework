from paddleocr import PaddleOCR, draw_ocr
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# OCR 객체 생성 (한국어)
ocr = PaddleOCR(use_angle_cls=True, lang='korean')

# 이미지 불러오기
image_path = '12333.png'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# OCR 수행
results = ocr.ocr(image, cls=True)

# 결과 정리
boxes = []
texts = []

for line in results:
    for (box, (text, score)) in line:
        boxes.append(box)
        texts.append(text)

# PIL 이미지로 변환 (한글 텍스트 그리기 위함)
image_pil = Image.fromarray(image_rgb)
draw = ImageDraw.Draw(image_pil)

# 한글 폰트 설정
# font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows
font_path = "/System/Library/Fonts/AppleGothic.ttf"  # macOS
# font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # Linux 예시

font = ImageFont.truetype(font_path, 20)

# 바운딩 박스 및 텍스트 그리기
for box, text in zip(boxes, texts):
    # box는 4개의 좌표 (좌상단부터 시계방향)
    points = [tuple(map(int, pt)) for pt in box]
    draw.line(points + [points[0]], fill=(0, 255, 0), width=2)  # 초록색 테두리
    draw.text(points[0], text, font=font, fill=(255, 0, 0))     # 빨간 텍스트

# 시각화용 OpenCV 이미지로 변환
result_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# 전체 텍스트 출력 (옵션)
print("Detected Text:", ' '.join(texts))

# 결과 이미지 보여주기
cv2.imshow("PaddleOCR Result", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
