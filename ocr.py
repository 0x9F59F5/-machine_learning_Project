import pytesseract
import cv2
import matplotlib.pyplot as plt
import os

# 이미지 파일 경로
path = '이미지 절대 경로'

# 파일 경로가 존재하는지 확인
if not os.path.exists(path):
    print(f"오류: 파일이 {path} 위치에 없습니다.")
else:

    image = cv2.imread(path)

    # 이미지가 제대로 읽혔는지 확인
    if image is None:
        print(f"오류: 이미지 파일 {path}을(를) 읽을 수 없습니다.")
    else:
        # 이미지를 그레이스케일로 변환
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 이미지 이진화
        _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        # 이미지에서 텍스트 추출
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(binary_image, lang='kor+eng', config=custom_config)
        print(text)

        boxes = pytesseract.image_to_boxes(binary_image, config=custom_config)

        # 이미지와 텍스트에 바운딩 박스를 그림
        for b in boxes.splitlines():
            b = b.split(' ')
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(image, (x, gray_image.shape[0] - y), (w, gray_image.shape[0] - h), (0, 255, 0), 1)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Extracted Text')
        plt.show()
