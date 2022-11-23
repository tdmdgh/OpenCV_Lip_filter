# OpenCV_Lip_filter

이 프로젝트의 목적은 인물 사진의 입술에 립스틱을 바른것처럼 화장을 시켜주는 필터를 만드는 것이다.

shape_predictor_68_face_landmarks.dat을 기반으로 facial-landmarks를 감지할 수 있었다.

68개의 점들중 입을 표현하는 48번 점부터 68번의 점까지만 추출해서 그 위에 색을 입혔다.

입술에 입힐 수 있는 색은 총 13가지로, 색상을 골라 cv2.drawContours()를 사용해 인물사진의 그 색상코드에 맞는 색을 overlay한다.


#소스 참조 
### https://github.com/PyImageSearch/imutils/blob/9f740a53bcc2ed7eba2558afed8b4c17fd8a1d4c/imutils/face_utils/helpers.py
