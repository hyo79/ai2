#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
#파일 이름 streamlit_app.py
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '1hB__Hfh_J6xD-1RsbnorI3YPwN4u6Z-E'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 당신의 취향은??")
    if image is not None:
        st.image(image, caption="당신의 최애 웹툰", use_column_width=True)
    st.write(f"아마도 이 표지는: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 비슷한 장르의 영상을 추천해드려요!")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(3):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_column_width=True)
    # 2nd Row - YouTube Videos
    for i in range(3):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(3):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': [
            "https://i.ibb.co/p0KgMYK/r1.jpg",
            "https://i.ibb.co/jDYrQ3f/r2.jpg",
            "https://i.ibb.co/LNNWVBM/r3.webp"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=n8GcBstSFD8",
            "https://www.youtube.com/watch?v=Ceoe2wf-bbo",
            "https://www.youtube.com/watch?v=5VPwCGe9S5c"
        ],
        'texts': [
            "두근두근 로맨스 광인!!",
            "당신의 취향을 저격할",
            "영상입니다"
        ]
    },
    labels[1]: {
        'images': [
            "https://i.ibb.co/sC1B4rF/s1.png",
            "https://i.ibb.co/yWbnTsf/s2.webp",
            "https://i.ibb.co/kHjKfrw/s3.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=biPh95JF3no",
            "https://www.youtube.com/watch?v=dji5vHuoVtM",
            "https://www.youtube.com/watch?v=Jd-EXmSxV2A"
        ],
        'texts': [
            "오싹오싹 스릴러 덕후!!",
            "당신의 취향을 저격할",
            "영상입니다 (심장주의)"
        ]
    },
    labels[2]: {
        'images': [
            "https://i.ibb.co/jVBcjF1/pan1.jpg",
            "https://i.ibb.co/ZN4BBWW/pan2.webp",
            "https://i.ibb.co/VvkgThL/pan3.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=dlSkyNLOncY",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=-5Dc8EMVYBo"
        ],
        'texts': [
            "꿈과 같은 경험을 선사하는 판타지!!",
            "당신의 취향을 저격할",
            "영상입니다"
        ]
    }
}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)

