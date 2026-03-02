import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
import os

# 화면 기본 설정
st.set_page_config(page_title="100m Sprint AI", page_icon="⚡", layout="wide")

st.title("⚡ 100m 달리기 리얼 역학 분석 AI")
st.write("안전하고 과학적인 자세 교정부터 세계 제패까지, 단거리 달리기 맞춤형 AI 역학 분석")
st.warning("🔒 업로드된 모든 영상은 AI가 분석 후 임시 처리되어 즉시 소멸됩니다. 서버에 저장되지 않습니다.")
st.write("---")

# 1. 기본 정보 입력
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    gender = st.radio("1️⃣ 성별", ["남성", "여성"], horizontal=True)
with col_info2:
    region = st.radio("2️⃣ 기준", ["글로벌 기준", "대한민국 기준"], horizontal=True)
with col_info3:
    # 한국/글로벌 기준과 성별을 조합하여 목표 기록 세분화
    if region == "글로벌 기준":
        if gender == "남성":
            elite_label = "글로벌 엘리트 (서브 9.9초) 🌍"
        else:
            elite_label = "글로벌 엘리트 (서브 10.8초) 🌍"
    else:
        if gender == "남성":
            elite_label = "한국 엘리트 (서브 10.1초) 🇰🇷"
        else:
            elite_label = "한국 엘리트 (서브 11.4초) 🇰🇷"
            
    target_time = st.selectbox("3️⃣ 목표 기록", [elite_label, "11초대 진입", "12초대 진입", "자세 교정 및 부상 방지"])

st.write("---")

# 2. 달리기 영상 업로드
st.subheader("4️⃣ 100m 스프린트 영상 업로드")
st.error("⚠️ 정밀한 AI 분석을 위해 반드시 **10초 이내의 측면 영상**만 올려주세요! (최대 속도 구간 권장)")
video_file = st.file_uploader("최고 속도로 달리는 측면 영상을 올려주세요 (MP4/MOV)", type=['mp4', 'mov', 'avi'])
st.write("---")

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

if video_file is not None and st.button("실시간 역학 분석 및 처방 발급하기"):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # 100m 핵심 역학 지표
    min_drive_angle = 180.0  # 무릎 거상 (어깨-골반-무릎). 작을수록 다리를 높이 든 것.
    max_push_angle = 0.0     # 후방 추진 (골반-무릎-발목). 클수록 끝까지 밀어낸 것.
    
    drive_frame = None
    push_frame = None
    frame_count = 0

    with st.spinner("AI가 100m 스프린트 프레임을 정밀 분석 중입니다..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            if frame_count % 3 != 0: # 속도 최적화를 위해 3프레임 스킵
                continue

            # 해상도 제한 (메모리 보호)
            h, w = frame.shape[:2]
            if w > 800:
                frame = cv2.resize(frame, (800, int(h * 800 / w)))

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)
            
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                
                # 왼쪽 다리 기준 추출
                l_s = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                l_h = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
                l_k = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y]
                l_a = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y]

                # 1. 무릎 거상 각도 (Knee Drive Angle)
                drive_angle = calculate_angle(l_s, l_h, l_k)
                if drive_angle < min_drive_angle:
                    min_drive_angle = drive_angle
                    annotated_drive = img_rgb.copy()
                    mp_drawing.draw_landmarks(annotated_drive, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    drive_frame = annotated_drive

                # 2. 후방 추진 각도 (Push-off Angle)
                push_angle = calculate_angle(l_h, l_k, l_a)
                if push_angle > max_push_angle:
                    max_push_angle = push_angle
                    annotated_push = img_rgb.copy()
                    mp_drawing.draw_landmarks(annotated_push, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    push_frame = annotated_push

        cap.release()

    # 영상 즉시 영구 파기
    try:
        os.unlink(tfile.name)
    except:
        pass

    if drive_frame is not None and push_frame is not None:
        st.subheader(f"📐 [{region} / {gender}] 100m 스프린트 생체역학 판독")
        col1, col2 = st.columns(2)
        with col1:
            st.image(drive_frame, channels="RGB", use_column_width=True)
            st.markdown(f"**⚡ 무릎 거상 각도 (Knee Drive): {min_drive_angle:.1f}도**")
            st.caption("어깨-골반-무릎 각도. 90도에 가까울수록 허벅지를 지면과 평행하게 잘 들어 올린 것입니다.")

        with col2:
            st.image(push_frame, channels="RGB", use_column_width=True)
            st.markdown(f"**🚀 추진 신전 각도 (Push-off): {max_push_angle:.1f}도**")
            st.caption("골반-무릎-발목 각도. 180도에 가까울수록 지면을 끝까지 강력하게 밀어낸 것입니다.")
        
        st.write("---")
        st.subheader("🎓 전문 생체역학 리포트 및 처방")
        
        # 글로벌 엘리트 vs 한국 엘리트 비교 로직
        if "글로벌 엘리트" in target_time:
            # 글로벌 기준: 거상 95도 이하 (더 높이 듦), 신전 165도 이상 (끝까지 밈)
            if min_drive_angle > 95:
                st.error(f"⚠️ **[글로벌 스탠다드 경고 / 보폭(Stride) 손실]**\n\n**[팩트]** 현재 무릎 거상 각도는 **{min_drive_angle:.1f}도**로, 우사인 볼트나 글로벌 탑 랭커들(90~95도)에 비해 허벅지가 충분히 올라오지 않습니다.\n\n**[처방]** 전방 대퇴사두근과 장요근의 폭발적인 수축이 필요합니다. 'A-Skip' 훈련 비중을 높여 체공 시간을 확보하세요.")
            elif max_push_angle < 165:
                st.warning(f"⚠️ **[글로벌 스탠다드 경고 / 추진력(Power) 누수]**\n\n**[팩트]** 지면을 차고 나가는 순간의 각도가 **{max_push_angle:.1f}도**로, 지면 반발력을 끝까지 뽑아내지 못하고 일찍 발을 뗍니다.\n\n**[처방]** 후면 사슬(둔근과 햄스트링)의 폭발력이 부족합니다. 발목의 배측굴곡을 유지하며 트랙을 완전히 부수듯 밀어내는 연습이 필요합니다.")
            else:
                st.success(f"🔥 **[글로벌 엘리트 통과 / 월드클래스 메커니즘]**\n\n**[팩트]** 무릎 거상({min_drive_angle:.1f}도)과 후방 추진({max_push_angle:.1f}도) 모두 올림픽 파이널리스트 수준의 완벽한 탄성을 보여줍니다. 100%의 힘이 전방으로 전환되고 있습니다.")
                
        elif "한국 엘리트" in target_time:
            # 한국 기준: 거상 100도 이하, 신전 160도 이상
            if min_drive_angle > 100:
                st.error(f"⚠️ **[국내 엘리트 경고 / 숏 피치 발생]**\n\n**[팩트]** 무릎 거상 각도가 **{min_drive_angle:.1f}도**입니다. 다리를 들어 올리는 궤적이 짧아 속도가 붙을수록 다리가 뒤로 헛도는 '백킥(Back-kick)' 현상이 발생할 수 있습니다.\n\n**[처방]** 골반의 가동 범위를 넓혀야 합니다. 무릎을 가슴 쪽으로 당겨오는 리커버리 속도를 0.1초 단축하세요.")
            elif max_push_angle < 160:
                st.warning(f"⚠️ **[국내 엘리트 경고 / 불완전 신전]**\n\n**[팩트]** 추진 신전 각도가 **{max_push_angle:.1f}도**에 머물러 있습니다. 국내 최고 기록을 깨기엔 지면을 누르는 파워가 약합니다.\n\n**[처방]** 플라이오메트릭(Plyometric) 훈련을 통해 지면 접촉 시간을 줄이면서도 강력하게 밀어내는 '트리플 익스텐션(Triple Extension)' 타이밍을 맞추세요.")
            else:
                st.success(f"🔥 **[한국 엘리트 통과 / 국가대표급 폼]**\n\n**[팩트]** 거상({min_drive_angle:.1f}도)과 추진({max_push_angle:.1f}도)의 밸런스가 매우 훌륭합니다. 한국 신기록을 향한 완벽한 역학적 기본기를 갖췄습니다.")
        else:
            # 일반 동호인 / 학생 선수 기준
            if min_drive_angle > 110:
                st.info(f"💡 **[피드백]** 무릎이 충분히 올라오지 않아 다리가 무겁게 느껴질 수 있습니다. 허벅지를 배꼽 높이까지 올린다는 느낌으로 질주하세요.")
            else:
                st.success(f"✅ **[안정적 자세]** 다리 회전 리듬이 매우 좋습니다. 현재의 폼에서 근력을 키우는 데 집중하세요.")

        st.write("---")
        st.success("✅ 프라이버시 보호를 위해 업로드된 영상은 분석 직후 서버에서 영구 삭제되었습니다.")
    else:
        st.error("⚠️ 영상에서 선수의 동작을 인식하지 못했습니다. 전신이 잘 보이는 측면 영상을 다시 올려주세요.")

st.write("---")
st.info("💡 **서비스 개선을 위한 소중한 의견을 들려주세요!**\n\n기능 제안, 오작동 신고 등 어떤 피드백이든 환영합니다.\n\n📧 **[youclsrn1@gmail.com](mailto:youclsrn1@gmail.com)**")

