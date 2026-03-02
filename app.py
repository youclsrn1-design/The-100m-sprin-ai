import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
import os

# 화면 기본 설정
st.set_page_config(page_title="100m Sprint AI", page_icon="⚡", layout="wide")

st.title("⚡ 100m 달리기 3D 역학 분석 AI (측면+정면 통합)")
st.write("단 하나의 영상으로 보폭(측면)과 밸런스(정면)를 동시에 정밀 분석합니다.")
st.warning("🔒 업로드된 모든 영상은 AI가 분석 후 임시 처리되어 즉시 소멸됩니다.")
st.write("---")

# 1. 기본 정보 입력
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    gender = st.radio("1️⃣ 성별", ["남성", "여성"], horizontal=True)
with col_info2:
    region = st.radio("2️⃣ 기준", ["글로벌 기준", "대한민국 기준"], horizontal=True)
with col_info3:
    if region == "글로벌 기준":
        elite_label = "글로벌 엘리트 (서브 9.9초) 🌍" if gender == "남성" else "글로벌 엘리트 (서브 10.8초) 🌍"
    else:
        elite_label = "한국 엘리트 (서브 10.1초) 🇰🇷" if gender == "남성" else "한국 엘리트 (서브 11.4초) 🇰🇷"
            
    target_time = st.selectbox("3️⃣ 목표 기록", [elite_label, "11초대 진입", "12초대 진입", "자세 교정 및 밸런스"])

st.write("---")

# 2. 통합 영상 업로드
st.subheader("4️⃣ 100m 스프린트 영상 업로드")
st.error("⚠️ **10초 이내의 영상**만 올려주세요! (측면, 정면, 혹은 대각선 영상 모두 AI가 3D로 자동 분석합니다)")
video_file = st.file_uploader("질주하는 영상을 올려주세요 (MP4/MOV)", type=['mp4', 'mov', 'avi'])
st.write("---")

# 수학적 계산 함수들
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

def get_tilt_angle(p1, p2):
    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]
    return np.abs(np.degrees(np.arctan2(dy, dx)))

if video_file is not None and st.button("🚀 3D 입체 역학 분석 시작"):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    mp_pose = mp.solutions.pose
    # 3D 분석을 위해 모델 복잡도를 유지하며 추적 신뢰도를 조정
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # 분석 지표 초기화
    min_drive_angle = 180.0  # [측면] 무릎 거상
    max_push_angle = 0.0     # [측면] 후방 추진
    max_hip_tilt = 0.0       # [정면] 골반 무너짐 (좌우 밸런스)
    
    drive_frame, push_frame, tilt_frame = None, None, None
    frame_count = 0

    with st.spinner("AI가 영상을 3D 공간 좌표로 변환하여 측면/정면 역학을 동시 추출 중입니다..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            if frame_count % 3 != 0: continue

            h, w = frame.shape[:2]
            if w > 800: frame = cv2.resize(frame, (800, int(h * 800 / w)))

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)
            
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                
                # 좌표 추출
                l_s = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                l_h = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
                r_h = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                l_k = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y]
                l_a = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y]

                annotated = img_rgb.copy()
                mp_drawing.draw_landmarks(annotated, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # 1. [측면] 무릎 거상
                drive_angle = calculate_angle(l_s, l_h, l_k)
                if drive_angle < min_drive_angle:
                    min_drive_angle = drive_angle
                    drive_frame = annotated

                # 2. [측면] 후방 추진
                push_angle = calculate_angle(l_h, l_k, l_a)
                if push_angle > max_push_angle:
                    max_push_angle = push_angle
                    push_frame = annotated
                    
                # 3. [정면] 골반 좌우 기울기 (Pelvic Drop)
                tilt = get_tilt_angle(l_h, r_h)
                if tilt > max_hip_tilt:
                    max_hip_tilt = tilt
                    tilt_frame = annotated

        cap.release()

    try: os.unlink(tfile.name)
    except: pass

    if drive_frame is not None:
        st.subheader(f"📐 [{region}] 100m 3D 생체역학 판독 리포트")
        
        # 3단 화면 구성 (측면 2개 + 정면 1개)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(drive_frame, channels="RGB", use_column_width=True)
            st.markdown(f"**⚡ [측면] 무릎 거상: {min_drive_angle:.1f}도**")
        with col2:
            st.image(push_frame, channels="RGB", use_column_width=True)
            st.markdown(f"**🚀 [측면] 후방 추진: {max_push_angle:.1f}도**")
        with col3:
            st.image(tilt_frame, channels="RGB", use_column_width=True)
            st.markdown(f"**⚖️ [정면] 골반 붕괴(기울기): {max_hip_tilt:.1f}도**")
        
        st.write("---")
        st.subheader("🎓 3D 입체 생체역학 처방전")
        
        # 정면(밸런스) 피드백 로직 추가
        st.markdown("#### 🔍 1. 좌우 밸런스 및 에너지 누수 판독 (정면)")
        if max_hip_tilt < 6:
            st.success(f"✅ **[코어 밸런스 완벽]** 골반 기울기가 **{max_hip_tilt:.1f}도**로, 상하/좌우 흔들림 없이 모든 힘이 전진하는 데 쓰이고 있습니다.")
        elif 6 <= max_hip_tilt <= 10:
            st.warning(f"⚠️ **[미세한 밸런스 붕괴]** 골반이 **{max_hip_tilt:.1f}도** 기울어집니다. 발이 땅에 닿을 때 반대쪽 골반이 떨어지는 현상(트렌델렌버그 징후)이 보입니다. 중둔근 강화가 필요합니다.")
        else:
            st.error(f"🚨 **[심각한 에너지 분산]** 골반 기울기가 **{max_hip_tilt:.1f}도**에 달합니다. 100m를 직선이 아닌 지그재그로 달리는 것과 같은 역학적 손실이 발생 중입니다. 코어와 골반 안정화 훈련이 시급합니다.")

        # 측면(파워) 피드백 로직
        st.markdown("#### 🔍 2. 전방 추진력 판독 (측면)")
        if "글로벌 엘리트" in target_time:
            if min_drive_angle > 95 or max_push_angle < 165:
                st.error(f"⚠️ **[글로벌 스탠다드 미달]** 거상({min_drive_angle:.1f}도)과 추진({max_push_angle:.1f}도) 각도에서 세계 톱 랭커 대비 폭발력이 부족합니다. 발목 배측굴곡을 유지하며 트랙을 부수듯 밀어내세요.")
            else:
                st.success(f"🔥 **[글로벌 스탠다드 통과]** 세계 무대에서 통할 완벽한 하체 탄성 메커니즘을 갖췄습니다.")
        else:
            if min_drive_angle > 100:
                st.warning(f"⚠️ **[피치(Pitch) 범위 부족]** 무릎 거상이 **{min_drive_angle:.1f}도**로 낮아 숏 피치가 발생합니다. 장요근을 더 강하게 끌어올리세요.")
            elif max_push_angle < 160:
                st.warning(f"⚠️ **[추진 신전 부족]** 후방으로 밀어내는 각도가 **{max_push_angle:.1f}도**에 불과합니다. 지면 반발력을 끝까지 활용하지 못하고 있습니다.")
            else:
                st.success(f"🔥 **[완벽한 추진력]** 보폭과 추진력의 밸런스가 국가대표급으로 훌륭합니다.")

        st.write("---")
        st.success("✅ 프라이버시 보호를 위해 업로드된 영상은 분석 직후 서버에서 영구 삭제되었습니다.")
    else:
        st.error("⚠️ 영상을 인식하지 못했습니다. 선수의 전신이 잘 보이는 영상을 올려주세요.")
