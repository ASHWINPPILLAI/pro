# Road Guardian: Driver Drowsiness & Distraction Detection (Safety)

## Project Overview
Road Guardian is a driver-monitoring safety system that detects drowsiness and distraction using facial landmarks. The system tracks blink frequency using **Eye Aspect Ratio (EAR)** and mouth dynamics using **Mouth Aspect Ratio (MAR)** to detect prolonged eye closure and yawning, while reducing false alarms caused by speech.

The system is designed to remain robust in **low-light conditions** and when drivers **wear glasses**.

---

## Problem Statement
Many road accidents occur due to drivers falling asleep or getting distracted (e.g., phone usage), leading to delayed reactions and loss of vehicle control. Early detection can trigger timely alerts and reduce crash risk.

---

## Research Gap
Most existing systems degrade in:
- **Low-light conditions** (night driving)
- **Glasses / reflective lenses** that hide eye contours
- **Speech vs. yawning confusion**, producing false positives

This project focuses on **robustness in low light** and **accurate yawning vs. talking distinction**.

---

## Proposed Methodology

### Input
- Live video stream of the driver’s face.

### Technique
- **Dlib 68-point facial landmarks** for facial geometry tracking
- **Eye Aspect Ratio (EAR)** for eye-closure detection
- **Mouth Aspect Ratio (MAR)** to detect yawning and distinguish from speech
- **CLAHE preprocessing** to enhance contrast in low-light frames

### Detection Logic
- If **EAR < threshold** for **X seconds** → trigger alarm
- If **drowsiness is detected 3 times within 10 minutes** → escalate by sending a **GPS-based alert** to a family contact

---

## Novelty Highlights
- Low-light robustness using **CLAHE**
- Glasses-friendly detection using **landmark ratios**
- MAR-based yawning detection to reduce false alerts
- GPS alert escalation after repeated drowsiness events

---

## Dataset
- **YawDD (Yawn Detection Dataset)**

---

## Expected Outcomes
- Reduced false positives during speech
- Reliable detection for drivers wearing glasses
- Improved night-time performance
- Faster safety response via alert escalation

---

# Quick Start (Full App)

## 1) Requirements
- Python **3.10+**
- A webcam
- Dlib 68-point model  
  `shape_predictor_68_face_landmarks.dat`  
  Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

## 2) Install
```bash
pip install -r requirements.txt
