# Road Guardian: Driver Drowsiness & Distraction Detection (Safety)

## Project Overview
Road Guardian is a driver-monitoring safety system that detects drowsiness and distraction using facial landmarks. The system tracks blink frequency (Eye Aspect Ratio / EAR) and mouth dynamics (Mouth Aspect Ratio / MAR) to detect prolonged eye closure and yawning while reducing false alarms from speech. It is designed to remain robust in low-light conditions and when drivers wear glasses.

## Problem Statement
Many road accidents occur when drivers fall asleep or look at their phones. Early detection of eye closure and yawning can trigger timely alerts and reduce crash risk.

## Research Gap
Most off-the-shelf approaches degrade in:
- **Low-light conditions** (night driving).
- **Glasses / reflective lenses** that hide eye contours.
- **Speech vs. yawning** confusion that produces false positives.

This project focuses on **robustness in low light** and **yawning vs. talking distinction**.

## Methodology
### Input
- Live video stream of the driver’s face.

### Technique
- **Dlib 68-point facial landmarks** for tracking facial geometry.
- **EAR (Eye Aspect Ratio)** to measure eye closure duration.
- **MAR (Mouth Aspect Ratio)** to detect yawning and differentiate from speech.
- **CLAHE preprocessing** to enhance contrast in low-light frames.

### Detection Logic
- **If EAR < threshold for X seconds → trigger alarm.**
- **If drowsiness is detected 3 times in 10 minutes → escalate** by sending a GPS-based alert to a family contact.

## Novelty Highlights
- Low-light robustness via CLAHE.
- Glasses-friendly detection by focusing on landmark ratios rather than raw pixel intensity.
- MAR-based yawning check to reduce false alerts while talking.
- GPS escalation on repeated drowsiness events.

## Dataset
- **YawDD** (Yawn Detection Dataset).

---

# Quick Start (Full App)

## 1) Requirements
- Python **3.10+**
- A webcam
- Dlib 68-point model:  
  `shape_predictor_68_face_landmarks.dat`  
  Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

## 2) Install
```bash
pip install -r requirements.txt
```

## 3) Run
```bash
python app.py --shape-predictor /path/to/shape_predictor_68_face_landmarks.dat
```

### Optional Flags
- `--source 0` (camera index)
- `--ear 0.23` (EAR threshold)
- `--mar 0.7` (MAR threshold)
- `--no-clahe` (disable low-light enhancement)
- `--gps-endpoint <url>` (override GPS lookup endpoint)

---

## Repository Maintenance (GitHub Sync)
```bash
git status -sb
git add .
git commit -m "Describe your update"
git push origin <your-branch>
```
