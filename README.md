# Road Guardian: Driver Drowsiness and Distraction Detection (Safety)

## Topic
Driver drowsiness and distraction detection using facial landmarks and blink frequency analysis.

## Problem Statement
Road accidents are frequently caused by drivers falling asleep or looking at their phones, leading to delayed reaction times and loss of vehicle control.

## Research Gap
Many existing systems fail when drivers wear glasses or when detection occurs at night. This project emphasizes **robustness in low-light conditions** and explores **distinguishing yawning vs. talking** to reduce false alarms.

## Proposed Methodology
### Input
* Live video stream of the driver’s face.

### Technique
* Use **Dlib** with 68 facial landmarks to track:
  * **Eye Aspect Ratio (EAR)** for blink and eye-closure detection.
  * **Mouth Aspect Ratio (MAR)** to detect yawning and reduce confusion with speech.

### Logic
* If **EAR < threshold** for **x seconds** → trigger an alarm.
* If **drowsiness is detected 3 times within 10 minutes**, escalate by sending a GPS location alert to a family member.

## Novelty
* Focus on low-light robustness and glasses-friendly detection.
* Incorporate a **yawning vs. talking** classifier to reduce false positives.
* Add a **GPS-based alert escalation** after repeated drowsiness events.

## Dataset
* **YawDD** (Yawn Detection Dataset).

## Expected Outcomes
* Reduced false positives for drivers wearing glasses or speaking.
* Reliable detection under low-light conditions.
* Timely alerts and escalation for improved safety response.

---

# Running the Demo App

## Prerequisites
1. Python 3.10+ recommended.
2. Dlib 68-point facial landmark model: `shape_predictor_68_face_landmarks.dat`.
   * Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   * Unzip the file so you have `shape_predictor_68_face_landmarks.dat`.

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Run
```bash
python app.py --shape-predictor /path/to/shape_predictor_68_face_landmarks.dat
```

Optional flags:
* `--source 0` to change the camera index.
* `--ear 0.23` and `--mar 0.7` to tune thresholds.
* `--no-clahe` to disable low-light contrast enhancement.

## Notes
* The demo uses **CLAHE** to boost low-light robustness before face detection.
* If the system detects 3 drowsiness events within 10 minutes, it prints a GPS-based alert (via IP lookup).

## Keeping It Updated on GitHub
Use the following workflow to save changes and keep the repository updated:

```bash
git status -sb
git add .
git commit -m "Describe your update"
git push origin <your-branch>
```
