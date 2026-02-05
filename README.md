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
