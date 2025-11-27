# CV_GAI_Pothole_Detection

# Problem Definition

Urban road networks suffer from potholes that degrade ride quality, increase vehicle maintenance costs, and pose safety risks.
Manual inspection and fragmented reporting channels delay remediation.

This project presents an end-to-end pothole detection and reporting system that:

1) Detects potholes in images and videos.

2) Selects a high-accuracy model through empirical evaluation.

3) Automatically generates and sends formal complaint letters to local authorities based on geolocation data significantly reducing reporting time.
   

# Objectives

1) Develop a robust pothole detection pipeline for images and videos with high precision and recall.

2) Enhance dataset diversity using GAN-based synthetic data augmentation and external datasets.

3) Compare and fine-tune YOLOv8 and Retrieval-Driven Transformers (RETDTR) to select the best model.

4) Create a user-facing Streamlit interface that:

*Accepts image/video uploads.

* Performs detection.

* Generates and formats a formal complaint letter using LLM-based automation.

# Methodology / Architecture
* Data Pipeline

1) Raw Image Collection (~170 images, in-house).

2) Image Cleaning & Annotation.

3) Synthetic Augmentation using GANs for diversity and class balance.

4) External Dataset Integration (Roboflow, ~9,000 images) for pretraining and transfer learning.

* Model Development

1) Baseline: Train and tune YOLOv8 on the in-house GAN-augmented dataset.

2) Transfer Learning: Fine-tune the best YOLOv8 model on Roboflow data.

3) Transformer Comparison: Train a RETDTR model for benchmarking.

4) Model Selection: Based on performance, latency, and model size.

* Video Processing

1) Extract frames from video.

2) Run per-frame detection.

3) Apply temporal smoothing/tracking to reduce flicker and false positives.

* Post-Processing & Decision Logic

1) Confidence thresholding, Non-Max Suppression, and IoU filtering.

2) Temporal aggregation for video sequences.

3) Shadow rejection and wet-surface detection heuristics.

* Report Generation & Delivery

1) Extract bounding box, timestamp, and geolocation (from device or manual input).

2) Generate formal complaint letters using LLMs (Ollama 8B, Gemini Flash 2.5, Mistral).

3) Select or fuse best outputs.

4) Deliver reports via:

* Email

1) Municipal API

2) Downloadable PDF/Word file

# Experimental Setup & Evaluation
* Datasets

1) In-house test set: Held out from 170 + GAN-augmented data.

2) Cross-dataset validation: Using unseen Roboflow subsets.

3) Video test sequences: Various lighting, motion, and angle conditions.

* Metrics

1) Detection Performance:

mAP@0.5, mAP@[0.5:0.95], Precision, Recall, F1-score, IoU, False Positive Rate

2) Operational Efficiency:

Inference latency (CPU/GPU/edge device), model size, memory usage

3) Robustness

AUC under different environments (day/night, wet/dry), cross-dataset mAP drop

4) End-to-End Performance

* Letter generation accuracy (presence of required fields)

* Human review quality scores

* Successful report delivery rate

# Results
<img width="520" height="520" alt="image" src="https://github.com/user-attachments/assets/7b81cfec-c3d3-4541-b8b8-d62b6ce3f0c1" />

Fig 1	Pothole detection using YOLOv8 on online dataset
<img width="572" height="572" alt="image" src="https://github.com/user-attachments/assets/5f8eb164-fabd-45d5-981b-74fcc474bcde" />

Fig 2	Validation on online dataset using YOLOv8
<img width="589" height="589" alt="image" src="https://github.com/user-attachments/assets/46fe4c37-3474-4582-a20d-cd1f48184290" />

Fig 3	Validation using RETDTR
<img width="579" height="579" alt="image" src="https://github.com/user-attachments/assets/63dad75f-ca78-46c2-b155-7702c0f2f1ac" />

Fig 4	Pothole detection from video 1
<img width="532" height="532" alt="image" src="https://github.com/user-attachments/assets/5c291a4c-12c5-4d7f-930a-c444f9f9fb07" />

Fig 5	Pothole detection from video 2
<img width="688" height="703" alt="image" src="https://github.com/user-attachments/assets/84490135-87cd-41a6-a67e-9989cf67c166" />

Fig 6	Detection on Streamlit app using base YOLO model

![Uploading image.png…]()

Fig 7	Generation of formal complaint letter using Gemini Flash 2.5 (Google API)

# Tech Stack

* Python, PyTorch, YOLOv8, RETDTR

* Streamlit (UI)

* GANs (Data Augmentation)

* Google API (Gemini Flash 2.5) – LLM-powered letter generation

* Ollama 8B, Mistral – Alternative LLMs

# Future Enhancements

* Real-time edge deployment on low-power devices.

* Integration with municipal complaint dashboards.

* Expansion to other road hazards (cracks, waterlogging, debris).




   
