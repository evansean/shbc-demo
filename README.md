# SHBC 2025 Demo App: DME and vCDR

This demo application was developed for the **Singapore Health and Biomedical Congress (SHBC) 2025** exhibition booth.  
It showcases two AI projects side by side:

- **DME (Diabetic Macular Edema)** – performs calculation and prediction using our trained model.  
- **vCDR (Vertical Cup-to-Disc Ratio)** – estimates the ratio between the optic cup and optic disc boundaries in fundus images. This measurement is an important indicator for glaucoma detection. The system uses deep learning models trained and validated on both standard fundus images and ultra-widefield or smartphone-based images to provide accurate, generalizable results.

---

## 🚀 Running the App

1. Create a new conda environment:
   ```bash
   conda create -n shbc python=3.10
   conda activate shbc
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Run the app:
   ```bash
   python app.py
Open the link shown in your terminal in your browser.

## 📸 Demo Screenshot

<img width="1600" height="999" alt="image" src="https://github.com/user-attachments/assets/485478ab-83c2-430d-96ca-1dbded7b425f" />


*Left: DME prediction panel | Right: vCDR estimation panel*


