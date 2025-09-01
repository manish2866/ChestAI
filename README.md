# 🩺 ChestAI: Pneumonia Detection from Chest X-Rays  

## 📌 Project Overview  
Pneumonia is a life-threatening respiratory disease that requires timely and accurate diagnosis.  
This project leverages **Deep Learning** to classify chest X-ray images as **Normal** or **Pneumonia**.  
We developed and evaluated multiple CNN-based models, applied **GANs** for synthetic data generation,  
and used **Autoencoders** to reduce image dimensionality and speed up training.  

The final deliverable is a **Streamlit web application** that allows users to upload chest X-ray images  
and receive predictions in real time.  

---

## 🎯 Objectives  
- Develop a deep learning model to classify chest X-rays into pneumonia (+) and normal (-).  
- Improve model robustness using **GAN-based data augmentation** for imbalanced datasets.  
- Apply **Autoencoders** for dimensionality reduction and faster training without sacrificing accuracy.  
- Compare advanced CNN architectures (ResNet, DenseNet, GoogLeNet, VGG, AlexNet).  
- Provide a user-friendly interface for prediction via a **Streamlit app**.  

---

## 📊 Dataset  
- **Source**: [Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)  
- **Total Images**: 5,856 (1,583 Normal, 4,273 Pneumonia)  
- **Image Size**: 224x224 grayscale  
- **Split**: 80% Train, 10% Validation, 10% Test【50†source】【51†source】  

**Preprocessing & Augmentation**  
- Random rotation (±15°), horizontal flip, resizing, cropping.  
- Normalization of pixel intensity.  
- GAN-based synthetic image generation (1280 per class).  

---

## 🧠 Methodology  

### 1. Baseline CNN  
- 3 convolutional layers (64, 128, 256 filters).  
- Fully connected layers: 1024 → 512 → 2 outputs.  
- Accuracy: **92.84%**【50†source】.  

### 2. Improved CNN  
- Added dropout layers (0.5).  
- Early stopping to prevent overfitting.  
- Validation accuracy improved, but training time remained high.  

### 3. Advanced Architectures  
- **VGG-19**: Accuracy 75.88% (high parameters, inefficient).  
- **DenseNet**: Accuracy 92.04%, F1 ≈ 0.92.  
- **ResNet-34**: Accuracy 86.80%, robust to GAN-augmented data.  
- **GoogLeNet**: Accuracy 95.11%, best performing【50†source】.  
- **AlexNet**: Moderate performance, explored for comparison【52†source】.  

### 4. Autoencoder Preprocessing  
- Reduced images from **224×224×1 → 14×14×256** latent representation.  
- Training time reduced from ~6000s → **17.52s**.  
- Accuracy maintained at ~90.77%【52†source】.  

### 5. GAN-based Augmentation  
- Generated **2,560 synthetic images** (1280 per class).  
- Helped mitigate class imbalance.  
- ResNet improved with GAN images; DenseNet performance decreased (sensitive to GAN artifacts)【52†source】.  

---

## 📈 Evaluation Metrics  
- **Accuracy**: Overall correct predictions.  
- **Precision, Recall, F1 Score**: Balance between false positives & false negatives.  
- **ROC-AUC Curve**: Area under curve for classification robustness.  
- **Confusion Matrix**: Distribution of predicted vs actual labels.  

| Model       | Accuracy | Precision | Recall | F1 Score |  
|-------------|----------|-----------|--------|----------|  
| VGG-19      | 75.88%   | 0.93      | 0.92   | 0.92     |  
| DenseNet    | 92.04%   | 0.92      | 0.92   | 0.92     |  
| ResNet-34   | 86.80%   | 0.88      | 0.87   | 0.85     |  
| GoogLeNet   | 95.11%   | 0.95      | 0.95   | 0.95     |  
| Autoencoder+CNN | 90.77% | 0.91   | 0.90   | 0.90     |  

---

## 🛠 Running the Application  

### 1. Install Dependencies  
```bash
pip install -r requirements.txt
```  
Dependencies include:  
- `streamlit`  
- `torch`  
- `torchvision`  
- `Pillow`【49†source】  

### 2. Run the App  
```bash
streamlit run app.py
```  

### 3. Usage  
- Upload a chest X-ray image.  
- The app preprocesses the image and loads the trained model (`.pkl`).  
- Output: **Prediction** (Normal / Pneumonia).  

---

## 📌 Key Findings  
- **GoogLeNet** performed the best overall.  
- **ResNet-34** improved with GAN-augmented data.  
- **Autoencoders** drastically reduced training time while maintaining accuracy.  
- GAN augmentation helped mitigate class imbalance, but model response varied.  

---

## 📖 References  
- Shobhita Sundaram, Neha Hulkund – GAN-based Data Augmentation for Chest X-ray Classification.  
- Aggarwal et al. – Diagnostic accuracy of deep learning in medical imaging.  
- Kundu et al. – Pneumonia detection in chest X-ray images using ensembles.  
- [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)  
- Ian Goodfellow et al. – Generative Adversarial Networks (2014).  
