# 🌿 Plant Disease Detection Pro



## 🚀 Overview
**Plant Disease Detection Pro** is a cutting-edge, AI-powered agricultural tool designed to help farmers and gardeners identify plant diseases instantly. By leveraging Deep Learning and computer vision, the app provides accurate classifications, treatment recommendations, and connects users with local agricultural experts.

## ✨ Key Features
- **📸 Multi-Mode Capture**: Upload leaf images from your gallery or use the real-time camera interface.
- **🧠 High-Precision AI**: Powered by a custom-trained **MobileNetV2** model for fast and accurate detection.
- **🩺 Treatment Advice**: Get instant prevention and cure suggestions for detected diseases.
- **🕒 Scan History**: Keep track of your recent scans directly in the sidebar for easy comparison.
- **👨‍⚕️ Expert Directory**: Find and contact plant health experts in your specific city or town.
- **🌡️ Smart Context**: Built-in environmental monitoring (weather & humidity) to assess disease risk factors.

## 🛠️ Technology Stack
- **Frontend**: [Streamlit](https://streamlit.io/) (Premium custom CSS styling)
- **Deep Learning**: [TensorFlow](https://www.tensorflow.org/) & [Keras](https://keras.io/)
- **Model Architecture**: MobileNetV2 (Transfer Learning)
- **Data Handling**: Pandas & NumPy
- **Image Processing**: Pillow (PIL)

## 📦 Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/VedantVH/Plant-Disease-Detection.git
   cd Plant-Disease-Detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run streamlit_app.py
   ```

## 🧠 Model Training
The model was trained using the **PlantVillage Dataset**, focusing on crops like Tomatoes, Potatoes, and Peppers. We utilized **Transfer Learning** with MobileNetV2 to achieve high efficiency and accuracy.
- **Input Size**: 224x224 RGB images
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request to help improve the accuracy or features of this project.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed with ❤️ to support global food security and sustainable farming.*