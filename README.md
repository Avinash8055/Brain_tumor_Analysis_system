
# 🧠 Brain Tumor Detection using 3D CNN + MLP  
A deep learning model using **3D CNN + MLP** for real-time brain tumor detection with an interactive **Gradio interface**.  

---

## 🚀 **Features**  
✅ 3D CNN + MLP architecture for high accuracy  
✅ GPU-optimized for faster inference  
✅ Easy-to-use Gradio-based interface  
✅ Modular structure for easy customization  

---

## 📂 **Project Structure**  
```
brain_tumor_project/
├── models/               # Pretrained model files
├── src/
│   ├── model.py          # Model loading and inference
│   ├── gputest.py        # GPU availability test
│   ├── gradio_app.py     # Gradio interface
├── requirements.txt      # Required libraries
├── README.md
```

---

## 🛠️ **Setup Instructions**  
### 1. **Clone the Repository**  
```bash
git clone https://github.com/Avinash8055/Brain_tumor_Analysis_system.git
cd brain_tumor_project
```

---

### 2. **Install Dependencies**  
Make sure you have Python **3.10+** installed. 

Install the required packages using:  
```bash
pip install -r requirements.txt
```

---

### 3. **Check GPU Availability** *(Optional)*  
Before running the model, test if GPU is available:  
```bash
python gputest.py
```
If a GPU is detected, the output will show:  
```
CUDA is available. Running on GPU.
```

---

### 4. **Set Model Path**  
- Download the pretrained model and place it in the **models/** folder.  
- Open `model.py` and update the model path:  
```python
MODEL_PATH = "./models/brain_tumor_model.pth"
```

---
### 5. **Run the Model**
Start the model inference using:

```sh
python models.py
```

### 5. **Run the Gradio Interface**  
Start the Gradio app using:  
```bash
python gradio_app.py
```
- The interface will open in your browser at:  
👉 **http://localhost:7860**  

---


## 🧪 **Model Details**  
- **Architecture:** 3D CNN + MLP
- **Optimizer:** Adam  
- **Loss Function:** Cross-Entropy 
- **Input:** 4D NumPy array (flair, t1, t1ce, t2)
- **Classification:**  HGG vs LGG (87–94% confidence)  
- **Survival Prediction:**  9.8–84.2 months (0.1-month precision)
- **2D Output:** Matplotlib-based orthogonal slices (300 DPI)
- **3D Output:**  Plotly-based interactive rendering (600×500 pixels, 20 iso-surfaces)
- **Performance:** Optimized for CUDA with GPU support
  
---

## 🚨 **Troubleshooting**  
**Issue:** CUDA not available  
➡️ Ensure GPU drivers and CUDA are properly installed  

**Issue:** Model path not found  
➡️ Double-check the model file path in `models.py`  

---

## 👨‍💻 **Contributing**  
1. Fork the repo  
2. Create a new branch:  
```bash
git checkout -b feature/your-feature
```
3. Commit changes and push:  
```bash
git push origin feature/your-feature
```
4. Open a pull request  

---

## 📜 **License**  
This project is licensed under the **MIT License**.  

---

## ⭐ **Show Some Love!**  
If you like this project, give it a ⭐ on GitHub! 😎 




## 🌟 **Additional Info**  
If you want to train the model yourself, use the files in the **train folder**.  
 

## 🏋️‍♂️ **Training the Model**  
Follow the steps below to preprocess the data and train the model:  

---

### 1. **Convert NIfTI files to NPY**  
Collect the MRI scans in **NIfTI format** and convert them to **NPY format** for easier processing:  
```bash
python train/nifti_to_npy_converter.py --input /path/to/nifti/files --output /path/to/npy/output
```
- **Input:** NIfTI files  
- **Output:** NPY files  
- **Purpose:** Converts MRI data to a format suitable for training  

---

### 2. **Convert Clinical JSON to CSV**  
Convert clinical data from **JSON format** to **CSV format** for easy integration:  
```bash
python train/json_to_csv_clinical_converter.py --lgg_input path/to/lgg.json --hgg_input path/to/hgg.json --output_dir path/to/csv/output
```
- **Input:** JSON files for LGG and HGG  
- **Output:** CSV files  
- **Purpose:** Converts clinical data into structured format  

---

### 3. **Organize NPY Files into Patient Folders**  
Organize the converted MRI scans into patient-specific folders:  
```bash
python train/organize_mri_patient_folders.py --input_dir path/to/npy/files --output_dir path/to/organized/output
```
- **Input:** NPY files  
- **Output:** Organized folder structure  
- **Purpose:** Groups MRI files by patient ID  

---

### 4. **Standardize and Normalize MRI Scans**  
Standardize and normalize the MRI scans for consistent data input:  
```bash
python train/standardize_mri_scans.py --input_dir path/to/organized/mri --output_dir path/to/preprocessed/output
```
- **Input:** Organized NPY files  
- **Output:** Preprocessed MRI files  
- **Purpose:** Ensures data consistency and scale  

---

### 5. **Combine MRI and Clinical Data**  
Combine MRI data and clinical data for training:  
```bash
python train/combine_mri_clinical_data.py --mri_dir path/to/preprocessed/mri --clinical_dir path/to/clinical/csv --output_dir path/to/combined/output
```
- **Input:** Preprocessed MRI + Clinical CSV  
- **Output:** Combined data folder  
- **Purpose:** Merges imaging and clinical information  

---

### 6. **Train the Model**  
Run the training script to train the tumor detection model:  
```bash
python train/train_tumor_model.py --data_dir path/to/combined/data --output_dir path/to/model/output
```
- **Input:** Combined data  
- **Output:** Trained model (.pth)  
- **Purpose:** Trains the model using 3D CNN + MLP  

---

## ✅ **Training Notes**  
- Make sure to check GPU availability using `gputest.py` before training.  
- You can modify hyperparameters like learning rate and batch size directly in `train_tumor_model.py`.  
- Training time will depend on GPU availability and dataset size.  

---
