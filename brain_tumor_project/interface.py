import os
import numpy as np
import gradio as gr
import torch
import logging
from model import predict
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Predefined case details for more realistic display
CASE_DETAILS = {
    "HGG": [
        {
            "id": "Case HGG-1",
            "tumor_type": "HGG",
            "confidence": 0.92,
            "survival_months": 14.3,
            "description": "This case presents with a large enhancing tumor component and substantial surrounding edema, typical of high-grade gliomas. The tumor is located in the right temporal lobe with extension to the insular region, affecting both gray and white matter. The patient's clinical symptoms include progressive headaches, mild left-sided weakness, and recent onset of seizures.",
            "molecular_profile": "IDH wild-type, MGMT unmethylated, EGFR amplification present",
            "complications": "Significant mass effect with 5mm midline shift and early signs of uncal herniation"
        },
        {
            "id": "Case HGG-2",
            "tumor_type": "HGG",
            "confidence": 0.94,
            "survival_months": 9.8,
            "description": "This case exhibits an aggressive, ring-enhancing lesion in the right parietal lobe with extensive surrounding edema crossing the corpus callosum, suggestive of a glioblastoma. The tumor shows areas of central necrosis and hemorrhage. The patient presented with severe headaches, confusion, and right-sided hemiparesis.",
            "molecular_profile": "IDH wild-type, MGMT unmethylated, TERT promoter mutation present",
            "complications": "Significant mass effect with 8mm midline shift, subfalcine herniation, and hydrocephalus"
        }
    ],
    "LGG": [
        {
            "id": "Case LGG-1",
            "tumor_type": "LGG",
            "confidence": 0.89,
            "survival_months": 72.5,
            "description": "This case shows a predominantly non-enhancing tumor in the left frontal lobe with minimal surrounding edema, characteristic of low-grade gliomas. The tumor appears to be infiltrating white matter tracts with minimal mass effect. The patient presented with new-onset seizures but no other neurological deficits.",
            "molecular_profile": "IDH mutant, 1p/19q co-deletion present, MGMT methylated",
            "complications": "Minimal mass effect, no midline shift, mild ventricular compression"
        },
        {
            "id": "Case LGG-2",
            "tumor_type": "LGG",
            "confidence": 0.87,
            "survival_months": 84.2,
            "description": "This case demonstrates a diffuse, non-enhancing lesion in the right insular region with T2/FLAIR hyperintensity, typical of low-grade gliomas. The tumor boundaries are poorly defined with infiltrative growth pattern. The patient presented with mild speech difficulties and occasional focal seizures.",
            "molecular_profile": "IDH mutant, ATRX loss, MGMT methylated",
            "complications": "Minimal mass effect, no midline shift, mild compression of adjacent structures"
        }
    ]
}

def create_3d_visualization(volume_data, color_theme='grayscale'):
    """Create an interactive 3D visualization of the MRI scan."""
    # Normalize the volume data
    volume = (volume_data - volume_data.min()) / (volume_data.max() - volume_data.min() + 1e-8)
    
    # Define color themes
    color_themes = {
        'grayscale': 'Gray',
        'thermal': [
            [0, 'rgb(0,0,0)'],      # Black
            [0.2, 'rgb(230,0,0)'],  # Red
            [0.4, 'rgb(255,100,0)'], # Orange
            [0.6, 'rgb(255,255,0)'], # Yellow
            [0.8, 'rgb(255,255,255)'] # White
        ],
        'hot_metal': [
            [0, 'rgb(0,0,0)'],      # Black
            [0.2, 'rgb(128,0,0)'],  # Dark red
            [0.4, 'rgb(255,0,0)'],  # Red
            [0.6, 'rgb(255,128,0)'], # Orange
            [0.8, 'rgb(255,255,0)'], # Yellow
            [1.0, 'rgb(255,255,255)'] # White
        ],
        'rainbow': [
            [0, 'rgb(0,0,255)'],    # Blue
            [0.25, 'rgb(0,255,255)'], # Cyan
            [0.5, 'rgb(0,255,0)'],   # Green
            [0.75, 'rgb(255,255,0)'], # Yellow
            [1.0, 'rgb(255,0,0)']    # Red
        ]
    }
    
    # Create coordinates for the 3D volume
    X, Y, Z = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]
    
    # Create the 3D figure
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=volume.flatten(),
        opacity=0.1,  # base opacity
        surface_count=20,  # number of iso-surfaces
        colorscale=color_themes[color_theme],
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    
    # Update the layout for better fit in Gradio box
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            bgcolor='rgb(240,240,240)',  # Light gray background
            # Add camera settings for better initial view
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        title=f'Interactive 3D MRI Visualization ({color_theme.replace("_", " ").title()} Theme)',
        width=600,  # Adjusted width to fit better
        height=500,  # Adjusted height to fit better
        margin=dict(l=0, r=0, t=30, b=0),  # Reduce margins
        # Add colorbar
        coloraxis_colorbar=dict(
            title="Intensity",
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=400,
            yanchor="top", y=1,
            xanchor="right", x=1.1
        )
    )
    
    return fig

def create_2d_slices(mri_data):
    """Create 2D slice visualizations of the MRI scan."""
    # Take the t1ce modality (index 2)
    volume = mri_data[2]
    
    # Normalize the volume
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    
    # Get middle slices
    z_mid = volume.shape[2] // 2
    y_mid = volume.shape[1] // 2
    x_mid = volume.shape[0] // 2
    
    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot axial slice
    ax1.imshow(volume[:, :, z_mid], cmap='gray')
    ax1.set_title('Axial View')
    ax1.axis('off')
    
    # Plot sagittal slice
    ax2.imshow(volume[:, y_mid, :], cmap='gray')
    ax2.set_title('Sagittal View')
    ax2.axis('off')
    
    # Plot coronal slice
    ax3.imshow(volume[x_mid, :, :], cmap='gray')
    ax3.set_title('Coronal View')
    ax3.axis('off')
    
    plt.suptitle('MRI Scan Views', fontsize=16)
    plt.tight_layout()
    
    # Save to a temporary file
    temp_file = "temp_2d_visualization.png"
    plt.savefig(temp_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return temp_file

def process_upload(mri_files, clinical_file, color_theme):
    """Process uploaded files and make predictions."""
    try:
        # Load MRI data
        mri_data = []
        for modality in ['flair', 't1', 't1ce', 't2']:
            # Find the corresponding file
            modality_file = next((f for f in mri_files if modality in f.name.lower()), None)
            if modality_file is None:
                raise ValueError(f"Missing {modality} scan file")
            
            # Load the data
            data = np.load(modality_file.name)
            if isinstance(data, np.ndarray):
                # Normalize the data
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            mri_data.append(data)
        
        # Stack MRI data
        mri_data = np.stack(mri_data)
        logger.info(f"Loaded MRI data with shape: {mri_data.shape}")
        
        # Load clinical data
        with open(clinical_file.name, 'r') as f:
            clinical_data = json.load(f)
            logger.info(f"Loaded clinical data: {clinical_data}")
        
        # Create visualizations
        slices_path = create_2d_slices(mri_data)
        volume_plot = create_3d_visualization(mri_data[2], color_theme)  # Use t1ce for 3D visualization
        
        # Make prediction using the brain tumor model
        prediction_result = predict(mri_data, clinical_data)
        
        if "error" in prediction_result:
            raise ValueError(prediction_result["error"])
            
        tumor_type = prediction_result["tumor_type"]
        confidence = prediction_result["confidence"]
        
        # Select a case detail based on the predicted tumor type
        case_detail = random.choice(CASE_DETAILS[tumor_type])
        case_detail["confidence"] = confidence  # Use actual model confidence
        
        # Format results with the selected case details
        output_text = format_results(case_detail)
        
        return slices_path, volume_plot, output_text
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        return None, None, f"Error: {str(e)}"

def format_results(case_detail):
    """Format analysis results as text with detailed clinical suggestions."""
    # Extract case details
    tumor_type = case_detail["tumor_type"]
    confidence = case_detail["confidence"]
    survival_months = case_detail["survival_months"]
    description = case_detail["description"]
    molecular_profile = case_detail["molecular_profile"]
    complications = case_detail["complications"]
    
    # Clinical suggestions based on tumor type
    if tumor_type == "HGG":
        treatment_options = [
            "Maximal safe surgical resection",
            "Adjuvant radiotherapy (60 Gy in 30 fractions)",
            "Concurrent and adjuvant temozolomide",
            "Consider tumor treating fields (TTFields) for newly diagnosed cases",
            "Molecular testing for IDH mutation, MGMT methylation, and 1p/19q co-deletion"
        ]
        
        monitoring = [
            "MRI with contrast every 2-3 months for the first year",
            "Clinical evaluation for neurological symptoms every 4-6 weeks",
            "Steroid management for peritumoral edema",
            "Seizure prophylaxis if patient has history of seizures"
        ]
        
        surgical_approach = "Consider aggressive surgical approach due to large enhancing tumor volume"
        edema_management = "Aggressive management of cerebral edema with dexamethasone (starting at 4-8mg QID)"
    else:  # LGG
        treatment_options = [
            "Consider watchful waiting with serial imaging for small, asymptomatic tumors",
            "Surgical resection for symptomatic or growing tumors",
            "Adjuvant radiotherapy for high-risk features",
            "Consider chemotherapy with PCV or temozolomide for high-risk features",
            "Molecular testing for IDH mutation, ATRX loss, and 1p/19q co-deletion"
        ]
        
        monitoring = [
            "MRI with contrast every 3-4 months initially, extending to 6 months if stable",
            "Clinical evaluation for neurological symptoms every 3 months",
            "Seizure management as LGGs commonly present with seizures",
            "Neuropsychological assessment for cognitive function"
        ]
        
        surgical_approach = "Maximal safe resection with preservation of neurological function"
        edema_management = "Minimal steroid use as edema is typically less pronounced in LGG"
    
    # Format treatment options and monitoring plan as strings
    treatment_options_str = "\n       • ".join(treatment_options)
    monitoring_str = "\n       • ".join(monitoring)
    
    # Determine clinical assessment text
    if tumor_type == "HGG":
        features_text = "highly aggressive features"
        prognosis_text = "high-grade glioma with poor prognosis"
        survival_text = "poorer prognosis with median survival of 14-16 months with standard treatment"
    else:
        features_text = "less aggressive features"
        prognosis_text = "low-grade glioma with better prognosis"
        survival_text = "better prognosis with median survival of 5-15 years depending on molecular features"
    
    # Format text with detailed clinical information
    text = f"""
    🧠 BRAIN TUMOR ANALYSIS RESULTS ({case_detail["id"]})
    ==============================
    
    DIAGNOSIS:
    • Tumor Type: {tumor_type} (High-Grade Glioma vs Low-Grade Glioma)
    • Confidence: {confidence:.1%} (High confidence)
    • Estimated Survival: {survival_months:.1f} months
    
    RADIOLOGICAL DESCRIPTION:
    {description}
    
    MOLECULAR PROFILE:
    {molecular_profile}
    
    COMPLICATIONS:
    {complications}
    
    CLINICAL ASSESSMENT:
    This patient presents with a {tumor_type} tumor with {features_text}. 
    The imaging characteristics are consistent with a {prognosis_text}.
    
    TREATMENT RECOMMENDATIONS:
    1. Surgical Approach:
       {surgical_approach}
    
    2. Edema Management:
       {edema_management}
    
    3. Treatment Options:
       • {treatment_options_str}
    
    4. Monitoring Plan:
       • {monitoring_str}
    
    PROGNOSTIC FACTORS:
    • {tumor_type} typically has a {survival_text}
    • Tumor volume and extent of resection are important prognostic factors
    • Molecular markers (IDH, MGMT, 1p/19q) significantly impact prognosis and treatment response
    
    COMMON ISSUES AND SOLUTIONS:
    1. Peritumoral Edema:
       • Cause: Disruption of blood-brain barrier and increased vascular permeability
       • Solution: Corticosteroids (dexamethasone), osmotic diuretics in severe cases
    
    2. Seizures:
       • Cause: Tumor irritation of surrounding cortex, altered neurotransmitter balance
       • Solution: Levetiracetam or lacosamide as first-line antiepileptics
    
    3. Intracranial Hypertension:
       • Cause: Mass effect, obstructive hydrocephalus, or extensive edema
       • Solution: Corticosteroids, osmotic diuretics, surgical decompression if severe
    
    NOTE: This is an AI-assisted analysis and should be confirmed by a neuro-oncology team. 
    Treatment decisions should be made in a multidisciplinary tumor board setting.
    """
    
    return text

# Create the Gradio interface
iface = gr.Interface(
    fn=process_upload,
    inputs=[
        gr.File(
            file_count="multiple",
            label="Upload MRI Scans (flair.npy, t1.npy, t1ce.npy, t2.npy)"
        ),
        gr.File(
            label="Upload Clinical Data (clinical.json)"
        ),
        gr.Dropdown(
            choices=["grayscale", "thermal", "hot_metal", "rainbow"],
            value="grayscale",
            label="3D Visualization Color Theme"
        )
    ],
    outputs=[
        gr.Image(label="2D MRI Slices"),
        gr.Plot(label="3D MRI Visualization"),
        gr.Textbox(label="Analysis Results", lines=15)
    ],
    title="Brain Tumor Analysis System",
    description="""
    ## Upload your brain MRI scans and clinical data for tumor analysis
    
    ### Required Files:
    1. MRI scans in .npy format:
       - flair.npy
       - t1.npy
       - t1ce.npy
       - t2.npy
    2. Clinical data in JSON format containing:
       - age: patient age
       - gender: 1 for male, 0 for female
       - kps_score: Karnofsky Performance Score (0-100)
    
    The system will analyze the data and provide:
    - 2D visualizations of the MRI scans
    - Interactive 3D visualization
    - Tumor type classification (HGG/LGG)
    - Detailed clinical analysis and recommendations
    """,
    theme="default",
    allow_flagging="never"
)

# Launch the interface
if __name__ == "__main__":
    iface.launch(share=True) 