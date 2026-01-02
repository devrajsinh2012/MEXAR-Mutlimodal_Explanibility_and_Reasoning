# import streamlit as st
# import torch
# import torch.nn as nn
# import pandas as pd
# import pickle
# import numpy as np

# # 1. THIS MUST BE THE FIRST STREAMLIT COMMAND
# st.set_page_config(page_title="Mexar Nano: SympScan", page_icon="🧬", layout="wide")

# # 2. Define the CORRECT Model Architecture (128 hidden units)
# class MexarStudent(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(MexarStudent, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 128), 
#             nn.ReLU(),
#             nn.Linear(128, 64),        
#             nn.ReLU(),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, x):
#         return self.net(x)

# # 3. Resource Loading (Moved AFTER set_page_config)
# @st.cache_resource
# def load_resources():
#     # Load metadata
#     with open('mexar_metadata.pkl', 'rb') as f:
#         metadata = pickle.load(f)
    
#     # Initialize and load model weights
#     input_dim = metadata['input_dim']
#     num_classes = metadata['num_classes']
#     model = MexarStudent(input_dim, num_classes)
    
#     # Load the .pth file
#     state_dict = torch.load('mexar_nano_student.pth', map_location=torch.device('cpu'))
#     model.load_state_dict(state_dict)
#     model.eval()
    
#     # Load recommendation datasets
#     # Note: Using 'utf-8-sig' or 'latin1' if your CSVs have special characters
#     datasets = {
#         "desc": pd.read_csv('description.csv'),
#         "prec": pd.read_csv('precautions.csv'),
#         "meds": pd.read_csv('medications.csv'),
#         "diet": pd.read_csv('diets.csv'),
#         "work": pd.read_csv('workout.csv')
#     }
#     return model, metadata, datasets

# # Load everything
# model, metadata, db = load_resources()

# # 4. UI Layout
# st.title("🧬 Mexar Nano: SympScan")
# st.subheader("Intelligent Disease Prediction & Recovery Guide")

# # Create a sidebar for info
# st.sidebar.header("About")
# st.sidebar.info("This system uses a Distilled Nano Model to analyze symptoms and provide health suggestions.")

# # Symptom Selection
# selected_symptoms = st.multiselect(
#     "Select your symptoms:", 
#     options=metadata['symptom_names'],
#     help="You can type to search for symptoms"
# )

# if st.button("Generate Health Report"):
#     if not selected_symptoms:
#         st.error("Please select at least one symptom to continue.")
#     else:
#         # Prepare input vector
#         input_vector = np.zeros(metadata['input_dim'])
#         for s in selected_symptoms:
#             if s in metadata['symptom_names']:
#                 idx = metadata['symptom_names'].index(s)
#                 input_vector[idx] = 1
        
#         # Predict
#         input_tensor = torch.FloatTensor(input_vector).unsqueeze(0)
#         with torch.no_grad():
#             output = model(input_tensor)
#             _, pred_idx = torch.max(output, 1)
#             predicted_disease = metadata['disease_names'][pred_idx.item()]

#         # Display Result
#         st.success(f"### Predicted Condition: **{predicted_disease}**")
#         st.divider()
        
#         # Display Details in columns
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("#### ℹ️ About the Condition")
#             desc = db['desc'][db['desc']['Disease'] == predicted_disease]['Description'].values
#             st.write(desc[0] if len(desc) > 0 else "Description not found.")
            
#             st.markdown("#### ⚠️ Safety Precautions")
#             prec = db['prec'][db['prec']['Disease'] == predicted_disease].iloc[:, 1:].values
#             if len(prec) > 0:
#                 for p in prec[0]:
#                     if pd.notna(p) and str(p).strip() != "":
#                         st.write(f"• {p}")
#             else:
#                 st.write("No specific precautions listed.")

#         with col2:
#             st.markdown("#### 💊 Possible Medications")
#             meds = db['meds'][db['meds']['Disease'] == predicted_disease]['Medication'].values
#             if len(meds) > 0:
#                 # Cleaning up list-string formatting if present
#                 st.write(meds[0].replace("[", "").replace("]", "").replace("'", ""))
#             else:
#                 st.write("Consult a doctor for medication.")

#             st.markdown("#### 🥗 Diet & 🏃 Workout")
#             diet = db['diet'][db['diet']['Disease'] == predicted_disease]['Diet'].values
#             st.write(f"**Diet:** {diet[0]}" if len(diet) > 0 else "Standard balanced diet.")
            
#             work = db['work'][db['work']['Disease'] == predicted_disease]['Workouts'].values
#             st.write(f"**Exercise:** {work[0]}" if len(work) > 0 else "Rest as needed.")

# st.warning("⚠️ **Medical Disclaimer:** This tool is for educational purposes. Always consult a healthcare professional for diagnosis.")











import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import pickle
import numpy as np
import ast

# 1. Page Config
st.set_page_config(page_title="Mexar Nano: Smart Health", page_icon="🩺", layout="wide")

# 2. Model Architecture
class MexarStudent(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MexarStudent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),        
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)

# 3. Resource Loading
@st.cache_resource
def load_resources():
    with open('mexar_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    model = MexarStudent(metadata['input_dim'], metadata['num_classes'])
    model.load_state_dict(torch.load('mexar_nano_student.pth', map_location=torch.device('cpu')))
    model.eval()
    
    # Load CSVs and standardize Disease column names to lowercase for perfect matching
    def read_and_standardize(file):
        df = pd.read_csv(file)
        # Find the column that represents the Disease name
        potential_cols = ['Disease', 'diseases', 'disease']
        actual_col = next((c for c in potential_cols if c in df.columns), df.columns[0])
        df['match_key'] = df[actual_col].astype(str).str.lower().str.strip()
        return df

    datasets = {
        "desc": read_and_standardize('description.csv'),
        "prec": read_and_standardize('precautions.csv'),
        "meds": read_and_standardize('medications.csv'),
        "diet": read_and_standardize('diets.csv'),
        "work": read_and_standardize('workout.csv')
    }
    return model, metadata, datasets

model, metadata, db = load_resources()

# 4. Logic Functions
def get_recommendations(disease_name):
    key = disease_name.lower().strip()
    results = {}
    for name, df in db.items():
        row = df[df['match_key'] == key]
        results[name] = row.iloc[0] if not row.empty else None
    return results

def parse_list_string(val):
    if pd.isna(val): return []
    try:
        # Try to convert "['a', 'b']" string to actual list
        return ast.literal_eval(val)
    except:
        # If it's just a normal string, return it as a single-item list
        return [val]

# 5. UI Layout
st.title("🧬 Mexar Nano: Diagnostic Assistant")
user_input = st.text_area("Describe your symptoms in detail:", placeholder="e.g. I am experiencing chest pain and shortness of breath...")

if st.button("Analyze Statement"):
    if len(user_input) < 10:
        st.warning("Please provide more information.")
    else:
        # Symptom Extraction
        input_text = user_input.lower()
        detected = [s for s in metadata['symptom_names'] if s.lower().replace("_", " ") in input_text]
        
        if not detected:
            st.error("Could not identify specific symptoms. Try being more descriptive.")
        else:
            # Prediction
            vec = np.zeros(metadata['input_dim'])
            for s in detected: vec[metadata['symptom_names'].index(s)] = 1
            with torch.no_grad():
                output = model(torch.FloatTensor(vec).unsqueeze(0))
                pred_disease = metadata['disease_names'][torch.max(output, 1)[1].item()]

            # Display
            st.success(f"### Predicted Condition: {pred_disease}")
            st.info(f"**Identified Symptoms:** {', '.join(detected)}")
            st.divider()

            res = get_recommendations(pred_disease)
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📖 Description")
                if res['desc'] is not None:
                    st.write(res['desc']['Description'])
                
                st.subheader("🛡️ Precautions")
                if res['prec'] is not None:
                    # Precautions are usually in separate columns (Precaution_1, etc.)
                    p_cols = [c for c in res['prec'].index if 'Precaution' in c]
                    for c in p_cols:
                        val = res['prec'][c]
                        if pd.notna(val) and str(val).strip(): st.write(f"• {val}")

            with col2:
                st.subheader("🥗 Recommended Diet")
                if res['diet'] is not None:
                    items = parse_list_string(res['diet']['Diet'])
                    for i in items: st.write(f"✅ {i}")

                st.subheader("🏃 Suggested Workouts")
                if res['work'] is not None:
                    # Finds the workout column regardless of "Workout" or "Workouts" name
                    w_col = 'Workouts' if 'Workouts' in res['work'].index else 'Workout'
                    items = parse_list_string(res['work'][w_col])
                    for i in items: st.write(f"💪 {i}")

st.markdown("---")
st.caption("Disclaimer: This tool provides AI-generated suggestions based on trained data. Consult a doctor for medical diagnosis.")