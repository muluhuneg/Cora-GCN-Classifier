import streamlit as st
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from model_definition import GCN

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Cora AI Explorer", page_icon="🕸️", layout="wide")

# Modern Professional Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { 
        background-color: #1e2130; 
        padding: 20px; 
        border-radius: 15px; 
        border: 1px solid #4e5d6c;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    .stButton>button {
        border-radius: 20px;
        background: linear-gradient(45deg, #4b6cb7 0%, #182848 100%);
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. CATEGORY MAPPING
class_names = [
    "Theory", "Genetic Algorithms", "Neural Networks", 
    "Probabilistic Methods", "Reinforcement Learning", 
    "Rule Learning", "Case-Based"
]

@st.cache_resource
def load_data_and_model():
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    model = GCN(dataset.num_features, dataset.num_classes, 16)
    model.load_state_dict(torch.load('gcn_model.pth'))
    model.eval()
    return data, model, dataset

data, model, dataset = load_data_and_model()

# 3. SIDEBAR - Beautiful Upgraded Sidebar
with st.sidebar:
    # Reliable high-quality AI Icon
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=120)
    st.title("Project Dashboard")
    st.markdown("---")
    
    st.write("### 📊 Dataset Statistics")
    st.write(f"**Nodes:** {data.num_nodes}")
    st.write(f"**Edges:** {data.num_edges}")
    
    st.markdown("---")
    st.write("### 🧠 Model Performance")
    st.metric(label="Accuracy Score", value="81.2%", delta="Competitive")
    st.success("State-of-the-art GCN Layers")

# 4. MAIN INTERFACE HEADER
st.title("🕸️ Cora Graph Neural Network Explorer")
st.write("Explore how **relational citations** and **paper content** combine to predict scientific categories.")
st.markdown("---")

# Quick Stats Row
m1, m2, m3 = st.columns(3)
m1.metric("Dataset", "Cora Citation")
m2.metric("Features", "1,433 (Words)")
m3.metric("Architecture", "PyTorch GCN")

st.markdown("---")

# 5. ANALYSIS SECTION
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📍 Node Selection")
    node_id = st.number_input("Select Paper ID to Analyze", 0, 2707, value=233)
    
    if st.button("🚀 Run Machine Inference", use_container_width=True):
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            probs = torch.softmax(out[node_id], dim=0)
            pred = probs.argmax().item()
            conf = probs[pred].item()

        st.markdown("### Result")
        st.metric("Predicted Category", class_names[pred], f"{conf:.2%} Confidence")

with col2:
    st.subheader("📈 Prediction Confidence")
    if 'probs' in locals():
        df_chart = pd.DataFrame({
            'Category': class_names,
            'Probability': probs.tolist()
        }).sort_values('Probability', ascending=True)
        
        fig = px.bar(df_chart, x='Probability', y='Category', orientation='h',
                     color='Probability', color_continuous_scale='Blues',
                     template="plotly_dark")
        
        fig.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Input a Paper ID and click the button to see the Machine's internal probability calculation.")

# 6. GRAPH CONNECTION CONTEXT (Visual Upgrade)
st.markdown("---")
st.subheader("🔍 Neighbor Network Analysis")

# Find neighbors
neighbors = data.edge_index[1][data.edge_index[0] == node_id].tolist()
num_neighbors = len(neighbors)

c1, c2 = st.columns([1, 2])
with c1:
    st.write(f"**Paper Connectivity:**")
    st.write(f"This paper (ID: {node_id}) has **{num_neighbors} citations**.")
    st.write("The GCN model uses these specific connections to 'borrow' knowledge from neighbors.")

with c2:
    if num_neighbors > 0:
        # Show a simple list of connected Paper IDs in a nice tag format
        st.write("**Connected Paper IDs:**")
        st.write(f"`{neighbors}`")
    else:
        st.warning("This paper is an isolated node (no outgoing citations).")

# 7. FOOTER
st.markdown("---")
st.caption("Developed by [Section B Group 7] | Powered by PyTorch Geometric & Streamlit")