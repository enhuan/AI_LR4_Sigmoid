import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Neural Network Intelligence: Sigmoid App",
    page_icon="🧠",
    layout="wide"
)

# =============================
# Sidebar: Parameter Control
# =============================
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("### 1. Visualization Settings")
x_min = st.sidebar.slider("Min x value", -10.0, 0.0, -5.0)
x_max = st.sidebar.slider("Max x value", 0.0, 10.0, 5.0)
num_points = st.sidebar.slider("Number of points", 10, 500, 100)

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Model Complexity")
neurons = st.sidebar.slider("Hidden Layer Neurons", 1, 50, 15)
epochs = st.sidebar.slider("Epochs", 50, 800, 300)
learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.001, 0.01, 0.05, 0.1], value=0.01)

# =============================
# Header & Introduction
# =============================
st.title("Neural Network Architecture: Sigmoid Activation")
st.write(
    """
    This application explores the **Sigmoid** activation function, a foundational element in 
    logistic regression and probability modeling. We demonstrate how its characteristic S-curve 
    enables a neural network to solve non-linear problems like **logistic growth**, where data 
    reaches a saturation point or carrying capacity.
    """
)

# =============================
# Section 1: Interactive Sigmoid Visualization
# =============================
st.markdown("---")
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📈 Interactive Sigmoid Curve")
    x_val = np.linspace(x_min, x_max, num_points)
    # Sigmoid formula
    y_val = 1 / (1 + np.exp(-x_val))

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(x_val, y_val, color='#2ca02c', linewidth=2, label="Sigmoid(x)")
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel("Input", fontsize=7)
    ax.set_ylabel("Output", fontsize=7)
    ax.tick_params(labelsize=6)
    st.pyplot(fig, use_container_width=False)

with col2:
    st.subheader("⚙️ Mathematical Framework")
    st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")
    st.info("""
        **Core Technical Advantages:**
        * **Probability Mapping:** Constrains output values between 0 and 1, making it ideal for normalization and binary classification.
        * **Continuous Differentiability:** Provides a smooth gradient that is essential for gradient-based optimization.
        * **Saturation Modeling:** Naturally represents systems that reach a fixed limit or steady state.
        * **Biological Analogy:** Historically inspired by the "firing rate" of biological neurons.
        """)

# =============================
# Section 2: Data-Driven Neural Network Model (Logistic Growth)
# =============================
st.markdown("---")
st.subheader("🤖 Live Model Training: Solving Logistic Growth")
st.write("Below, we generate data representing an **S-Curve (Logistic Growth)**. This simulates biological populations or market adoption.")

# 1. Generate Logistic/S-Curve Data
X_np = np.linspace(-6, 6, 50).reshape(-1, 1)
y_np = 1 / (1 + np.exp(-1.5 * (X_np - 0.5))) + np.random.normal(0, 0.05, X_np.shape)

X_torch = torch.from_numpy(X_np).float()
y_torch = torch.from_numpy(y_np).float()

# 2. Define Model Structure
model = nn.Sequential(
    nn.Linear(1, neurons),
    nn.Sigmoid(), # Core focused component
    nn.Linear(neurons, 1)
)

# 3. Training and Result Plotting
if st.button('🚀 Execute Training Process'):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    progress_bar = st.progress(0)
    for epoch in range(epochs):
        optimizer.zero_grad()
        prediction = model(X_torch)
        loss = criterion(prediction, y_torch)
        loss.backward()
        optimizer.step()
        progress_bar.progress((epoch + 1) / epochs)

    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        fig_res, ax_res = plt.subplots(figsize=(3, 2))
        ax_res.scatter(X_np, y_np, s=8, color='gray', alpha=0.6, label='Noisy S-Curve')
        with torch.no_grad():
            y_pred = model(X_torch).numpy()
        ax_res.plot(X_np, y_pred, color='red', linewidth=1.5, label='Sigmoid-NN Fit')
        ax_res.set_title(f"Fit achieved using {neurons} Neurons", fontsize=8)
        ax_res.tick_params(labelsize=6)
        ax_res.legend(prop={'size': 5})
        st.pyplot(fig_res, use_container_width=False)

    with res_col2:
        st.success(f"Training Complete!")
        st.metric("Final Training Loss (MSE)", f"{loss.item():.4f}")

        st.info("""
        **Understanding the Training Process:**
        * **Model Complexity:** Each neuron represents a Sigmoid unit. The network combines these S-curves to approximate the non-linear logistic growth of the data.
        * **Epochs:** Full passes of the dataset. Because Sigmoid has smaller gradients than ReLU, it often requires more epochs to converge on a smooth fit.
        * **Learning Rate:** Controls the adjustment steps. A precise rate is vital for Sigmoid to avoid 'saturation,' where the model stops learning because gradients are too close to zero.
        * **Optimization (Adam):** The engine that iteratively updates weights. It navigates the 'Vanishing Gradient' challenge inherent in Sigmoid functions to minimize the MSE.
        """)

st.markdown("---")
st.caption("BSD3513 Introduction to Artificial Intelligence | Lab 4 – Neural Networks")
