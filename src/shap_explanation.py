import shap
import numpy as np
import matplotlib.pyplot as plt

def explain_predictions(embedding: np.ndarray):
    # Create background dataset for SHAP
    background = np.random.randn(100, embedding.shape[1])
    # KernelExplainer with mean prediction function
    explainer = shap.KernelExplainer(lambda x: np.mean(x, axis=1), background)
    shap_values = explainer.shap_values(embedding, nsamples=100)

    # Summary plot
    shap.summary_plot(shap_values, embedding)
    plt.show()