from kedro.pipeline import Pipeline, node, pipeline
from .nodes import model_train

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=model_train,
            inputs=[
                "X_train",
                "X_test",
                "y_train",
                "y_test",
                "params:train",
                "selected_features"
            ],
            outputs=[
                "trained_model",
                "final_features",
                "train_metrics",
                "shap_summary_plot"
            ],
            name="model_train_node"
        )
    ])
