from kedro.pipeline import Pipeline, node, pipeline
from .nodes import model_train

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=model_train,
            inputs=[
                "X_train_fe",
                "X_test_fe",
                "y_train_matched",
                "y_test",
                "params:train",
                "selected_features"
            ],
            outputs=[
                "trained_model",
                "final_features",
                "train_metrics",
                "output_plot"
            ],
            name="model_train_node"
        )
    ])
