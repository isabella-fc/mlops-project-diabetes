from kedro.pipeline import node, Pipeline
from .nodes import model_selection


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=model_selection,
            inputs=[
                "X_train_fe",  
                "X_test_fe",
                "y_train_matched",
                "y_test",
                "champion_dict",
                "champion_model",
                "params:model_selection"
            ],
            outputs="new_champion_model",
            name="model_selection_node"
        )
    ])