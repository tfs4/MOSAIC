"""MOSAIC: Morphology-Oriented Spatial Attention for Inferring Cell Expression.

Two-phase framework that (A) predicts single-cell gene expression from an H&E
patch and its K spatial nearest neighbours and (B) assigns cell lineages from
the predicted expression alone.
"""

from mosaic.pipeline import (  # noqa: F401
    IMAGE_PATHS,
    ExprClassifier,
    Img2ExprDataset,
    Img2ExprGnn,
    Img2ExprInferDataset,
    evaluate_classifier,
    generate_predictions,
    knn_indices,
    load_image_data,
    normalize_expr,
    run,
    set_seed,
    train_classifier,
    train_expression_model,
)

__version__ = "1.0.0"
