from .fairface_classifiers import get_fairface_classifier_4, get_fairface_classifier_7
from .latent_classifier_resnet_enc_multihead import get_pcd_classifier


CLASSIFIER_CONFIGS = {
    "pcd_all_heads": {
        "path": {
            "config": "models/base_classifier_resnet_gender_age_race/FFHQA_classifier_resnet_enc_multihead.yaml", 
            "weights": "models/base_classifier_resnet_gender_age_race/checkpoints_classifier/best_classifier.pt",
        }, 
        "get_func": get_pcd_classifier,
    },
}
