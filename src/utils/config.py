from omegaconf import OmegaConf

PATH = "src/configs/default.yml"


def load_config():
    """
    Load configs
    Returns:
        DictConfig: service configs
    """
    config = OmegaConf.load(PATH)
    return config
