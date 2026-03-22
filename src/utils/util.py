import os
import shutil
import random
import torch
import inspect
from PIL import Image
from transformers import PretrainedConfig

def image_grid(imgs, rows, cols):
    """
    Combines multiple images into a grid layout.
    
    Args:
        imgs: List of PIL Image objects
        rows: Number of rows in the grid
        cols: Number of columns in the grid
    
    Returns:
        PIL Image object containing the grid
    """
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def mask_text(text_list, mask_ratio=0.5):
    """
    Randomly masks words in text sequences for data augmentation.
    
    Args:
        text_list: List of text strings to process
        mask_ratio: Probability of masking each word (default: 0.5)
    
    Returns:
        List of masked text strings
    """
    masked_text_list = []
    for text in text_list:
        words = text.split(".")
        mask = torch.tensor([random.random() > mask_ratio for _ in range(len(words))], dtype=torch.float)
        masked_words = [word for i, word in enumerate(words) if mask[i]]
        masked_text = ".".join(masked_words)
        masked_text_list.append(masked_text)
    return masked_text_list

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    """
    Dynamically loads the appropriate Text Encoder class based on model configuration.
    
    Args:
        pretrained_model_name_or_path: Path or identifier of the pretrained model
        revision: Model revision/git branch
    
    Returns:
        Appropriate Text Encoder class
    
    Raises:
        ValueError: If the model architecture is not supported
    """
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def get_original_py_file(imported_class):
    """
    Retrieves the source file path of a class for backup purposes.
    
    Args:
        imported_class: Class object to inspect
    
    Returns:
        String path to the source file
    """
    module = inspect.getmodule(imported_class)
    return module.__file__

def copy_files_to_directory(target_dir, current_file_path, extra_files=[]):
    """
    Backups current script and key model files to output directory for reproducibility.
    
    Args:
        target_dir: Destination directory for backup
        current_file_path: Path to the main script being executed
        extra_files: List of additional files to backup (e.g., model files)
    """
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Create backup subdirectory with timestamp
    target_dir = os.path.join(target_dir, f"code_backup_{current_time}")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    
    # Backup main script
    if current_file_path:
        shutil.copy(current_file_path, target_dir)
    
    # Backup additional specified files
    for file in extra_files:
        if file and os.path.exists(file):
            shutil.copy(file, target_dir)