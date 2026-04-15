from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "label-to-cuboid-service"
    host: str = "0.0.0.0"
    port: int = 8000

    log_level: str = "INFO"
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    gpu_batch_size: int = Field(default=1, ge=1)
    max_images_per_request: int = Field(default=32, ge=1)

    sam3_model_id: str = "facebook/sam3"
    sam3_checkpoint_path: str = ""
    depth_model_id: str = "apple/DepthPro-hf"
    depth_local_path: str = ""
    device: str = "cuda"
    model_dtype: str = "float16"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

