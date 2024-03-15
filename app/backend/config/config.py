import os


class Config:
    TITLE: str = "Waffle APP Backend API"
    DESCRIPTION: str = "Backend API for Waffle APP"

    WAFFLE_HUB_ROOT_DIR: str = os.getenv("WAFFLE_HUB_ROOT_DIR", "hubs")
    WAFFLE_DATASET_ROOT_DIR: str = os.getenv("WAFFLE_HUB_ROOT_DIR", "datasets")

    # uvicorn
    API_HOST: str = "0.0.0.0"  # for local
    API_PORT: int = os.getenv("API_PORT", 6001)

    # Redis
    # REDIS_HOST: str = "redis"
    REDIS_HOST: str = "0.0.0.0"  # for local
    REDIS_PORT: int = 6379

    # # Minio
    # # MINIO_HOST: str = "s3"
    # MINIO_HOST: str = "0.0.0.0"  # for local
    # MINIO_PORT: int = 9000
    # MINIO_ROOT_USER: str = "minio"
    # MINIO_ROOT_PASSWORD: str = "init123!!"
    # MINIO_DIR: str = "minio"
    # MINIO_DATA_BUCKET: str = "data"
    pass


__all__ = ["Config"]
