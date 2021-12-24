"""
    Config 클래스나 랜덤 시드를 고정하는 코드.
    환경변수 값들도 포함합니다.
"""
import random
import numpy as np
import torch

GOOGLE_APPLICATION_CREDENTIAL = './credential.json' # GCP 서비스 접근을 위한 파일
MLFLOW_TRACKING_URI = 'http://localhost:80' # GCP 서버 주소

class Config:
    def __init__(
        self,
        dropout1 : float, 
        dropout2 : float, 
        learning_rate : float, 
        label_smoothing : float, 
        epochs: int, 
        embedding_dim: int, 
        channel: int
        ) -> None:
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.learning_rate = learning_rate
        self.label_smoothing = label_smoothing
        self.epochs = epochs
        self.embedding_dim = embedding_dim
        self.channel = channel

def set_seed(random_seed) -> None:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
