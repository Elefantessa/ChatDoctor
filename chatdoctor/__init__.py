"""
ChatDoctor - A Medical AI Chatbot
Fine-tuned on LLaMA using medical domain knowledge.
"""

__version__ = "2.0.0"
__author__ = "ChatDoctor Team"

from chatdoctor.core.model import ChatDoctorModel
from chatdoctor.core.config import Config

__all__ = [
    "ChatDoctorModel",
    "Config",
    "__version__",
]
