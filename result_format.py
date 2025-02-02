from pydantic import BaseModel, Field
from typing import List, Optional

class YAMLResponse(BaseModel):
    """Structure for YAML configuration response"""
    yaml_config: str
    confidence_score: float = Field(description="Confidence score of the retrieval")
    sources: List[str] = Field(description="Source documents used")
    explanation: Optional[str] = Field(description="Explanation of the generated configuration")
