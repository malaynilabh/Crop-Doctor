from pydantic import BaseModel, ConfigDict, Field
from typing import List


class DiagnosisExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")  # LLM can't smuggle in unexpected keys

    diagnosis: str = Field(
        description="Explanation of the predicted plant disease."
    )

    disease_meaning: str = Field(
        description="Simple explanation of what the disease means."
    )

    symptoms: List[str] = Field(
        description="Common symptoms associated with the disease."
    )

    possible_causes: List[str] = Field(
        description="Possible causes or contributing conditions for the disease."
    )

    immediate_steps: List[str] = Field(
        description="Practical immediate steps the farmer can take."
    )

    prevention: List[str] = Field(
        description="Practical disease prevention guidance."
    )

    uncertainty: str = Field(
        description="Important uncertainty or limitation of the prediction."
    )


class ReviewVerdict(BaseModel):
    """Structured output for the self-review 'critic' step — see llm.py."""

    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(
        description="Whether the draft explanation meets every review criterion."
    )
    feedback: str = Field(
        description="Specific, actionable feedback on what to fix. 'none' if passed is true."
    )