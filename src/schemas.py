from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class ExtractedFeatures(BaseModel):
    # Numeric
    Gr_Liv_Area: Optional[float] = Field(None, alias="Gr Liv Area")
    Garage_Area: Optional[float] = Field(None, alias="Garage Area")
    Year_Built: Optional[int] = Field(None, alias="Year Built")
    Total_Bsmt_SF: Optional[float] = Field(None, alias="Total Bsmt SF")
    Lot_Area: Optional[float] = Field(None, alias="Lot Area")
    # Ordinal
    Overall_Qual: Optional[int] = Field(None, alias="Overall Qual")
    Overall_Cond: Optional[int] = Field(None, alias="Overall Cond")
    Bsmt_Qual: Optional[str] = Field(None, alias="Bsmt Qual")
    # Nominal
    Neighborhood: Optional[str] = None
    MS_Zoning: Optional[str] = Field(None, alias="MS Zoning")
    Sale_Condition: Optional[str] = Field(None, alias="Sale Condition")

    completeness: Dict[str, bool] = Field(default_factory=dict)
    missing_features: List[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True

class PredictionRequest(BaseModel):
    query: str

class PredictionResponse(BaseModel):
    extracted: ExtractedFeatures
    predicted_price: float
    interpretation: str