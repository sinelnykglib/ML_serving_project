# app/schemas.py
from pydantic import BaseModel
from typing import Optional

class AirbnbInput(BaseModel):
    # Categorical
    neighbourhood_group: str
    
    # Binary
    instant_bookable: str
    host_identity_verified: str
    
    # Numeric
    service_fee: Optional[float] = 0
    minimum_nights: Optional[int] = 1
    number_of_reviews: Optional[int] = 0
    reviews_per_month: Optional[float] = 0.0
    review_rate_number: Optional[float] = 0.0
    calculated_host_listings_count: Optional[int] = 0
    availability_365: Optional[int] = 0
    lat: Optional[float] = 0.0
    long: Optional[float] = 0.0
    construction_year: Optional[int] = 0
