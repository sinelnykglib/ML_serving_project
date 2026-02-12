from pydantic import BaseModel
from typing import Optional

class AirbnbInput(BaseModel):
    accommodates: int
    bedrooms: int
    bathrooms: float
    room_type: str
    neighbourhood_group: str
    instant_bookable: str
    host_identity_verified: str
    service_fee: Optional[float] = 0
    minimum_nights: Optional[int] = 1
    number_of_reviews: Optional[int] = 0
    reviews_per_month: Optional[float] = 0
    review_rate_number: Optional[float] = 0
    calculated_host_listings_count: Optional[int] = 1
    availability_365: Optional[int] = 0
    lat: Optional[float] = 0.0
    long: Optional[float] = 0.0
    construction_year: Optional[int] = 2000
    # Add other fields as necessary