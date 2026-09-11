"""
SQLAlchemy ORM Database Models for AgriSense AI.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class PredictionHistory(Base):
    __tablename__ = "prediction_history"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    input_parameters = Column(Text, nullable=False)  # JSON string
    recommended_crop = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    top_recommendations = Column(Text, nullable=False)  # JSON string
    
    feedbacks = relationship("Feedback", back_populates="prediction")

class Feedback(Base):
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("prediction_history.id"), nullable=True)
    rating = Column(Integer, nullable=False)
    comments = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    prediction = relationship("PredictionHistory", back_populates="feedbacks")
