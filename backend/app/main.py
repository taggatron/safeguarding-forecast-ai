"""Main FastAPI application entry point for Safeguarding Forecast AI."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="Safeguarding Forecast AI",
    description="Predictive Safeguarding Demand Intelligence for South Devon College (Aggregate Operational Demand Forecasting Prototype)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "application": "Safeguarding Forecast AI",
        "subtitle": "Predictive Safeguarding Demand Intelligence",
        "organisation": "South Devon College",
        "docs": "/docs",
        "health": "/api/health",
    }

