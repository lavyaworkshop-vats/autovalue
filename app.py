"""AutoValue AI - Application Entrypoint
Run this script to launch the FastAPI backend and interactive web dashboard.
"""
import os
import uvicorn

if __name__ == '__main__':
    print("=" * 65)
    print("  AutoValue AI - Used Vehicle Valuation & Market Intelligence")
    print("=" * 65)
    print("  • Web Dashboard:  http://localhost:8000")
    print("  • API Swagger:    http://localhost:8000/docs")
    print("  • OpenAPI Specs:  http://localhost:8000/openapi.json")
    print("=" * 65)
    
    # Run Uvicorn ASGI server
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )
