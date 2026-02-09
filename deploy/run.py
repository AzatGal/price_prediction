#!/usr/bin/env python3
"""
Скрипт для запуска Real Estate Price Predictor
"""
import uvicorn
import os
import sys

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 60)
    print("  Real Estate Price Predictor")
    print("=" * 60)
    print("\nЗапуск сервера...")
    print("Откройте в браузере: http://localhost:8000\n")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
