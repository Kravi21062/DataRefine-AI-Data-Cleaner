version: '3.9'
services:
  datarefine-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - APP_TITLE=DataRefine
      - FRONTEND_ORIGIN=http://localhost:5173
      - ALLOWED_ORIGINS=http://localhost:5173
    volumes:
      - ./backend:/app/backend
