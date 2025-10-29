# Deployment Guide

## Local Development

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone and navigate to the app directory:**
```bash
cd employee-promotion-app
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the app:**
```bash
# Option 1: Using the launcher script
python run_app.py

# Option 2: Direct streamlit command
streamlit run app.py
```

4. **Access the app:**
Open your browser and go to `http://localhost:8501`

## Production Deployment

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub:**
```bash
git add .
git commit -m "Add Streamlit app"
git push origin main
```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select the `employee-promotion-app` folder
   - Deploy!

### Option 2: Docker Deployment

1. **Create Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. **Build and run:**
```bash
docker build -t employee-promotion-app .
docker run -p 8501:8501 employee-promotion-app
```

### Option 3: Heroku

1. **Create Procfile:**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. **Deploy:**
```bash
heroku create your-app-name
git push heroku main
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: For AI insights (optional)
- `MODEL_PATH`: Path to trained model files
- `DATA_PATH`: Path to data files

### Model Integration
To integrate with your V3 trained model:

1. **Copy model files:**
```bash
cp ../v3_artifacts/*.pkl models/
```

2. **Update config:**
Edit `config/app_config.yaml` to point to your model files.

## Features

### 📊 Dashboard
- KPI metrics and visualizations
- Recent predictions display
- Performance charts

### 🔮 Predictions
- CSV file upload
- Manual form input
- Sample data demonstration
- Results export

### 📈 Model Analysis
- Performance metrics
- Confusion matrix
- ROC/PR curves
- Feature importance

### 🤖 AI Insights
- Pattern analysis
- Risk identification
- Personalized recommendations
- Batch insights generation

## Troubleshooting

### Common Issues

1. **Import errors:**
   - Ensure all dependencies are installed
   - Check Python path configuration

2. **Model loading errors:**
   - Verify model files exist
   - Check file permissions

3. **Data processing errors:**
   - Ensure input data has required columns
   - Check data types and formats

### Performance Optimization

1. **Caching:**
   - Use `@st.cache_data` for expensive operations
   - Cache model loading and data processing

2. **Memory management:**
   - Process data in chunks for large files
   - Clear cache periodically

3. **UI optimization:**
   - Use `use_container_width=True` for responsive design
   - Implement pagination for large tables

## Security Considerations

1. **Data privacy:**
   - Don't store sensitive employee data
   - Use secure file uploads
   - Implement data anonymization

2. **Access control:**
   - Add authentication if needed
   - Restrict file upload types
   - Validate input data

3. **Model security:**
   - Protect model files
   - Validate predictions
   - Monitor for adversarial inputs

## Monitoring and Maintenance

1. **Performance monitoring:**
   - Track prediction accuracy
   - Monitor response times
   - Log errors and exceptions

2. **Model updates:**
   - Regular retraining schedule
   - A/B testing for new models
   - Version control for model files

3. **User feedback:**
   - Collect prediction feedback
   - Monitor user interactions
   - Update based on feedback

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the logs for error messages
3. Ensure all dependencies are up to date
4. Verify data format and model compatibility
