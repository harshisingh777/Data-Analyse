# 📊 DataLens - AutoML Training Platform

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

## 🚀 Overview

DataLens is a sophisticated AutoML (Automated Machine Learning) web platform that democratizes machine learning by enabling users to train intelligent models without deep technical expertise. Simply upload your CSV dataset, and DataLens will automatically analyze patterns, select optimal features, and train the best-performing model for your data.

### ✨ Key Features

- **🤖 Fully Automated Training**: No manual model selection or hyperparameter tuning required
- **🎯 Smart Target Detection**: Automatically identifies target variables and determines task type
- **📊 Comprehensive Analytics**: Detailed performance metrics and feature importance analysis
- **⚡ Real-time Processing**: Live training progress with beautiful animations
- **🎨 Modern UI/UX**: Responsive design with engaging visual feedback
- **📈 Performance Insights**: Cross-validation metrics and model comparison

## 🏗️ Architecture

### Frontend Components
- **HTML5**: Semantic structure with accessibility considerations
- **CSS3**: Modern styling with animations, gradients, and responsive design
- **Vanilla JavaScript**: File handling, API communication, and dynamic UI updates

### Backend Integration
- **RESTful API**: Communicates with Flask backend via `/autotrain` endpoint
- **File Upload**: Secure CSV processing with size validation (100MB limit)
- **Real-time Feedback**: Progress tracking and error handling

## 📋 Prerequisites

### System Requirements
- Modern web browser (Chrome 90+, Firefox 88+, Safari 14+, Edge 90+)
- Internet connection for initial loading
- Minimum 4GB RAM recommended for large datasets

### Backend Dependencies
The frontend expects a Flask backend server running on `localhost:5000` with the following endpoint:
```
POST /autotrain
Content-Type: multipart/form-data
Body: CSV file upload
```

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/datalens-automl.git
cd datalens-automl
```

### 2. Serve the Frontend
You can serve the HTML file using any web server:

#### Option A: Python HTTP Server
```bash
python -m http.server 8080
```

#### Option B: Node.js HTTP Server
```bash
npx http-server -p 8080
```

#### Option C: Live Server (VS Code Extension)
- Install the "Live Server" extension in VS Code
- Right-click on `index.html` and select "Open with Live Server"

### 3. Backend Setup (Required)
Ensure your backend server is running on `http://localhost:5000` with the `/autotrain` endpoint.

Expected backend response format:
```json
{
  "target": "target_column_name",
  "task": "classification|regression",
  "metrics": {
    "accuracy": {"mean": 0.95, "std": 0.02},
    "f1_score": {"mean": 0.94, "std": 0.03}
  },
  "feature_importances": [
    ["feature_1", 0.25],
    ["feature_2", 0.20],
    ["feature_3", 0.15]
  ]
}
```

## 🎯 Usage Guide

### Step 1: Upload Dataset
1. Click the "Choose Dataset File" button
2. Select a CSV file (max 100MB)
3. File information will be displayed below the upload area

### Step 2: Start Training
1. Click the "🚀 Start Training" button
2. The interface will show a loading animation
3. Training progress is displayed in real-time

### Step 3: View Results
Once training completes, you'll see:
- **Model Details**: Target variable and task type
- **Performance Metrics**: Cross-validation scores with confidence intervals
- **Feature Importances**: Ranked list of most influential features
- **Visual Analytics**: Progress bars and metric cards

## 📊 Supported Data Formats

### CSV Requirements
- **Format**: Comma-separated values (.csv)
- **Size**: Up to 100MB
- **Headers**: First row should contain column names
- **Encoding**: UTF-8 recommended

### Data Types Supported
- **Numerical**: Integers, floats, decimals
- **Categorical**: Strings, text labels
- **Boolean**: True/false values
- **Date/Time**: ISO format dates (auto-detected)

### Dataset Guidelines
- Minimum 100 rows recommended
- Missing values are automatically handled
- Target variable can be numerical (regression) or categorical (classification)

## 🎨 UI Components & Features

### Visual Elements
- **Animated Background**: Moving gradient with floating particles
- **Loading Animations**: Smooth spinners and progress indicators
- **Responsive Cards**: Clean, modern card-based layout
- **Interactive Buttons**: Hover effects and state changes

### Color Scheme
- **Primary**: #4CAF50 (Success Green)
- **Secondary**: #2196F3 (Material Blue)
- **Accent**: #FF9800 (Amber)
- **Background**: Gradient blend of pastels

### Responsive Design
- **Desktop**: Grid-based layout with side-by-side components
- **Tablet**: Stacked layout with maintained proportions
- **Mobile**: Single-column layout with optimized touch targets

## 🔧 Customization Options

### Styling Modifications
Edit the CSS section in the HTML file to customize:
- Color themes and brand colors
- Animation speeds and effects
- Layout breakpoints and grid systems
- Typography and font choices

### Backend Integration
To integrate with a different backend:
1. Update the API endpoint URL in the `trainModel()` function
2. Modify the request/response format handling
3. Adjust error handling for your API structure

## 📈 Performance Metrics

The platform displays various metrics based on the task type:

### Classification Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: Positive prediction accuracy
- **Recall**: True positive detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating curve

### Regression Metrics
- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error

## 🛡️ Security Considerations

### File Upload Safety
- Client-side file type validation
- Size limitations (100MB max)
- No server-side file persistence in frontend

### Data Privacy
- Files are processed server-side only
- No client-side data storage
- No browser localStorage/sessionStorage usage

## 🚨 Error Handling

### Common Error Scenarios
- **No file selected**: Warning alert with instructions
- **File too large**: Size validation with user feedback
- **Network errors**: Connection failure messages
- **Backend errors**: Server response error display

### User Feedback
- Clear error messages with actionable advice
- Visual indicators for different error types
- Graceful degradation for offline scenarios

## 🎛️ Browser Compatibility

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | 90+ | ✅ Fully Supported |
| Firefox | 88+ | ✅ Fully Supported |
| Safari | 14+ | ✅ Fully Supported |
| Edge | 90+ | ✅ Fully Supported |
| Opera | 76+ | ✅ Fully Supported |

### Progressive Enhancement
- Base functionality works on older browsers
- Advanced animations gracefully degrade
- Fallback styles for unsupported features

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the Repository**
   ```bash
   git fork https://github.com/your-username/datalens-automl.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Commit Changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```

4. **Push to Branch**
   ```bash
   git push origin feature/amazing-feature
   ```

5. **Open a Pull Request**

### Contribution Guidelines
- Follow the existing code style and structure
- Add comments for complex functionality
- Test on multiple browsers before submitting
- Update documentation for new features

## 🐛 Troubleshooting

### Common Issues

#### Training Button Not Working
- **Cause**: No file selected
- **Solution**: Upload a CSV file before clicking train

#### Connection Failed Error
- **Cause**: Backend server not running
- **Solution**: Ensure Flask server is running on localhost:5000

#### File Upload Issues
- **Cause**: File size exceeds 100MB or wrong format
- **Solution**: Use smaller CSV files with proper formatting

#### Display Issues on Mobile
- **Cause**: Viewport settings or CSS conflicts
- **Solution**: Clear browser cache and ensure responsive meta tag

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Authors & Acknowledgments

- **Development Team**: DataLens Contributors
- **Design Inspiration**: Material Design principles
- **Animation Libraries**: CSS3 keyframes and transitions

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/your-username/datalens-automl/issues)
- **Documentation**: [Wiki Pages](https://github.com/your-username/datalens-automl/wiki)
- **Email**: support@datalens.com

## 🗺️ Roadmap

### Upcoming Features
- [ ] Multiple file format support (Excel, JSON, Parquet)
- [ ] Real-time model comparison dashboard
- [ ] Export trained models (pickle, ONNX)
- [ ] Dark mode theme toggle
- [ ] Drag-and-drop file upload
- [ ] Integration with popular ML platforms

### Long-term Goals
- [ ] No-code model deployment
- [ ] Collaborative workspace features
- [ ] Advanced visualization tools
- [ ] API key authentication
- [ ] Cloud storage integration

---

**Made with ❤️ by the DataLens Team**

*Transform your data into intelligent models with automated machine learning*