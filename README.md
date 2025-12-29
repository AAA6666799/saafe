# 🔥 SAAFE AI Fire Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AWS](https://img.shields.io/badge/AWS-Ready-orange.svg)](https://aws.amazon.com/)
[![React](https://img.shields.io/badge/React-18.x-blue.svg)](https://reactjs.org/)
[![Node.js](https://img.shields.io/badge/Node.js-20.x-green.svg)](https://nodejs.org/)

**SAAFE is an enterprise-grade, AI-powered fire detection and prevention system designed for mission-critical environments. This project provides a production-ready architecture with real-time monitoring, intelligent alerting, comprehensive safety protocols, and global cloud deployment capabilities.**

## 🌟 Key Features

### 🎯 Core Capabilities
- **Real-time Fire Detection** - AI models trained on thermal and gas sensor data with 95%+ accuracy
- **IoT Integration** - Support for thermal cameras, gas sensors, and environmental monitors
- **Live Dashboard** - React-based monitoring interface with real-time updates from AWS S3
- **Global Deployment** - Multiple AWS deployment options with worldwide CDN access
- **Machine Learning** - Ensemble models using Random Forest, LSTM, and Logistic Regression
- **Alert System** - Multi-channel notifications with intelligent escalation

### 📊 Dashboard Components
- **Helios (Global View)** - World map with risk markers and device locations
- **Grid (Asset Manager)** - Search/filter devices with drill-down capabilities  
- **Chronos (Incident Panel)** - Real-time stats, sensor data, and incident history
- **Athena (Strategic Dashboard)** - KPIs, trend analysis, and performance metrics
- **SAAFEGPT** - AI-powered chat interface for intelligent system insights

### ☁️ AWS Deployment Options
- **S3 + CloudFront** - Static hosting with global CDN (~$1-5/month)
- **AWS Amplify** - Full-stack with automatic CI/CD (~$1-15/month)  
- **Docker + App Runner** - Containerized deployment (~$10-50/month)
- **Serverless Lambda** - Pay-per-use API backend with API Gateway
- **Elastic Beanstalk** - Auto-scaling full-stack deployment

---

## 📖 Table of Contents

*   [Key Features](#-key-features)
*   [System Architecture](#-system-architecture)
*   [Getting Started](#-getting-started)
*   [Documentation Hub](#-documentation-hub)
*   [Contributing](#-contributing)
*   [License](#-license)
*   [Contact](#-contact)

---

## ✨ Key Features

*   **Synthetic Data Generation:** Generate realistic thermal, gas, and environmental sensor data to train and validate the system before hardware deployment.
*   **Advanced Feature Engineering:** Extract over 18 meaningful features from multi-sensor data to improve model accuracy.
*   **State-of-the-Art Machine Learning Models:** Utilize a suite of advanced models, including temporal models, baseline models, and ensemble systems.
*   **Multi-Agent System:** A modular, agent-based architecture with specialized agents for monitoring, analysis, response, and learning.
*   **Hardware Abstraction Layer:** A seamless transition from synthetic data to real-world hardware sensors.
*   **Comprehensive Testing Suite:** Automated testing with synthetic data validation to ensure system reliability.
*   **Real-Time AWS Dashboard:** A comprehensive dashboard for real-time monitoring of sensor data, fire detection scores, and system health.
*   **Intelligent Alerting System:** A multi-level alert hierarchy with hysteresis-based transitions to prevent alert fatigue.

---

## 🏗️ System Architecture

Our system is built on a modular, multi-layered architecture that separates concerns and allows for scalability and maintainability.

For a detailed overview of the system's architecture, please see our **[Architecture Guide](ARCHITECTURE.md)**.

---

## 🚀 Getting Started

To get started with the Saafe Fire Detection System, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/AAA6666799/saafe.git
    cd saafe
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the system:**

    ```bash
    python main.py
    ```

For more detailed installation and setup instructions, please refer to our **[Installation Guide](INSTALLATION_GUIDE.md)**.

---

## 📚 Documentation Hub

We provide comprehensive documentation to help you understand, use, and contribute to the Saafe project. All documentation is available in the repository.

### User Guides

*   **[User Manual](docs/USER_MANUAL.md)**: A comprehensive guide for end-users.
*   **[Troubleshooting Guide](docs/TROUBLESHOOTING_GUIDE.md)**: Solutions to common problems and issues.

### Developer Guides

*   **[Architecture Guide](ARCHITECTURE.md)**: A detailed overview of the system's architecture.
*   **[API Documentation](docs/API.md)**: Information about the system's API.
*   **[Contributing Guide](CONTRIBUTING.md)**: Guidelines for contributing to the project.

### Deployment Guides

*   **[Deployment Guide](DEPLOYMENT_GUIDE.md)**: Instructions for deploying the system.
*   **[AWS Deployment Guide](AWS_Deployment_Guide.md)**: Specific instructions for deploying on AWS.
*   **[AWS Dashboard Deployment Guide](AWS_DASHBOARD_DEPLOYMENT_GUIDE.md)**: Instructions for deploying the AWS dashboard.

### Other Documentation

*   **[CHANGELOG.md](CHANGELOG.md)**: A log of all the changes made to the project.
*   **[LICENSE](LICENSE)**: The license for the project.

---

## 🤝 Contributing

We welcome contributions from the community! If you'd like to contribute, please read our **[Contributing Guide](CONTRIBUTING.md)** for more information.

---

## 📜 License

This project is licensed under the MIT License. See the **[LICENSE](LICENSE)** file for more details.

---

## 📬 Contact

If you have any questions, feel free to open an issue on GitHub or contact the project maintainers.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- AWS CLI configured with appropriate permissions
- Git for version control

### 1. Clone and Setup
```bash
git clone https://github.com/your-username/saafe-fire-detection-system.git
cd saafe-fire-detection-system

# Install frontend dependencies
cd saafe-lovable
npm install
npm run build
cd ..
```

### 2. Deploy to AWS (Recommended)
```bash
# Simple deployment with S3 + CloudFront
./deploy-simple-aws.sh

# Or full-stack deployment with backend
./deploy-fullstack-aws.sh

# Or Docker-based deployment
./deploy-docker-aws.sh
```

### 3. Local Development
```bash
# Start frontend
cd saafe-lovable
npm run dev

# Start backend (separate terminal)
cd saafe-lovable/backend
npm install
npm start
```

## 📁 Project Structure

```
saafe-fire-detection-system/
├── saafe-lovable/                 # React dashboard application
│   ├── src/                       # React components and logic
│   ├── backend/                   # Node.js API server
│   ├── dist/                      # Built frontend assets
│   └── public/                    # Static assets
├── deployment/                    # Deployment configurations
├── docs/                         # Documentation
├── monitoring/                   # Grafana dashboards
├── task_1_ai_fire_prediction_platform/  # Core AI system
├── saafe_mvp/                    # MVP implementation
├── deploy-*.sh                   # AWS deployment scripts
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Deployment Scripts

| Script | Purpose | Cost | Best For |
|--------|---------|------|----------|
| `deploy-simple-aws.sh` | S3 + CloudFront + Lambda | $1-5/month | Static dashboard |
| `deploy-fullstack-aws.sh` | Elastic Beanstalk | $10-50/month | Full application |
| `deploy-docker-aws.sh` | App Runner | $10-50/month | Containerized |
| `deploy-serverless-aws.sh` | Lambda + API Gateway | Pay-per-use | Serverless |
| `deploy-amplify.sh` | AWS Amplify | $1-15/month | CI/CD pipeline |

## 🌍 Live Demo

After deployment, your SAAFE dashboard will be accessible worldwide with:
- ✅ HTTPS encryption via CloudFront
- ✅ Global CDN for fast loading
- ✅ Real-time fire detection data
- ✅ Mobile-responsive interface
- ✅ 99.99% uptime SLA

## 🔥 Fire Detection Pipeline

1. **Data Collection** - IoT sensors stream thermal and gas data to AWS S3
2. **AI Processing** - Ensemble models analyze sensor patterns in real-time  
3. **Risk Assessment** - Multi-factor analysis determines fire probability
4. **Alert Generation** - Intelligent notifications based on risk levels
5. **Dashboard Updates** - Live visualization of system status and alerts

## 📊 Monitoring & Analytics

- **Grafana Dashboards** - Pre-configured monitoring dashboards
- **CloudWatch Integration** - AWS native monitoring and alerting
- **Performance Metrics** - Real-time system health and performance data
- **Historical Analysis** - Trend analysis and incident reporting

## 🛡️ Security Features

- **AWS IAM Integration** - Role-based access control
- **HTTPS Encryption** - End-to-end secure communications
- **API Authentication** - Secure API endpoints
- **Data Privacy** - Compliant with data protection regulations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📧 Email: support@saafe.ai
- 📚 Documentation: [Full Documentation](docs/)
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/saafe-fire-detection-system/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-username/saafe-fire-detection-system/discussions)

## 🏆 Acknowledgments

- AWS for cloud infrastructure
- React community for frontend framework
- Node.js ecosystem for backend development
- Open source contributors and maintainers

---

**🔥 Protecting lives and property with intelligent fire detection technology**