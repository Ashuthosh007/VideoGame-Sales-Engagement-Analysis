# 🎮 Video Game Sales and Engagement Analysis

## 📌 Project Overview
This project analyzes and visualizes video game **sales** and **engagement data** to uncover insights into trends, user behavior, platform performance, and genre popularity.  
It combines datasets (`games.csv` and `vgsales.csv`) to explore **how ratings, wishlists, genres, and platforms impact global sales and engagement.**  

---

## 🛠️ Tech Stack
- **Python** (Data Cleaning, EDA, SQL setup, Visualization)
- **SQLite** (Database storage and querying)
- **Matplotlib & Seaborn** (Visualizations)
- **Pandas & NumPy** (Data preprocessing and analysis)
- **Power BI** (Interactive dashboards)

---

## 📂 Datasets
1. **games.csv** – Game engagement data  
   - Title, Rating, Genres, Plays, Backlogs, Wishlist, Release Date, Platform, Developer  
2. **vgsales.csv** – Sales data  
   - Name, Platform, Year, Genre, Publisher, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales  

---

## 🚀 Features

### 🔍 **Data Analytics**
- ✅ **Data Cleaning & Preprocessing**: Handle missing values, normalize formats
- ✅ **Exploratory Data Analysis**: 12+ comprehensive visualizations
- ✅ **Statistical Analysis**: Correlation analysis, trend identification
- ✅ **Performance Metrics**: KPIs, success scores, engagement rates

### 🗄️ **Database Management**
- ✅ **SQL Database Creation**: Structured tables with relationships
- ✅ **Data Integrity**: Primary keys, foreign keys, constraints
- ✅ **Query Optimization**: Efficient data retrieval and analysis
- ✅ **Backup & Recovery**: Database maintenance procedures

### 📊 **Interactive Dashboards**
- ✅ **8 Dashboard Pages**: Executive summary, genre analysis, regional insights
- ✅ **50+ DAX Measures**: Advanced calculations and KPIs
- ✅ **Real-time Monitoring**: Live performance tracking
- ✅ **Mobile Optimization**: Responsive design for all devices

### 🤖 **Advanced Analytics**
- ✅ **Predictive Modeling**: Sales forecasting with confidence intervals
- ✅ **Clustering Analysis**: Game performance segmentation
- ✅ **What-If Scenarios**: Strategic planning simulations
- ✅ **Automated Alerts**: Performance monitoring and notifications 

---

## 📊 Example Insights
- Best-selling platforms & genres  
- Regional sales distribution (NA, EU, JP, Others)  
- Relationship between **ratings & sales**  
- Top wishlisted and highest-rated games  
- Productivity of developers and publishers  

---

## 📥 Installation & Setup

1. Clone this repository or download the files  
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure `games.csv` and `vgsales.csv` are in the same directory as the script  
4. Run the script:

```bash
# Execute the main analysis script
python src/video_game_analysis.py
```

5. Power BI Setup
 1. Open Power BI Desktop
 2. Import `powerbi/VideoGameAnalysis.pbix`
 3. Update data source connections if needed
 4. Refresh data to load latest information

---

## 📦 Project Structure
```
📁 project-folder
│── video_game_analysis.py  # Main project script
│── requirements.txt        # Python dependencies
│── games.csv               # Engagement dataset
│── vgsales.csv             # Sales dataset
│── video_games.db          # SQLite database (created after running)
│── README.md               # Project documentation
│── VideoGameAnalysis.pbix  # PowerBi Dashboard
```

---

## 📈 Deliverables
- Cleaned datasets
- SQLite database (`video_games.db`)
- Python EDA visualizations
- Key insights and recommendations
- Power BI dashboards

---

## 🏆 Business Use Cases
- 🎯 **Marketing Strategy**: Target top genres, platforms, and regions  
- 🛠️ **Product Development**: Identify features linked to success  
- 📊 **Sales Forecasting**: Predict trends and demand  
- 🌍 **Resource Allocation**: Focus on high-potential markets  


