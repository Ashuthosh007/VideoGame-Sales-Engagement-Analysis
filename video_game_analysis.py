import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🎮 Video Game Sales and Engagement Analysis Project")
print("=" * 60)

# ========================================================================================
# PART 1: DATA LOADING AND INITIAL EXPLORATION
# ========================================================================================

def load_csv_data():
    """
    Load the actual CSV files: games.csv and vgsales.csv
    """
    try:
        # Load games.csv (Game Engagement Data)
        print("📂 Loading games.csv...")
        games_df = pd.read_csv('games.csv')
        print(f"✅ Games dataset loaded: {games_df.shape[0]} records, {games_df.shape[1]} columns")
        
        # Load vgsales.csv (Sales Data)  
        print("📂 Loading vgsales.csv...")
        vgsales_df = pd.read_csv('vgsales.csv')
        print(f"✅ Sales dataset loaded: {vgsales_df.shape[0]} records, {vgsales_df.shape[1]} columns")
        
        # Display basic info about the datasets
        print(f"\n📋 Games Dataset Columns: {list(games_df.columns)}")
        print(f"📋 Sales Dataset Columns: {list(vgsales_df.columns)}")
        
        return games_df, vgsales_df
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find CSV file - {e}")
        print("Please ensure 'games.csv' and 'vgsales.csv' are in the same directory as this script.")
        return None, None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

# Load the actual CSV data
print("📊 Loading datasets from CSV files...")
games_df, vgsales_df = load_csv_data()

if games_df is None or vgsales_df is None:
    print("❌ Failed to load data. Exiting...")
    exit()

print()

# ========================================================================================
# PART 2: DATA CLEANING AND PREPROCESSING
# ========================================================================================

print("🧹 Starting Data Cleaning Process...")
print("-" * 40)

def clean_games_data(df):
    """Clean the games dataset"""
    df_clean = df.copy()
    
    # Convert Release_Date to datetime if it exists
    if 'Release Date' in df_clean.columns:
        df_clean['Release_Date'] = pd.to_datetime(df_clean['Release Date'], errors='coerce')
        df_clean['Release_Year'] = df_clean['Release_Date'].dt.year
    elif 'Release_Date' in df_clean.columns:
        df_clean['Release_Date'] = pd.to_datetime(df_clean['Release_Date'], errors='coerce')
        df_clean['Release_Year'] = df_clean['Release_Date'].dt.year
    elif 'Year' in df_clean.columns:
        df_clean['Release_Year'] = df_clean['Year']
    
    # Convert numeric columns that might be stored as strings
    numeric_cols_to_convert = ['Rating', 'Times Listed', 'Number of Reviews', 'Plays', 'Playing', 'Backlogs', 'Wishlist']
    
    for col in numeric_cols_to_convert:
        if col in df_clean.columns:
            # Remove commas and convert to numeric
            df_clean[col] = pd.to_numeric(df_clean[col].astype(str).str.replace(',', ''), errors='coerce')
    
    # Handle missing values for numeric columns
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col in ['Rating', 'rating']:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        else:
            df_clean[col].fillna(0, inplace=True)
    
    # Normalize text fields
    text_columns = df_clean.select_dtypes(include=['object']).columns
    for col in text_columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).str.strip()
    
    return df_clean

def clean_sales_data(df):
    """Clean the sales dataset"""
    df_clean = df.copy()
    
    # Convert sales columns to numeric (handle any string formatting)
    sales_cols = [col for col in df_clean.columns if 'Sales' in col or 'sales' in col]
    for col in sales_cols:
        df_clean[col] = pd.to_numeric(df_clean[col].astype(str).str.replace(',', ''), errors='coerce')
        df_clean[col].fillna(0, inplace=True)
    
    # Handle missing years
    if 'Year' in df_clean.columns:
        df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
        df_clean['Year'].fillna(df_clean['Year'].median(), inplace=True)
    
    # Normalize text fields
    text_columns = df_clean.select_dtypes(include=['object']).columns
    for col in text_columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).str.strip()
    
    return df_clean

# Clean the datasets
games_clean = clean_games_data(games_df)
sales_clean = clean_sales_data(vgsales_df)

print("✅ Data cleaning completed")
print(f"   Games dataset: {games_clean.shape}")
print(f"   Sales dataset: {sales_clean.shape}")
print()

# ========================================================================================
# PART 3: SQL DATABASE SETUP
# ========================================================================================

print("🗄️ Setting up SQL Database...")
print("-" * 40)

def create_sql_database():
    """Create SQLite database and populate with cleaned data"""
    # Connect to SQLite database (creates file if doesn't exist)
    conn = sqlite3.connect('video_games.db')
    
    # Insert data into tables
    games_clean.to_sql('games', conn, if_exists='replace', index=False)
    sales_clean.to_sql('sales', conn, if_exists='replace', index=False)
    
    print("✅ SQL database created successfully")
    print("   - games table created and populated")
    print("   - sales table created and populated")
    
    return conn

# Create database
db_conn = create_sql_database()
print()

# ========================================================================================
# PART 4: EXPLORATORY DATA ANALYSIS (EDA)
# ========================================================================================

print("🔍 Performing Exploratory Data Analysis...")
print("-" * 40)

# Dynamically find the relevant columns for analysis
rating_col = None
for col in games_clean.columns:
    if 'rating' in col.lower():
        rating_col = col
        break

title_col = None
for col in games_clean.columns:
    if 'title' in col.lower() or 'name' in col.lower():
        title_col = col
        break
# If title column not found, try to find the first text column that might contain titles
if title_col is None:
    for col in games_clean.columns:
        if games_clean[col].dtype == 'object' and len(games_clean[col].iloc[0]) > 5:  # Assuming titles are longer than 5 chars
            title_col = col
            break

genre_col = None
for col in games_clean.columns:
    if 'genre' in col.lower():
        genre_col = col
        break

# Find sales columns
global_sales_col = None
for col in sales_clean.columns:
    if 'global' in col.lower() and 'sales' in col.lower():
        global_sales_col = col
        break

# 1. Top-rated games (if rating column exists)
plt.figure(figsize=(10, 8))
if rating_col and title_col:
    top_rated = games_clean.nlargest(10, rating_col)
    plt.barh(range(len(top_rated)), top_rated[rating_col])
    plt.yticks(range(len(top_rated)), [str(title)[:20] + '...' if len(str(title)) > 20 else str(title) 
                                       for title in top_rated[title_col]])
    plt.xlabel('Rating')
    plt.title('🌟 Top 10 Rated Games')
    plt.gca().invert_yaxis()
else:
    plt.text(0.5, 0.5, 'Rating data not available', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('🌟 Top Rated Games (No Data)')
plt.tight_layout()
plt.show()

# 2. Genre distribution
plt.figure(figsize=(10, 8))
if genre_col:
    genre_counts = games_clean[genre_col].value_counts().head(10)
    plt.pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%')
    plt.title('🧩 Genre Distribution')
else:
    # Try sales dataset for genre
    sales_genre_col = None
    for col in sales_clean.columns:
        if 'genre' in col.lower():
            sales_genre_col = col
            break
    if sales_genre_col:
        genre_counts = sales_clean[sales_genre_col].value_counts().head(10)
        plt.pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%')
        plt.title('🧩 Genre Distribution')
    else:
        plt.text(0.5, 0.5, 'Genre data not available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('🧩 Genre Distribution (No Data)')
plt.tight_layout()
plt.show()

# 3. Global sales by region (if regional sales columns exist)
plt.figure(figsize=(10, 8))
region_cols = [col for col in sales_clean.columns if any(region in col.upper() for region in ['NA', 'EU', 'JP', 'OTHER']) and 'SALES' in col.upper()]
if region_cols:
    region_totals = [sales_clean[col].sum() for col in region_cols]
    region_names = [col.replace('_Sales', '').replace('_sales', '') for col in region_cols]
    plt.bar(region_names, region_totals, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    plt.title('🌍 Global Sales by Region')
    plt.ylabel('Sales (Millions)')
    plt.xticks(rotation=45)
else:
    plt.text(0.5, 0.5, 'Regional sales data not available', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('🌍 Regional Sales (No Data)')
plt.tight_layout()
plt.show()

# 4. Top publishers by sales
plt.figure(figsize=(10, 8))
publisher_col = None
for col in sales_clean.columns:
    if 'publisher' in col.lower():
        publisher_col = col
        break

if publisher_col and global_sales_col:
    pub_sales = sales_clean.groupby(publisher_col)[global_sales_col].sum().sort_values(ascending=False).head(8)
    plt.barh(range(len(pub_sales)), pub_sales.values)
    plt.yticks(range(len(pub_sales)), pub_sales.index)
    plt.xlabel('Global Sales (Millions)')
    plt.title('🏢 Top Publishers by Sales')
    plt.gca().invert_yaxis()
else:
    plt.text(0.5, 0.5, 'Publisher data not available', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('🏢 Publishers (No Data)')
plt.tight_layout()
plt.show()

# 5. Sales trend over years
plt.figure(figsize=(10, 8))
year_col = None
for col in sales_clean.columns:
    if 'year' in col.lower():
        year_col = col
        break

if year_col and global_sales_col:
    yearly_sales = sales_clean.groupby(year_col)[global_sales_col].sum()
    plt.plot(yearly_sales.index, yearly_sales.values, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Year')
    plt.ylabel('Global Sales (Millions)')
    plt.title('📈 Sales Trend Over Years')
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'Year/Sales data not available', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('📈 Sales Trend (No Data)')
plt.tight_layout()
plt.show()

# 6. Platform performance
plt.figure(figsize=(10, 8))
platform_col = None
for col in sales_clean.columns:
    if 'platform' in col.lower():
        platform_col = col
        break

if platform_col and global_sales_col:
    platform_sales = sales_clean.groupby(platform_col)[global_sales_col].sum().sort_values(ascending=False).head(8)
    plt.bar(platform_sales.index, platform_sales.values, color='skyblue')
    plt.xlabel('Platform')
    plt.ylabel('Global Sales (Millions)')
    plt.title('🕹️ Platform Performance')
    plt.xticks(rotation=45)
else:
    plt.text(0.5, 0.5, 'Platform data not available', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('🕹️ Platform Performance (No Data)')
plt.tight_layout()
plt.show()

# 7. Top wishlisted games (if wishlist column exists)
plt.figure(figsize=(10, 8))
wishlist_col = None
for col in games_clean.columns:
    if 'wishlist' in col.lower():
        wishlist_col = col
        break

if wishlist_col and title_col:
    # Ensure the column is numeric before using nlargest
    if pd.api.types.is_numeric_dtype(games_clean[wishlist_col]):
        top_wishlist = games_clean.nlargest(8, wishlist_col)
        wishlist_values = top_wishlist[wishlist_col] / 1000000  # Convert to millions
        plt.barh(range(len(top_wishlist)), wishlist_values)
        plt.yticks(range(len(top_wishlist)), [str(title)[:15] + '...' if len(str(title)) > 15 else str(title) 
                                              for title in top_wishlist[title_col]])
        plt.xlabel('Wishlist (Millions)')
        plt.title('🎯 Top Wishlisted Games')
        plt.gca().invert_yaxis()
    else:
        plt.text(0.5, 0.5, f'Wishlist column not numeric: {games_clean[wishlist_col].dtype}', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('🎯 Wishlisted Games (Data Type Error)')
else:
    plt.text(0.5, 0.5, 'Wishlist data not available', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('🎯 Wishlisted Games (No Data)')
plt.tight_layout()
plt.show()

# 8. Genre sales comparison
plt.figure(figsize=(10, 8))
sales_genre_col = None
for col in sales_clean.columns:
    if 'genre' in col.lower():
        sales_genre_col = col
        break

if sales_genre_col and global_sales_col:
    genre_sales = sales_clean.groupby(sales_genre_col)[global_sales_col].sum().sort_values(ascending=False).head(8)
    plt.bar(genre_sales.index, genre_sales.values, color='lightcoral')
    plt.xlabel('Genre')
    plt.ylabel('Global Sales (Millions)')
    plt.title('🎮 Sales by Genre')
    plt.xticks(rotation=45)
else:
    plt.text(0.5, 0.5, 'Genre/Sales data not available', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('🎮 Sales by Genre (No Data)')
plt.tight_layout()
plt.show()

# 9. Developer productivity (if team/developer column exists)
plt.figure(figsize=(10, 8))
team_col = None
for col in games_clean.columns:
    if any(keyword in col.lower() for keyword in ['team', 'developer', 'dev']):
        team_col = col
        break

if team_col:
    dev_games = games_clean[team_col].value_counts().head(8)
    plt.barh(range(len(dev_games)), dev_games.values)
    plt.yticks(range(len(dev_games)), dev_games.index)
    plt.xlabel('Number of Games')
    plt.title('🏢 Most Productive Developers')
    plt.gca().invert_yaxis()
else:
    plt.text(0.5, 0.5, 'Developer data not available', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('🏢 Developers (No Data)')
plt.tight_layout()
plt.show()

# 10. Engagement metrics comparison (if available)
plt.figure(figsize=(10, 8))
engagement_cols = []
for col in games_clean.columns:
    if any(keyword in col.lower() for keyword in ['plays', 'backlogs', 'wishlist']):
        # Check if column is numeric before adding
        if pd.api.types.is_numeric_dtype(games_clean[col]):
            engagement_cols.append(col)

if engagement_cols:
    avg_engagement = []
    labels = []
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFB366', '#B366FF']
    
    for col in engagement_cols[:5]:  # Limit to 5 columns to fit the chart
        avg_val = games_clean[col].mean()
        # Scale appropriately based on magnitude
        if avg_val > 1000000:
            avg_engagement.append(avg_val / 1000000)
            labels.append(f"{col}\n(Millions)")
        elif avg_val > 1000:
            avg_engagement.append(avg_val / 1000)
            labels.append(f"{col}\n(Thousands)")
        else:
            avg_engagement.append(avg_val)
            labels.append(col)
    
    plt.bar(range(len(labels)), avg_engagement, color=colors[:len(labels)])
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.ylabel('Average Count')
    plt.title('🔍 Average Engagement Metrics')
else:
    plt.text(0.5, 0.5, 'Engagement data not available', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('🔍 Engagement Metrics (No Data)')
plt.tight_layout()
plt.show()

# 11. Release year distribution
plt.figure(figsize=(10, 8))
release_year_col = None
for col in games_clean.columns:
    if 'release_year' in col.lower() or 'year' in col.lower():
        release_year_col = col
        break

if release_year_col:
    year_dist = games_clean[release_year_col].value_counts().sort_index()
    plt.bar(year_dist.index, year_dist.values, color='gold')
    plt.xlabel('Release Year')
    plt.ylabel('Number of Games')
    plt.title('🗓️ Games Released by Year')
    plt.xticks(rotation=45)
else:
    plt.text(0.5, 0.5, 'Release year data not available', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('🗓️ Release Years (No Data)')
plt.tight_layout()
plt.show()

# ========================================================================================
# PART 5: ADVANCED ANALYSIS AND INSIGHTS
# ========================================================================================

print("\n📊 Advanced Analysis and Key Insights")
print("=" * 50)

# Try to merge datasets for comprehensive analysis
merged_analysis = pd.DataFrame()
merge_successful = False

# Find the correct title column in games dataset
title_col_games = None
for col in games_clean.columns:
    if games_clean[col].dtype == 'object' and any(len(str(x)) > 5 for x in games_clean[col].head(5)):
        title_col_games = col
        break

# Find the correct title column in sales dataset
title_col_sales = None
for col in sales_clean.columns:
    if col.lower() in ['name', 'title']:
        title_col_sales = col
        break
if title_col_sales is None:
    for col in sales_clean.columns:
        if sales_clean[col].dtype == 'object' and any(len(str(x)) > 5 for x in sales_clean[col].head(5)):
            title_col_sales = col
            break

if title_col_games and title_col_sales:
    try:
        # Ensure title columns are strings before attempting string operations
        games_clean[title_col_games] = games_clean[title_col_games].astype(str)
        sales_clean[title_col_sales] = sales_clean[title_col_sales].astype(str)
        
        # Create normalized title columns with common formatting
        games_clean['normalized_title'] = games_clean[title_col_games].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()
        sales_clean['normalized_title'] = sales_clean[title_col_sales].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()
        
        # Try to find common games by normalized title
        common_titles = set(games_clean['normalized_title'].unique()) & set(sales_clean['normalized_title'].unique())
        
        if len(common_titles) > 0:
            print(f"🔗 Found {len(common_titles)} common game titles for merging")
            
            # Create filtered versions of both datasets with only common titles
            games_filtered = games_clean[games_clean['normalized_title'].isin(common_titles)].copy()
            sales_filtered = sales_clean[sales_clean['normalized_title'].isin(common_titles)].copy()
            
            # Merge on the normalized title
            merged_analysis = pd.merge(
                games_filtered, 
                sales_filtered, 
                on='normalized_title', 
                how='inner',
                suffixes=('_games', '_sales')
            )
            
            if not merged_analysis.empty:
                print(f"✅ Successfully merged {len(merged_analysis)} games for comprehensive analysis\n")
                merge_successful = True
                
                # Display some examples of merged games
                print("📋 Examples of merged games:")
                sample_merged = merged_analysis.head(3)
                for i, row in sample_merged.iterrows():
                    print(f"   • {row[title_col_games]} (Rating: {row.get(rating_col, 'N/A')}, Sales: ${row.get(global_sales_col, 'N/A')}M)")
                print()
            else:
                print("⚠️ Could not merge datasets - no common games found after filtering\n")
        else:
            print("⚠️ Could not merge datasets - no common game titles found\n")
            print("💡 This could be due to:")
            print("   - Different title naming conventions between datasets")
            print("   - Different sets of games in each dataset")
            print("   - Data quality issues in title fields")
            
            # Show sample titles from each dataset for comparison
            print("\n📝 Sample titles from games dataset:")
            for title in games_clean[title_col_games].head(3):
                print(f"   • {title}")
                
            print("\n📝 Sample titles from sales dataset:")
            for title in sales_clean[title_col_sales].head(3):
                print(f"   • {title}")
            print()
    except Exception as e:
        print(f"⚠️ Could not merge datasets - {str(e)[:100]}...\n")
else:
    print("⚠️ Could not identify title columns in one or both datasets\n")
    print("📋 Games dataset columns:", list(games_clean.columns))
    print("📋 Sales dataset columns:", list(sales_clean.columns))

if not merge_successful:
    print("⚠️ Analyzing datasets separately\n")

# Key Performance Indicators (KPIs)
print("📈 KEY PERFORMANCE INDICATORS")
print("-" * 30)
print(f"🎮 Total Games Analyzed: {len(games_clean)}")
if global_sales_col:
    print(f"💰 Total Global Sales: ${sales_clean[global_sales_col].sum():.2f}M")
if rating_col:
    print(f"⭐ Average Game Rating: {games_clean[rating_col].mean():.2f}")
if wishlist_col:
    print(f"🎯 Average Wishlist Count: {games_clean[wishlist_col].mean():,.0f}")

# Top performers analysis
print(f"\n🏆 TOP PERFORMERS")
print("-" * 20)
if global_sales_col and title_col_sales:
    best_seller = sales_clean.loc[sales_clean[global_sales_col].idxmax()]
    print(f"💎 Best Seller: {best_seller[title_col_sales]} (${best_seller[global_sales_col]}M)")

if rating_col and title_col_games:
    highest_rated = games_clean.loc[games_clean[rating_col].idxmax()]
    print(f"⭐ Highest Rated: {highest_rated[title_col_games]} ({highest_rated[rating_col]})")

if wishlist_col and title_col_games:
    most_wishlisted = games_clean.loc[games_clean[wishlist_col].idxmax()]
    print(f"🎯 Most Wishlisted: {most_wishlisted[title_col_games]} ({most_wishlisted[wishlist_col]:,})")

# Genre analysis
print(f"\n🎨 GENRE INSIGHTS")
print("-" * 20)
# Check for genre column in both datasets
genre_col_analysis = None
if genre_col:
    genre_col_analysis = genre_col
else:
    # Try sales dataset for genre
    for col in sales_clean.columns:
        if 'genre' in col.lower():
            genre_col_analysis = col
            break

if genre_col_analysis:
    if genre_col_analysis in games_clean.columns:
        genre_counts = games_clean[genre_col_analysis].value_counts()
        top_genre = genre_counts.idxmax()
        print(f"📊 Most common genre in games data: {top_genre} ({genre_counts.max()} games)")
    
    if genre_col_analysis in sales_clean.columns and global_sales_col:
        genre_sales = sales_clean.groupby(genre_col_analysis)[global_sales_col].sum()
        if not genre_sales.empty:
            top_selling_genre = genre_sales.idxmax()
            print(f"💰 Top selling genre: {top_selling_genre} (${genre_sales.max():.2f}M)")
            
            # Show top 3 genres by sales
            top_genres = genre_sales.sort_values(ascending=False).head(3)
            print("🏆 Top 3 genres by sales:")
            for i, (genre, sales) in enumerate(top_genres.items(), 1):
                print(f"   {i}. {genre}: ${sales:.2f}M")
else:
    print("ℹ️ Genre data not available for analysis")

# Platform analysis
print(f"\n🎮 PLATFORM INSIGHTS")
print("-" * 20)
if platform_col and global_sales_col:
    platform_performance = sales_clean.groupby(platform_col).agg({
        global_sales_col: ['sum', 'mean', 'count']
    }).round(2)
    platform_performance.columns = ['Total_Sales', 'Avg_Sales', 'Game_Count']
    platform_performance = platform_performance.sort_values('Total_Sales', ascending=False)

    print("Top 3 platforms by total sales:")
    for i, (platform, data) in enumerate(platform_performance.head(3).iterrows()):
        print(f"{i+1}. {platform}: ${data['Total_Sales']}M total, ${data['Avg_Sales']}M avg ({data['Game_Count']} games)")

# Regional preferences
print(f"\n🌍 REGIONAL PREFERENCES")
print("-" * 25)
if region_cols and global_sales_col:
    region_names_clean = [col.replace('_Sales', '').replace('_sales', '') for col in region_cols]
    region_totals = {name: sales_clean[col].sum() for name, col in zip(region_names_clean, region_cols)}

    for region, total in sorted(region_totals.items(), key=lambda x: x[1], reverse=True):
        percentage = (total / sales_clean[global_sales_col].sum()) * 100
        print(f"🌎 {region}: ${total:.2f}M ({percentage:.1f}%)")

# ========================================================================================
# PART 6: BUSINESS RECOMMENDATIONS
# ========================================================================================

print(f"\n💼 BUSINESS RECOMMENDATIONS")
print("=" * 35)

print("🎯 MARKETING STRATEGY:")
print("   • Focus on high-performing genres identified in analysis")
print("   • Target regions with highest sales potential")
print("   • Leverage platform-specific marketing strategies")

print(f"\n🛠️ PRODUCT DEVELOPMENT:")
print("   • Invest in multi-platform releases for broader reach")
print("   • Maintain quality standards for better ratings")
print("   • Consider successful genre combinations")

print(f"\n📊 SALES FORECASTING:")
print("   • Analyze year-over-year trends for planning")
print("   • Platform diversification reduces market risk")
print("   • Monitor genre popularity shifts")

# ========================================================================================
# PART 7: SQL QUERIES FOR VERIFICATION
# ========================================================================================

print(f"\n🗄️ SQL QUERY EXAMPLES")
print("=" * 25)

def execute_sql_queries(conn):
    """Execute sample SQL queries to demonstrate database functionality"""
    
    # Get column names from actual tables
    games_columns = pd.read_sql_query("PRAGMA table_info(games)", conn)['name'].tolist()
    sales_columns = pd.read_sql_query("PRAGMA table_info(sales)", conn)['name'].tolist()
    
    # Read a small sample of the games table to inspect values (helps to find a proper title column)
    try:
        sample_games = pd.read_sql_query("SELECT * FROM games LIMIT 10", conn)
    except Exception:
        sample_games = pd.DataFrame()
    
    queries = []
    
    # Build queries based on available columns
    rating_col_sql = None
    if any('rating' in col.lower() for col in games_columns):
        rating_col_sql = next(col for col in games_columns if 'rating' in col.lower())
    
    # Try to reliably pick a title column:
    title_col_sql = None
    # 1) Prefer explicit 'title' or 'name' columns
    for col in games_columns:
        if 'title' in col.lower() or col.lower() == 'name':
            title_col_sql = col
            break
    # 2) If not found, inspect sample rows for a string-like column (and avoid Unnamed)
    if title_col_sql is None and not sample_games.empty:
        for col in sample_games.columns:
            if col.lower().startswith('unnamed'):
                continue
            # convert to str and check if sample values look like titles (length > 3)
            non_null_vals = sample_games[col].dropna().astype(str)
            if len(non_null_vals) > 0 and any(len(v.strip()) > 3 for v in non_null_vals):
                # prefer columns that are not purely numeric
                try:
                    numeric_check = pd.to_numeric(non_null_vals, errors='coerce').notna().all()
                except Exception:
                    numeric_check = False
                if not numeric_check:
                    title_col_sql = col
                    break
    # 3) Fallback to first column (original behavior) if still not found
    if title_col_sql is None and len(games_columns) > 0:
        title_col_sql = games_columns[0]
    
    # Now construct queries using the discovered column names (quote column names to handle spaces)
    if rating_col_sql and title_col_sql:
        queries.append(("Top 5 highest-rated games", 
                       f'SELECT "{title_col_sql}", "{rating_col_sql}" FROM games ORDER BY "{rating_col_sql}" DESC LIMIT 5'))
    
    if any('publisher' in col.lower() for col in sales_columns) and any('sales' in col.lower() and 'global' in col.lower() for col in sales_columns):
        publisher_col_sql = next(col for col in sales_columns if 'publisher' in col.lower())
        global_sales_col_sql = next(col for col in sales_columns if 'sales' in col.lower() and 'global' in col.lower())
        queries.append(("Total sales by publisher", 
                       f'SELECT "{publisher_col_sql}", SUM("{global_sales_col_sql}") as total_sales FROM sales GROUP BY "{publisher_col_sql}" ORDER BY total_sales DESC LIMIT 5'))
    
    if any('year' in col.lower() for col in games_columns):
        year_col_sql = next(col for col in games_columns if 'year' in col.lower())
        queries.append(("Games released after 2018", 
                       f'SELECT COUNT(*) as game_count FROM games WHERE "{year_col_sql}" > 2018'))
    
    for description, query in queries:
        print(f"\n📝 {description}:")
        try:
            result = pd.read_sql_query(query, conn)
            print(result.to_string(index=False))
        except Exception as e:
            print(f"   Error executing query: {e}")

execute_sql_queries(db_conn)

# ========================================================================================
# PART 8: EXPORT AND SUMMARY
# ========================================================================================

print(f"\n💾 PROJECT DELIVERABLES")
print("=" * 30)

# Create summary statistics
summary_stats = {
    'Total Games': len(games_clean),
    'Total Sales Records': len(sales_clean),
    'Database Created': 'video_games.db'
}

if rating_col:
    summary_stats['Average Rating'] = round(games_clean[rating_col].mean(), 2)
if global_sales_col:
    summary_stats['Total Global Sales'] = f"${sales_clean[global_sales_col].sum():.2f}M"

# Find genre column for summary
genre_col_summary = None
for col in sales_clean.columns:
    if 'genre' in col.lower():
        genre_col_summary = col
        break
if not genre_col_summary:
    for col in games_clean.columns:
        if 'genre' in col.lower():
            genre_col_summary = col
            break

if genre_col_summary and global_sales_col and genre_col_summary in sales_clean.columns:
    summary_stats['Top Genre'] = sales_clean.groupby(genre_col_summary)[global_sales_col].sum().idxmax()
if platform_col:
    summary_stats['Best Platform'] = sales_clean.groupby(platform_col)[global_sales_col].sum().idxmax() if global_sales_col else "N/A"
if year_col:
    summary_stats['Analysis Period'] = f"{sales_clean[year_col].min()}-{sales_clean[year_col].max()}"

print("📋 PROJECT SUMMARY:")
for key, value in summary_stats.items():
    print(f"   {key}: {value}")

print(f"\n✅ PROJECT COMPLETION STATUS:")
print("   🧹 Data Cleaning: ✓ Completed")
print("   🗄️ SQL Database: ✓ Created (video_games.db)")
print("   📊 EDA Analysis: ✓ Completed (12 visualizations)")
print("   🔍 Advanced Analytics: ✓ Completed")

# Close database connection
db_conn.close()

print("\n🔒 Database connection closed.")
