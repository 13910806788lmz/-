import time
import json
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import numpy as np
from wordcloud import WordCloud
from warnings import filterwarnings
from concat import df_10G, df_30G
filterwarnings('ignore')

# ================== 基础设置 ==================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ================== 记录并打印耗时 ==================
def log_time(task_name, start_time):
    elapsed = time.time() - start_time
    print(f"[{task_name}] 耗时: {elapsed:.2f}秒 | 内存使用: {psutil.virtual_memory().percent}%")
    return time.time()

# ================== 数据加载 ==================
def load_data(df_):
    start = time.time()

    print(f"\n{'='*40}\n正在加载数据: ")
    df = df_
    
    #print("数据概览：", df.info())
    print(f"初始数据量: {len(df):,} 条")
    print(f"内存占用: {df.memory_usage(deep=True).sum()/1024**3:.2f} GB")
    log_time("数据加载", start)
    return df

# ================== 数据匿名化处理 ==================
def anonymize_data(df):
    sensitive_cols = ['user_name', 'chinese_name', 'email', 'phone_number']
    df = df.drop(columns=[col for col in sensitive_cols if col in df.columns])
    return df

# ================== 时间格式转换与特征提取处理 ==================
def process_time_features(df):
    time_cols = ['timestamp', 'registration_date']
    for col in time_cols:
        if col in df.columns:
            # 转换为datetime类型
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # 提取时间特征
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
    return df

# ================== 数据预处理 ==================
def preprocess(df, dataset_name):
    start = time.time()
    print(f"\n{'='*40}\n{dataset_name}预处理开始")
    
    # 缺失值分析
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_report = pd.concat([missing, missing_pct], axis=1)
    missing_report.columns = ['缺失数量', '缺失比例(%)']
    print("\n[缺失值统计]")
    print(missing_report)

    # 重复值分析
    print("\n[重复值统计]", df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    print('[删除重复值后数据量统计]', len(df)) 
    
    # age异常值处理
    if 'age' in df.columns:
        df = df[(df['age'] >= 18) & (df['age'] <= 100)]  
        # 分位数分组
        bins = [18, 25, 35, 45, 55, 100]
        labels = ['18-25', '26-35', '36-45', '46-55', '55+']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)    

    # income异常值处理    
    if 'income' in df.columns:
        # 过滤负值
        df = df[df['income'] >= 0]
        # 分位数分组
        df['income_level'] = pd.qcut(
            df['income'], 
            q=4, 
            labels=['低', '中低', '中高', '高']
        )
   
    # gender异常值处理 存在‘未指定’‘其他’
    if 'gender' in df.columns:
        # 统一格式
        df['gender'] = df['gender'].str.strip().replace({
            'male': '男', 
            'female': '女',
            'm': '男',
            'f': '女'
        })
        
        # 处理异常值
        valid_genders = ['男', '女']
        df['gender'] = df['gender'].where(
            df['gender'].isin(valid_genders), 
            '其他'
        )
        # df = df[(df['gender'] == '男') | (df['gender'] == '女')]
    
    # chinese_address异常值处理 
    if 'chinese_address' in df.columns:
        # 提取省份（示例：匹配首个省级行政区）
        province_pattern = r'(北京|上海|天津|重庆|河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾|内蒙古|广西|西藏|宁夏|新疆|香港|澳门)'
        df['province'] = df['chinese_address'].str.extract(province_pattern)

    # purchase_history异常值处理 
    if 'purchase_history' in df.columns:
        def parse_record(record):
            try:
                fixed = record.replace('}{', '},{')
                return json.loads(f'[{fixed}]')
            except:
                return np.nan
        # 解析JSON
        df['parsed_purchases'] = df['purchase_history'].apply(parse_record)
        # 提取消费金额
        df['purchase_amounts'] = df['parsed_purchases'].apply(
            lambda x: [round(float(item.get('average_price', 0)), 2) for item in x]
            if isinstance(x, list) else []
        )
    
    # is_active异常值处理
    if 'is_active' in df.columns:
        # 转换为布尔类型
        df['is_active'] = df['is_active'].astype(bool)
    
    # credit_score异常值处理
    if 'credit_score' in df.columns:
        # 过滤不合理范围（假设0-1000）
        df = df[(df['credit_score'] >= 0) & (df['credit_score'] <= 1000)]
        bins = [0, 350, 500, 700, 1000]
        labels = ['差', '中', '良', '优']
        df['credit_distribution'] = pd.cut(df['credit_score'], bins=bins, labels=labels, include_lowest=True)
     
    log_time("预处理完成", start)
    return df

# ================== 可视化分析 ==================
def visualize(df, dataset_name):
    def visualize_age_distribution(df):
        """年龄分布可视化"""
        plt.figure(figsize=(10, 6))
        sns.histplot(df['age'], bins=20, kde=True, color='skyblue')
        plt.title('用户年龄分布')
        plt.xlabel('年龄')
        plt.ylabel('用户数量')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('visualize_age_distribution.png')
        # plt.show()

    def visualize_gender_distribution(df):
        """性别分布可视化"""
        gender_counts = df['gender'].value_counts()
        
        plt.figure(figsize=(10, 6))
        # 饼图
        plt.subplot(1, 2, 1)
        gender_counts.plot.pie(autopct='%1.11f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
        plt.title('性别比例')
        plt.ylabel('')
        
        # 柱状图
        plt.subplot(1, 2, 2)
        sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='pastel')
        plt.title('性别分布')
        plt.xlabel('性别')
        plt.ylabel('用户数量')
        
        plt.tight_layout()
        plt.savefig('visualize_gender_distribution.png')
        # plt.show()

    def visualize_income_analysis(df):
        """收入分析可视化"""
        plt.figure(figsize=(12, 6))
        
        # 箱线图
        plt.subplot(1, 2, 1)
        sns.boxplot(y=df['income'], color='lightgreen')
        plt.title('收入分布箱线图')
        plt.ylabel('收入（元）')
        
        # 分箱柱状图
        plt.subplot(1, 2, 2)
        sns.countplot(x='income_level', data=df, order=['低','中低','中高','高'], palette='Blues')
        plt.title('收入等级分布')
        plt.xlabel('收入等级')
        plt.ylabel('用户数量')
        
        plt.tight_layout()
        plt.savefig('visualize_income_analysis.png')
        # plt.show()

    def visualize_geo_distribution(df):
        """地理分布可视化"""
        plt.figure(figsize=(12, 6))
        
        # 国家分布
        plt.subplot(1, 2, 1)
        top_countries = df['country'].value_counts().head(5)
        sns.barplot(y=top_countries.index, x=top_countries.values, palette='viridis')
        plt.title('Top 5 国家分布')
        plt.xlabel('用户数量')
        
        # 省份分布
        plt.subplot(1, 2, 2)
        top_provinces = df['province'].value_counts().head(5)
        sns.barplot(y=top_provinces.index, x=top_provinces.values, palette='magma')
        plt.title('Top 5 省份分布')
        plt.xlabel('用户数量')
        
        plt.tight_layout()
        plt.savefig('visualize_geo_distribution.png')
        # plt.show()

    def visualize_purchase_behavior(df):
        """消费行为可视化"""
        plt.figure(figsize=(16, 12))
        
        # 消费金额分布
        plt.subplot(2, 2, 1)
        sns.histplot(df['purchase_amounts'].explode().astype(float), bins=30, kde=True, color='purple')
        plt.title('单次消费金额分布')
        plt.xlabel('消费金额（元）')
        
        # 消费频率分布
        plt.subplot(2, 2, 2)
        purchase_counts = df['purchase_amounts'].apply(len)
        sns.histplot(purchase_counts, bins=15, kde=True, color='orange')
        plt.title('用户消费次数分布')
        plt.xlabel('消费次数')
        
        # 消费时间热力图
        plt.subplot(2, 2, 3)
        df['purchase_hour'] = df['timestamp'].dt.hour
        hour_counts = df['purchase_hour'].value_counts().sort_index()
        sns.heatmap(hour_counts.values.reshape(1, -1), annot=True, fmt="d", cmap='YlGnBu', cbar=False)
        plt.title('消费时段分布')
        plt.xlabel('小时(0-23)')
        plt.yticks([])
        
        # 消费类别词云
        plt.subplot(2, 2, 4)
        categories = df['parsed_purchases'].apply(
            lambda x: [item.get('category') for item in x] 
            if isinstance(x, list) else []
        ).explode().value_counts().head(5).index.tolist()

        text = ' '.join(categories)
        wordcloud = WordCloud(width=800, height=400, background_color='white', font_path='msyh.ttc').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('消费类别词云')
        
        plt.tight_layout()
        plt.savefig('visualize_purchase_behavior.png')
        # plt.show()

    def visualize_activity_analysis(df):
        """活跃度分析可视化"""
        plt.figure(figsize=(10, 6))
        
        # 活跃用户占比
        active_ratio = df['is_active'].mean()
        labels = ['活跃用户', '非活跃用户']
        sizes = [active_ratio, 1-active_ratio]
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66ff66','#ff6666'])
        plt.title('用户活跃度比例')
        plt.savefig('visualize_activity_analysis.png')
        # plt.show()

    def visualize_credit_analysis(df):
        """信用评分可视化"""
        plt.figure(figsize=(12, 6))
        
        # 信用评分分布
        plt.subplot(1, 2, 1)
        sns.histplot(df['credit_score'], bins=20, kde=True, color='teal')
        plt.title('信用评分分布')
        plt.xlabel('信用评分')
        
        # 信用等级分布
        plt.subplot(1, 2, 2)
        sns.countplot(x='credit_distribution', data=df, order=['差','中','良','优'], palette='RdYlGn')
        plt.title('信用等级分布')
        plt.xlabel('信用等级')
        plt.ylabel('用户数量')
        plt.tight_layout()
        plt.savefig('visualize_credit_analysis.png')
        # plt.show()

    print('\n可视化分析', dataset_name)
    visualize_age_distribution(df)
    visualize_gender_distribution(df)
    visualize_income_analysis(df)
    visualize_geo_distribution(df)
    visualize_purchase_behavior(df)
    visualize_activity_analysis(df)
    visualize_credit_analysis(df)

# ================== 用户画像分析 ==================
def user_profiling(df):
    print(f"\n{'='*40}\n构建用户画像") 
    start = time.time()
    
    # 展平所有消费金额
    all_amounts = df['purchase_amounts'].explode()
    valid_amounts = pd.to_numeric(all_amounts, errors='coerce').dropna()

    profile = {
        'median_age': df['age'].median(),
        'age_distribution': df['age_group'].value_counts(normalize=True).to_dict(),
        'gender_dist': df['gender'].value_counts(normalize=True).to_dict(), 
        'top_countries': df['country'].value_counts().head(5).index.tolist(),
        'top_provinces': df['province'].value_counts().head(5).index.tolist(),
        'avg_income': df['income'].mean(),
        'income_distribution': df['income_level'].value_counts(normalize=True).to_dict(),
        'avg_purchase': valid_amounts.mean(),
        'purchase_freq': len(valid_amounts) / len(df),
        'top_categories': df['parsed_purchases'].apply(
            lambda x: [item.get('category') for item in x] 
            if isinstance(x, list) else []
        ).explode().value_counts().head(5).index.tolist(),
        'active_ratio': df['is_active'].mean(),
        'avg_credit': df['credit_score'].mean(),
        'credit_distribution': df['credit_distribution'].value_counts(normalize=True).to_dict(),
    }

    print(f"- 年龄中位数: {profile['median_age']:.4f}岁")
    print(f"- 年龄分布: {{{', '.join(f'{k}: {v:.4f}' for k, v in profile['age_distribution'].items())}}}")
    print(f"- 性别分布: {{{', '.join(f'{k}: {v:.4f}' for k, v in profile['gender_dist'].items())}}}")
    print(f"- 高频国家: {', '.join(profile['top_countries'])}")
    print(f"- 高频城市: {', '.join(profile['top_provinces'])}")
    print(f"- 收入平均水平: {profile['avg_income']:.4f}元")
    print(f"- 收入分布: {{{', '.join(f'{k}: {v:.4f}' for k, v in profile['income_distribution'].items())}}}")
    print(f"- 消费平均水平: {profile['avg_purchase']:.4f}元")
    print(f"- 消费频率: {profile['purchase_freq']:.4f}")
    print(f"- 高频消费产品: {profile['top_categories']}")
    print(f"- 活跃度平均水平: {profile['active_ratio']:.4f}")
    print(f"- 信誉度平均水平: {profile['avg_credit']:.4f}分")
    print(f"- 信誉等级分布: {{{', '.join(f'{k}: {v:.4f}' for k, v in profile['credit_distribution'].items())}}}")
   
    log_time("\n画像分析完成", start)
    return profile

if __name__ == "__main__":
    total_start = time.time()

    #df_test = df_30G
    df_test = df_10G
    df = load_data(df_test)
    df = anonymize_data(df)
    df = process_time_features(df)   
    df_clean = preprocess(df, "10G数据集")
    visualize(df_clean, "10G数据集")
    profile_1g = user_profiling(df_clean)
    total_elapsed = time.time() - total_start
    print(f"\n{'='*40}\n总耗时: {total_elapsed/60:.2f}分钟")
