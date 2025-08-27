import pandas as pd
import numpy as np
import os
from functools import wraps

# --- 에러 처리 데코레이터 ---
def handle_file_processing_errors(func):
    """
    파일을 읽고 데이터프레임으로 만들어 함수에 전달하는 데코레이터
    """
    @wraps(func)
    def wrapper(filepaths, *args, **kwargs):
        if not isinstance(filepaths, list):
            filepaths = [filepaths]
        
        df_list = []
        try:
            for filepath in filepaths:
                filepath = os.path.normpath(filepath)
                try:
                    print(f"🔄 '{os.path.basename(filepath)}' 파일을 읽습니다 (인코딩: utf-8)...")
                    df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
                    df_list.append(df)
                except UnicodeDecodeError:
                    print(f"⚠️ utf-8 읽기 실패. cp949 인코딩으로 다시 시도합니다...")
                    df = pd.read_csv(filepath, encoding='cp949', low_memory=False)
                    df_list.append(df)
            
            df_raw = pd.concat(df_list, ignore_index=True) if len(df_list) > 1 else df_list[0]
            return func(df_raw, *args, **kwargs)

        except FileNotFoundError as e:
            print(f"⚠️  파일 없음: {e.filename} 파일을 찾을 수 없습니다. 경로를 확인하세요.")
            return None
        except Exception as e:
            print(f"❌ 오류: '{func.__name__}' 함수 실행 중 오류가 발생했습니다: {e}")
            return None
    return wrapper

# --- 1. 소비액 성장률 계산 함수 ---
@handle_file_processing_errors
def calculate_spend_growth_rate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    지역별 지출액 데이터로부터 성장률을 계산하고 전체 및 상위 20개 지역을 반환
    """
    
    df.rename(columns={
        '광역지자체 명': 'sido_name', 
        '기초지자체 명': 'sigungu_name', 
        '기초지자체 소비액': 'current_consume', 
        '전년동기 기초지자체 소비액': 'prev_consume'
        }, inplace=True)
    
    df['region_id'] = df['sido_name'] + ' ' + df['sigungu_name']
    for col in ['current_consume', 'prev_consume']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df['growth_rate'] = (df['current_consume'] - df['prev_consume']) / df['prev_consume'].replace(0, np.nan)
    result_df = df[['region_id', 'growth_rate']].dropna().reset_index(drop=True)
    
    # 상위 20개 지역을 별도의 데이터프레임으로 생성
    top_20_growth = result_df.sort_values(by='growth_rate', ascending=False).head(20)

    return result_df, top_20_growth

# --- 2. 지역 기반 관광 정보 분석 ---
@handle_file_processing_errors
def analyze_tourism_resources(df: pd.DataFrame) -> pd.DataFrame:
    """
    분리된 관광정보 파일을 종합 및 전처리
    """

    if 'sigungucode' not in df.columns: df['sigungucode'] = ''
    df['sigungucode'] = df['sigungucode'].fillna('').astype(str)
    df['areacode'] = df['areacode'].fillna('').astype(str)
    df['region_id_code'] = df.apply(lambda r: f"{r['areacode']}-{r['sigungucode']}" if r['sigungucode'] else r['areacode'], axis=1)
    df['region_id'] = df['광역지역명'] + ' ' + df['시군구명']
    
    resource_counts = df.groupby(['region_id', 'region_id_code', 'lclsSystm1','lclsSystm2','lclsSystm3']).size().reset_index(name='count')
    total_counts = resource_counts.groupby('region_id')['count'].sum().to_dict()
    resource_counts['total_count'] = resource_counts['region_id'].map(total_counts)
    resource_counts['proportion'] = (resource_counts['count'] / resource_counts['total_count']) * 100
    resource_counts['proportion'] = resource_counts['proportion'].round(3)
    

    return resource_counts

# --- 3. 전국 문화 축제 데이터 전처리 ---
@handle_file_processing_errors
def calculate_festival_density(df: pd.DataFrame) -> pd.DataFrame:
    """
    지정된 기간을 기준으로 전국 모든 지역의 축제 밀도와 축제 목록을 계산
    """

    def get_region_from_address(address):
        # 주소에서 시도 및 시군구 추출
        if isinstance(address, str):
            parts = address.split()
            if len(parts) >= 2:
                sido = parts[0].replace("특별자치도", "도").replace("광역시", "시").replace("특별자치시", "시")
                return f"{sido} {parts[1]}"
        return None
    
    # 필요한 열만 선택 및 이름 매핑
    COLUMN_MAP = {
        '축제명': 'festival_name', 
        '축제시작일자': 'start_date', 
        '축제종료일자': 'end_date', 
        '소재지도로명주소': 'address_road'
        }
    
    df.rename(columns=COLUMN_MAP, inplace=True)
    
    df['region_id'] = df['address_road'].apply(get_region_from_address)
    df.dropna(subset=['region_id', 'festival_name'], inplace=True)
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
    df.dropna(subset=['start_date', 'end_date'], inplace=True)
    period_start = pd.to_datetime('2023-04-30'); period_end = pd.to_datetime('2025-04-30')
    df_filtered = df[(df['start_date'] <= period_end) & (df['end_date'] >= period_start)].copy()
    df_filtered['actual_start'] = df_filtered.apply(lambda row: max(row['start_date'], period_start), axis=1)
    df_filtered['actual_end'] = df_filtered.apply(lambda row: min(row['end_date'], period_end), axis=1)
    df_filtered['days_in_period'] = (df_filtered['actual_end'] - df_filtered['actual_start']).dt.days + 1
    
    # 지역별 축제 밀도 및 축제 목록 계산
    agg_functions = {'days_in_period': 'sum', 'festival_name': lambda names: ', '.join(names.unique())}
    density_df = df_filtered.groupby('region_id').agg(agg_functions).reset_index()
    density_df.rename(columns={'days_in_period': 'total_festival_days', 'festival_name': 'festivals_in_period'}, inplace=True)
    total_period_days = (period_end - period_start).days + 1
    density_df['festival_density'] = density_df['total_festival_days'] / total_period_days
    
    # 상위 10개 지역
    top_10_density = density_df.sort_values(by='festival_density', ascending=False).head(10).reset_index(drop=True)
    top_10_density['festivals_in_period'] = top_10_density['festivals_in_period'].str[:50] + '...'
    
    # print
    print(top_10_density[['region_id', 'festival_density', 'total_festival_days', 'festivals_in_period']].to_string(index=False))
    


    return density_df


# --- 메인 실행 블록 ---
if __name__ == '__main__':
    DATA_FOLDER = 'data'
    OUTPUT_FOLDER = 'advanced'
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 관광정보에 대한 모든 CSV 파일 경로 수집
    all_files_in_data = os.listdir(DATA_FOLDER)
    resource_files = [f for f in all_files_in_data if '관광정보' in f and f.endswith('.csv')]
    resource_filepaths = [os.path.join(DATA_FOLDER, f) for f in resource_files]

    FILE_PATHS = {
        "spend": os.path.join(DATA_FOLDER, "지역별 지출액_내국인_전처리.csv"),
        "resources": resource_filepaths,
        "festivals": os.path.join(DATA_FOLDER, "전국문화축제표준데이터.csv")
    }

    # --- 데이터 전처리 및 분석 실행 ---
    spend_growth_processed, top_20_growth_df = calculate_spend_growth_rate(FILE_PATHS["spend"])
    resources_analyzed = analyze_tourism_resources(FILE_PATHS["resources"])
    festival_density_all = calculate_festival_density(FILE_PATHS["festivals"])
    
    # --- 결과 저장 ---
    if spend_growth_processed is not None:
        # 전체 성장률 저장
        output_path_all = os.path.join(OUTPUT_FOLDER, 'spend_growth_rate_all.csv')
        spend_growth_processed.to_csv(output_path_all, index=False, encoding='utf-8-sig')

    if resources_analyzed is not None:
        # 관광자원 분석 결과 저장
        output_path = os.path.join(OUTPUT_FOLDER, 'tourism_resources_analyzed.csv')
        resources_analyzed.to_csv(output_path, index=False, encoding='utf-8-sig')

    if festival_density_all is not None:
        # 축제 밀도 분석 결과 저장
        output_path = os.path.join(OUTPUT_FOLDER, 'festival_density_all_regions.csv')
        festival_density_all.to_csv(output_path, index=False, encoding='utf-8-sig')