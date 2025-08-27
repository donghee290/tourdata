import json
import requests
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
import time

# 환경변수 로드
load_dotenv()

class TourismDataCollector:
    def __init__(self):
        self.config = self.load_config()
        self.service_key = self.load_service_key()
        self.output_dir = Path("data")
        self.output_dir.mkdir(exist_ok=True)
        
        # 수집할 목표 시군구 목록 사전 정의
        self.target_sigungu_list = self.config.get('target_regions', [])
        if not self.target_sigungu_list:
            print("경고: config.json에 'target_regions'가 비어있거나 없습니다.")
        
        # 코드 테이블 캐시
        self.area_codes = {}
        self.target_codes = [] # (area_name, area_code, sigungu_name, sigungu_code) 저장
        self.classification_codes = {}

    def load_config(self):
        """config.json 로드"""
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_service_key(self):
        """serviceKey 로드"""
        service_key = os.getenv('apiKeyDecoding')
        if not service_key:
            raise ValueError("환경변수 'SERVICE_KEY'가 설정되어 있지 않습니다.")
        return service_key

    def call_api(self, api_name, custom_params=None):
        """API 호출"""
        api_config = self.config['apis'][api_name]
        params = api_config['params'].copy()
        params['serviceKey'] = self.service_key
        
        if custom_params:
            params.update(custom_params)
            
        params = {k: v for k, v in params.items() if v != ""}
        
        try:
            response = requests.get(api_config['url'], params=params)
            response.raise_for_status()
            data = response.json()
            if 'response' in data and 'header' in data['response'] and data['response']['header'].get('resultCode') != '0000':
                raise Exception(f"API 오류: {data['response']['header'].get('resultMsg')}")
            return data
        except Exception as e:
            print(f"API 호출 실패 ({api_name}): {e}")
            return None

    def extract_items(self, response_data):
        """응답에서 데이터 추출"""
        if not response_data: return []
        try:
            items = response_data['response']['body']['items']['item']
            return items if isinstance(items, list) else [items]
        except (KeyError, TypeError):
            return []

    def get_all_pages(self, api_name, custom_params=None, max_pages=50):
        """전체 페이지 데이터 조회"""
        all_data = []
        page = 1
        while page <= max_pages:
            page_params = custom_params.copy() if custom_params else {}
            page_params['pageNo'] = page
            
            response = self.call_api(api_name, page_params)
            if not response: break
            
            items = self.extract_items(response)
            if not items: break
            
            all_data.extend(items)
            
            # 마지막 페이지 확인 (totalCount 기준)
            try:
                total_count = response['response']['body']['totalCount']
                num_rows = int(page_params.get('numOfRows', 10))
                if page * num_rows >= total_count:
                    break
            except (KeyError, TypeError, ValueError):
                # totalCount 정보가 없는 경우, 이전 로직(아이템 수) 사용
                num_rows = int(page_params.get('numOfRows', 10))
                if len(items) < num_rows:
                    break
            
            page += 1
            time.sleep(0.1) # API 서버 부하 감소를 위한 약간의 지연
        return all_data

    def load_area_codes(self):
        """광역 지역코드 조회"""
        if self.area_codes: 
            return self.area_codes
        data = self.get_all_pages("지역코드조회")
        self.area_codes = {item['name']: item['code'] for item in data if 'name' in item and 'code' in item}
        
        return self.area_codes

    def prepare_target_codes(self):
        """목표 시군구의 지역코드와 시군구코드 조회"""
        if self.target_codes: return self.target_codes
        
        self.load_area_codes() # 광역 지역코드 우선 로드
        
        # 목표 지역명을 광역/시군구로 분리
        parsed_targets = {}
        for target_full_name in self.target_sigungu_list:
            parts = target_full_name.split(' ')
            area_name = parts[0]
            sigungu_name = parts[1]
            if area_name not in parsed_targets:
                parsed_targets[area_name] = []
            parsed_targets[area_name].append(sigungu_name)

        # 광역별로 시군구 코드 조회
        for area_name, sigungu_list in parsed_targets.items():
            area_code = self.area_codes.get(area_name)
            if not area_code:
                print(f"⚠️ 경고: '{area_name}'에 대한 지역코드를 찾을 수 없습니다.")
                continue

            sigungu_data = self.get_all_pages("지역코드조회", {"areaCode": area_code})
            sigungu_map = {item['name']: item['code'] for item in sigungu_data}

            for sigungu_name in sigungu_list:
                sigungu_code = sigungu_map.get(sigungu_name)
                if sigungu_code:
                    self.target_codes.append((area_name, area_code, sigungu_name, sigungu_code))
                else:
                    print(f"    ⚠️ 경고: '{area_name} {sigungu_name}'의 시군구코드를 찾을 수 없습니다.")

        return self.target_codes

    def load_classification_codes(self):
        """콘텐츠 분류코드 조회"""
        data = self.get_all_pages("분류체계코드조회")
        if not data:
            return {}
        for item in data:
            if 'lclsSystem2Cd' in item and 'lclsSystem2Nm' in item:
                code = item['lclsSystem2Cd']
                name = item['lclsSystem2Nm']
                self.classification_codes[code] = name
                
        return self.classification_codes
    
    # ================== 데이터 수집 ==================
    
    def collect_tourism_stats(self, _type):
        """관광통계 수집"""

        try:
            data = self.get_all_pages(_type)
            
            if data:
                df = pd.DataFrame(data)
                start_date = self.config['apis'][_type]['params']['startYmd']
                end_date = self.config['apis'][_type]['params']['endYmd']
                filename = f"{_type}_{start_date}_{end_date}.csv"
                filepath = self.output_dir / filename
                df.to_csv(filepath, index=False, encoding='utf-8-sig')
                return df
            
        except Exception as e:
            print(f"관광통계 수집 실패: {e}")
        
        return pd.DataFrame()

    def collect_target_area_tourism(self):
        """목표 지역의 관광 정보만 수집"""
        
        # 1. 목표 지역 코드 준비
        self.prepare_target_codes()
        
        if not self.target_codes:
            print("❌ 수집할 목표 지역이 없습니다. 프로세스를 종료합니다.")
            return {}

        all_results = {}
        for content_type, content_type_id in self.config['content_types'].items():         
            content_type_data = []
            # 2. 목표 시군구를 하나씩 순회하며 데이터 수집
            for area_name, area_code, sigungu_name, sigungu_code in self.target_codes:
                # 3. API 호출을 지역별/콘텐츠타입별 1회로 최소화
                params = {
                    "areaCode": area_code,
                    "sigunguCode": sigungu_code,
                    "contentTypeId": content_type_id
                }
                
                items = self.get_all_pages("관광지정보", params)
                
                # 4. 수집된 데이터에 출처 정보(메타데이터) 추가
                for item in items:
                    if item:
                        item['광역지역명'] = area_name
                        item['시군구명'] = sigungu_name
                        item['콘텐츠타입'] = content_type
                
                content_type_data.extend(items)

            # 5. 콘텐츠 타입별로 결과 저장
            if content_type_data:
                df = pd.DataFrame(content_type_data)
                # 중복 제거
                df.drop_duplicates(subset=['contentid'], inplace=True, keep='first')
                filename = f"{content_type}_관광정보_{len(self.target_codes)}개_시군구.csv"
                filepath = self.output_dir / filename
                df.to_csv(filepath, index=False, encoding='utf-8-sig')
                all_results[content_type] = df
            else:
                print(f"  ⚠️ '{content_type}'에서 수집된 데이터가 없습니다.")
                all_results[content_type] = pd.DataFrame()
        
        return all_results
    
    def collect_classification_codes(self):
        """콘텐츠 분류코드 수집"""
        try:
            data = self.get_all_pages("분류체계코드조회")
            if data:
                df = pd.DataFrame(data)
                filename = "콘텐츠_분류체계_코드.csv"
                filepath = self.output_dir / filename
                df.to_csv(filepath, index=False, encoding='utf-8-sig')
                return df
        except Exception as e:
            print(f"콘텐츠 분류코드 수집 실패: {e}")
        
        return pd.DataFrame()

    def run(self):
        """전체 데이터 수집 실행"""
        print("=" * 60)
        print("관광 데이터 수집 시스템 시작")
        print("=" * 60)
        
        results = {}
        
        # 관광통계 수집
        types_to_collect = ["광역관광통계", "지자체관광통계"]
        for _type in types_to_collect:
            try:
                stats_df = self.collect_tourism_stats(_type)
                results['관광통계'] = stats_df
            except Exception as e:
                print(f"❌ 관광통계 수집 실패: {e}")
                results['관광통계'] = pd.DataFrame()
        
        # 전체 지역 관광정보 수집
        try:
            tourism_results = self.collect_target_area_tourism()
            results.update(tourism_results)
        except Exception as e:
            print(f"❌ 지역 관광정보 수집 중 심각한 오류 발생: {e}")
            
        try:
            class_df = self.collect_classification_codes()
            results['분류체계코드'] = class_df
        except Exception as e:
            print(f"❌ 분류체계코드 수집 실패: {e}")
            results['분류체계코드'] = pd.DataFrame()

        # 결과 요약
        print("\n" + "=" * 60)
        print("📊 수집 결과 요약")
        print("=" * 60)
        
        total_records = 0
        for data_type, df in results.items():
            count = len(df)
            if count > 0:
                print(f"✅ {data_type}: {count:,}건")
                total_records += count
            else:
                print(f"⚠️ {data_type}: 데이터 없음")
        
        print(f"\n📈 총 수집 데이터: {total_records:,}건")
        print(f"📂 저장 위치: {self.output_dir.absolute()}")
        print("=" * 60)
        
        return results

def main():
    try:
        collector = TourismDataCollector()
        collector.run()
    except Exception as e:
        raise

if __name__ == "__main__":
    main()