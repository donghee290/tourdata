# uniqueness_analyzer.py

import pandas as pd
import numpy as np
import os
from itertools import combinations
import json

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import LocalOutlierFactor
from scipy.cluster.hierarchy import linkage, fcluster
from scipy import stats
import networkx as nx

import warnings
warnings.filterwarnings('ignore')

class UniquenessBasedRegionalAnalysis:
    """
    지역 고유성 추출에 특화된 군집화 분석 클래스
    """
    
    def __init__(self, lda_file, resource_file, category_file):
        self.lda_file = lda_file
        self.resource_file = resource_file
        self.category_file = category_file
        self.feature_matrix = None
        self.region_names = None
        self.uniqueness_scores = {}
        self.hidden_assets = {}
        self.distinctive_features = {}
        self.category_mapping = {} 
        self.category_df = None
    
    def load_data(self):
        self.lda_df = pd.read_csv(self.lda_file)
        self.resource_df = pd.read_csv(self.resource_file)
        self.category_df = pd.read_csv(self.category_file)
        self.standardize_region_names()
        self.load_category_mapping()
        
        print("=== 데이터 로드 완료 ===")
        print(f"LDA 토픽 데이터: {self.lda_df.shape}")
        print(f"관광자원 데이터: {self.resource_df.shape}")
        return self
    
    def standardize_region_names(self):
        self.lda_df['region'] = self.lda_df['region'].replace({
            '부산광역시 서구': '부산 서구',
            '부산광역시 영도구': '부산 영도구'
        })
        self.resource_df['region_id'] = self.resource_df['region_id'].replace({
            '부산광역시 서구': '부산 서구',
            '부산광역시 영도구': '부산 영도구'
        })
        print("지역명 표준화 완료")
    
    def load_category_mapping(self):
        self.category_mapping = {}
        for _, row in self.category_df.drop_duplicates('lclsSystm2Cd').iterrows():
            self.category_mapping[row['lclsSystm2Cd']] = row['lclsSystm2Nm']
            
        print(f"카테고리 분류체계 로드 완료")
        return self
    
    def extract_enhanced_features(self):
        regions = self.lda_df['region'].unique()
        self.lda_features_dict = {}
        self.resource_features_dict = {}
        
        for region in regions:
            region_data = self.lda_df[self.lda_df['region'] == region]
            features = self._extract_lda_features(region_data)
            uniqueness_features = self._calculate_topic_uniqueness(region_data, self.lda_df)
            features.extend(uniqueness_features)
            self.lda_features_dict[region] = features
        
        for region in self.resource_df['region_id'].unique():
            region_data = self.resource_df[self.resource_df['region_id'] == region]
            features = self._extract_resource_features(region_data)
            rarity_features = self._calculate_resource_rarity(region_data, self.resource_df)
            features.extend(rarity_features)
            self.resource_features_dict[region] = features
        
        return self
    
    def _extract_resource_features(self, region_data):
        all_categories = list(self.category_mapping.keys())
        proportions = []
        
        for cat_code in all_categories:
            cat_data = region_data[region_data['lclsSystm2'] == cat_code]
            prop = cat_data['proportion'].values[0] if len(cat_data) > 0 else 0
            proportions.append(prop)
        
        props = np.array(proportions) / 100
        props = props[props > 0]
        diversity = -np.sum(props * np.log(props + 1e-10)) if len(props) > 0 else 0
        concentration = np.sum((np.array(proportions) / 100) ** 2)
        non_zero = [p for p in proportions if p > 0]
        specialization = max(proportions) / (np.mean(non_zero) + 1e-10) if non_zero else 0
        top3_indices = np.argsort(proportions)[-3:]
        top3_sum = sum([proportions[i] for i in top3_indices])
        theme_combinations = []
        category_indices = {}
        
        nature_culture = 0
        if 'NA' in category_indices:
            nature_culture += proportions[category_indices['NA']]
        if 'VE' in category_indices:
            nature_culture += proportions[category_indices['VE']]
        if 'HS' in category_indices:
            nature_culture += proportions[category_indices['HS']]
        theme_combinations.append(nature_culture / 100)
        
        activity_leisure = 0
        if 'EX' in category_indices:
            activity_leisure += proportions[category_indices['EX']]
        if 'LS' in category_indices:
            activity_leisure += proportions[category_indices['LS']]
        theme_combinations.append(activity_leisure / 100)
        
        festival_food = 0
        if 'EV' in category_indices:
            festival_food += proportions[category_indices['EV']]
        if 'FD' in category_indices:
            festival_food += proportions[category_indices['FD']]
        theme_combinations.append(festival_food / 100)
        
        urban_tourism = 0
        if 'SH' in category_indices:
            urban_tourism += proportions[category_indices['SH']]
        if 'AC' in category_indices:
            urban_tourism += proportions[category_indices['AC']]
        theme_combinations.append(urban_tourism / 100)
        
        max_single = max(proportions) / 100 if proportions else 0
        balance = 1 / (np.std(proportions) + 1) if proportions else 0
        
        return (proportions + 
                [diversity, concentration, specialization, top3_sum/100] +
                theme_combinations + 
                [max_single, balance])
    
    def _extract_lda_features(self, region_data):
        features = []
        
        for topic_id in sorted(region_data['topic'].unique()):
            topic_words = region_data[region_data['topic'] == topic_id]
            top5_beta = topic_words.nsmallest(5, 'rank')['beta'].mean()
            top3_concentration = topic_words.nsmallest(3, 'rank')['beta'].sum()
            beta_diversity = topic_words['beta'].std()
            max_beta = topic_words['beta'].max()
            unique_words = self._find_unique_words(topic_words['term'].tolist()[:5])
            topic_uniqueness = len(unique_words) / 5
            features.extend([top5_beta, top3_concentration, beta_diversity, max_beta, topic_uniqueness])
        
        while len(features) < 30:
            features.append(0)
        
        return features[:30]
    
    def _find_unique_words(self, words):
        all_words = self.lda_df.groupby('term')['region'].nunique()
        unique_words = [w for w in words if all_words.get(w, 0) <= 1]
        return unique_words
    
    def _calculate_topic_uniqueness(self, region_data, all_data):
        region_terms = set(region_data['term'].tolist())
        other_regions_terms = set(all_data[all_data['region'] != region_data['region'].iloc[0]]['term'].tolist())
        exclusive_terms = region_terms - other_regions_terms
        exclusivity_ratio = len(exclusive_terms) / len(region_terms) if region_terms else 0
        term_counts = all_data.groupby('term')['region'].nunique()
        region_term_rarity = [1/term_counts.get(term, 1) for term in region_terms]
        avg_rarity = np.mean(region_term_rarity) if region_term_rarity else 0
        return [exclusivity_ratio, avg_rarity]
    
    def _calculate_resource_rarity(self, region_data, all_data):
        top_category = region_data.nlargest(1, 'proportion')['lclsSystm2'].iloc[0]
        regions_with_category = all_data[all_data['lclsSystm2'] == top_category]['region_id'].nunique()
        total_regions = all_data['region_id'].nunique()
        rarity_score = 1 - (regions_with_category / total_regions)
        resource_pattern = tuple(sorted(region_data['lclsSystm2'].tolist()))
        pattern_frequency = sum(1 for r in all_data['region_id'].unique() 
                               if tuple(sorted(all_data[all_data['region_id'] == r]['lclsSystm2'].tolist())) == resource_pattern)
        combination_rarity = 1 / pattern_frequency if pattern_frequency > 0 else 1
        return [rarity_score, combination_rarity]
    
    def calculate_uniqueness_scores(self):
        self.extract_enhanced_features()
        common_regions = list(set(self.lda_features_dict.keys()) & set(self.resource_features_dict.keys()))
        
        if not common_regions:
            print("경고: LDA 데이터와 관광자원 데이터에 공통된 지역이 없습니다.")
            return self
        
        feature_list = []
        for region in common_regions:
            features = self.lda_features_dict[region] + self.resource_features_dict[region]
            feature_list.append(features)
        
        self.feature_matrix = np.array(feature_list)
        self.region_names = common_regions
        
        scaler = RobustScaler()
        self.feature_matrix_scaled = scaler.fit_transform(self.feature_matrix)
        
        for i, region in enumerate(self.region_names):
            lof_score = self._calculate_lof_score(i)
            rarity_score = self._calculate_statistical_rarity(i)
            network_score = self._calculate_network_uniqueness(i)
            
            self.uniqueness_scores[region] = {
                'lof': lof_score,
                'rarity': rarity_score,
                'network': network_score,
                'composite': 0.4 * lof_score + 0.3 * rarity_score + 0.3 * network_score
            }
        return self
    
    def _calculate_lof_score(self, region_idx):
        lof = LocalOutlierFactor(n_neighbors=min(3, len(self.region_names)-1), novelty=False, contamination='auto')
        lof.fit(self.feature_matrix_scaled)
        negative_outlier_factors = lof.negative_outlier_factor_
        sorted_factors = np.sort(negative_outlier_factors)
        rank = np.where(sorted_factors == negative_outlier_factors[region_idx])[0][0]
        return 1 - (rank / (len(self.region_names) - 1))
    
    def _calculate_statistical_rarity(self, region_idx):
        median = np.median(self.feature_matrix_scaled, axis=0)
        mad = np.median(np.abs(self.feature_matrix_scaled - median), axis=0)
        mad[mad == 0] = 1e-10
        robust_z_scores = np.abs(0.6745 * (self.feature_matrix_scaled - median) / mad)
        rare_features = np.sum(robust_z_scores[region_idx] > 3.5)
        return rare_features / self.feature_matrix_scaled.shape[1]
    
    def _calculate_network_uniqueness(self, region_idx):
        similarities = cosine_similarity(self.feature_matrix_scaled)
        threshold = np.percentile(similarities, 30)
        G = nx.Graph()
        for i in range(len(self.region_names)):
            G.add_node(i)
        for i in range(len(self.region_names)):
            for j in range(i+1, len(self.region_names)):
                if similarities[i, j] < threshold:
                    G.add_edge(i, j, weight=1-similarities[i, j])
        if G.number_of_edges() > 0:
            betweenness = nx.betweenness_centrality(G).get(region_idx, 0)
            degree = G.degree(region_idx) / (len(self.region_names) - 1)
            return (betweenness + degree) / 2
        return 0
    
    def discover_hidden_assets(self):
        # lclsSystm2Cd에 lclsSystm3Nm과 lclsSystm2Nm을 모두 매핑
        category_keyword_map = {}
        category_df_filtered = self.category_df.dropna(subset=['lclsSystm2Cd', 'lclsSystm2Nm'])
        for lcls2_code, group in category_df_filtered.groupby('lclsSystm2Cd'):
            keywords_from_lcls3 = group['lclsSystm3Nm'].tolist()
            keywords_from_lcls2 = group['lclsSystm2Nm'].drop_duplicates().tolist()
            
            combined_keywords = list(set(keywords_from_lcls3 + keywords_from_lcls2))
            category_keyword_map[lcls2_code] = combined_keywords
        
        lcls2_name_map = self.category_df.drop_duplicates('lclsSystm2Cd').set_index('lclsSystm2Cd')['lclsSystm2Nm'].to_dict()

        # lda 리스트에서 지역명을 제외하여 리스트 생성
        regions_to_exclude = set()
        for region_name in self.lda_df['region'].unique():
            province_name = region_name.split()[0]
            regions_to_exclude.add(province_name.lower())

        for region in self.region_names:
            lda_keywords_full = self.lda_df[self.lda_df['region'] == region]['term'].tolist()
            filtered_lda_keywords = []
            for kw in lda_keywords_full:
                kw_lower = kw.lower()
                # 키워드가 제외할 지역명에 포함되지 않는지 확인
                if not any(excl_word in kw_lower for excl_word in regions_to_exclude):
                    filtered_lda_keywords.append(kw)

            lda_keywords_lower = [kw.lower() for kw in filtered_lda_keywords]
            
            if region not in self.resource_df['region_id'].values:
                self.hidden_assets[region] = []
                continue
            
            region_resources = self.resource_df[self.resource_df['region_id'] == region]
            hidden = []
            
            for _, resource in region_resources.iterrows():
                category_code = resource['lclsSystm2'] 
                proportion = resource['proportion']
                category_name = lcls2_name_map.get(category_code, category_code)

                mentioned = False
                if category_code in category_keyword_map:
                    for keyword in category_keyword_map[category_code]:
                        if any(keyword in kw or kw in keyword for kw in lda_keywords_lower):
                            mentioned = True
                            break
                
                top5_categories = region_resources.nlargest(5, 'proportion')['lclsSystm2'].tolist()
                
                if not mentioned:
                    if proportion >= 10:
                        hidden.append({
                            'category': category_name,
                            'category_code': category_code,
                            'proportion': proportion,
                            'potential': 'HIGH' if proportion >= 20 else 'MEDIUM',
                            'reason': '높은 비중에도 불구하고 미언급'
                        })
                    elif proportion >= 5 and category_code in top5_categories:
                        hidden.append({
                            'category': category_name,
                            'category_code': category_code,
                            'proportion': proportion,
                            'potential': 'LOW',
                            'reason': '상위 카테고리이지만 미언급'
                        })
            
            unique_hidden = {}
            for asset in hidden:
                key = asset['category_code']
                if key not in unique_hidden or unique_hidden[key]['proportion'] < asset['proportion']:
                    unique_hidden[key] = asset
            
            self.hidden_assets[region] = list(unique_hidden.values())
        
        total_hidden = sum(len(assets) for assets in self.hidden_assets.values())
        print(f"\n숨은 자산 발굴 완료: 총 {total_hidden}개 발견")
        return self
    
    def identify_distinctive_combinations(self):
        for i, region in enumerate(self.region_names):
            if i >= len(self.feature_matrix):
                continue
            region_features = self.feature_matrix[i]
            feature_indices = np.argsort(region_features)[-5:]
            combinations_list = []
            for combo_size in [2, 3]:
                for combo in combinations(feature_indices, combo_size):
                    count = 0
                    for j, _ in enumerate(self.region_names):
                        if i != j:
                            other_features = self.feature_matrix[j]
                            if all(other_features[idx] > np.percentile(self.feature_matrix[:, idx], 70) 
                                   for idx in combo):
                                count += 1
                    if count <= 2:
                        combinations_list.append({
                            'features': combo,
                            'rarity': 1 / (count + 1),
                            'strength': np.mean([region_features[idx] for idx in combo])
                        })
            if combinations_list:
                best_combo = max(combinations_list, key=lambda x: x['rarity'] * x['strength'])
                self.distinctive_features[region] = best_combo
            else:
                self.distinctive_features[region] = None
        return self
    
    def perform_uniqueness_clustering(self):
        uniqueness_array = np.array([self.uniqueness_scores[r]['composite'] for r in self.region_names])
        enhanced_features = np.column_stack([
            self.feature_matrix_scaled,
            uniqueness_array.reshape(-1, 1) * 2
        ])
        linkage_matrix = linkage(enhanced_features, method='ward')
        n_clusters = min(5, max(3, len(self.region_names) // 3)) if len(self.region_names) > 5 else min(3, max(2, len(self.region_names) // 2))
        self.cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
        
        cluster_uniqueness = {}
        for cluster_id in range(n_clusters):
            mask = self.cluster_labels == cluster_id
            cluster_uniqueness[cluster_id] = np.mean(uniqueness_array[mask])
        
        sorted_clusters = sorted(cluster_uniqueness.items(), key=lambda x: x[1], reverse=True)
        new_labels = np.zeros_like(self.cluster_labels)
        for new_id, (old_id, _) in enumerate(sorted_clusters):
            new_labels[self.cluster_labels == old_id] = new_id
        self.cluster_labels = new_labels
        
        print(f"\n=== 계층적 클러스터링 결과 ===")
        print(f"├─ 총 클러스터 수: {n_clusters}")
        
        for cluster_id in range(n_clusters):
            mask = self.cluster_labels == cluster_id
            cluster_regions = [self.region_names[i] for i, m in enumerate(mask) if m]
            avg_uniqueness = np.mean(uniqueness_array[mask])
            cluster_type = "고유성 높음" if avg_uniqueness > 0.6 else "중간 고유성" if avg_uniqueness > 0.3 else "일반적"
            print(f"├─ 클러스터 {cluster_id} ({cluster_type}, 고유성: {avg_uniqueness:.3f})")
            print(f"│   └─ 지역: {', '.join(cluster_regions[:3])}")
            if len(cluster_regions) > 3:
                print(f"│        외 {len(cluster_regions)-3}개")
        
        self.linkage_matrix = linkage_matrix
        self.cluster_uniqueness_scores = cluster_uniqueness
        
        # PCA를 수행하여 결과를 PC1, PC2 컬럼으로 추가
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(self.feature_matrix_scaled)
        
        # 숨은 자산 데이터를 평탄화
        hidden_assets_data = []
        for region in self.region_names:
            if self.hidden_assets.get(region):
                for asset in self.hidden_assets[region]:
                    hidden_assets_data.append({
                        'region': region,
                        'hidden_category': asset['category'],
                        'hidden_category_code': asset['category_code'],
                        'hidden_proportion': asset['proportion'],
                        'hidden_potential': asset['potential'],
                        'hidden_reason': asset['reason']
                    })

        results_data = []
        for region in self.region_names:
            idx = self.region_names.index(region)
            
            # distinctive_features 데이터를 JSON 직렬화 전에 변환
            distinctive_combo = self.distinctive_features.get(region, None)
            if distinctive_combo:
                # NumPy int64를 파이썬 기본 int로 변환
                distinctive_combo['features'] = [int(i) for i in distinctive_combo['features']]
                distinctive_combo_json = json.dumps(distinctive_combo, ensure_ascii=False)
            else:
                distinctive_combo_json = json.dumps(None)
                
            # 자원 다양성 지수 추출 (features 리스트의 마지막 6번째 값)
            diversity_index = self.resource_features_dict[region][-6] if region in self.resource_features_dict else 0
            
            results_data.append({
                'region': region,
                'cluster': int(self.cluster_labels[idx]),
                'uniqueness_composite': self.uniqueness_scores[region]['composite'],
                'uniqueness_lof': self.uniqueness_scores[region]['lof'],
                'uniqueness_rarity': self.uniqueness_scores[region]['rarity'],
                'uniqueness_network': self.uniqueness_scores[region]['network'],
                'resource_diversity_index': diversity_index,
                'distinctive_combo_json': distinctive_combo_json,
                'PC1': features_2d[idx, 0],
                'PC2': features_2d[idx, 1],
                'hidden_assets_count': int(len(self.hidden_assets.get(region, [])))
            })
        
        results_df = pd.DataFrame(results_data)
        hidden_df = pd.DataFrame(hidden_assets_data)
        
        return results_df, hidden_df

    def export_uniqueness_results(self, results_df, hidden_df, output_prefix):
        results_df.to_csv(f'{output_prefix}_analysis.csv', index=False, encoding='utf-8-sig')
        if not hidden_df.empty:
            hidden_df.to_csv(f'{output_prefix}_hidden_assets.csv', index=False, encoding='utf-8-sig')
        
        print(f"\n결과 파일 저장 완료:")
        print(f"- {output_prefix}_analysis.csv")

if __name__ == "__main__":
    DATA_FOLDER = 'data'
    ADV_FOLDER = 'advanced'
    OUTPUT_FOLDER = 'uniqueness_output'
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("="*80)
    print(" "*20 + "지역 고유성 기반 관광 브랜딩 분석 시작")
    print("="*80)
    
    analyzer = UniquenessBasedRegionalAnalysis(
        lda_file=os.path.join(ADV_FOLDER, 'region_lda_topics_summary.csv'),
        resource_file=os.path.join(ADV_FOLDER, 'tourism_resources_analyzed.csv'),
        category_file=os.path.join(DATA_FOLDER, '콘텐츠_분류체계_코드.csv')
    )
    analyzer.load_data()
    analyzer.calculate_uniqueness_scores()
    analyzer.discover_hidden_assets()
    analyzer.identify_distinctive_combinations()
    results_df, hidden_df = analyzer.perform_uniqueness_clustering()
    analyzer.export_uniqueness_results(results_df, hidden_df, output_prefix=os.path.join(OUTPUT_FOLDER, 'regional_uniqueness'))
    print("\n분석 로직 실행 완료. 결과를 확인하고 시각화/리포팅 파일을 실행하세요.")