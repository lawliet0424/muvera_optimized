#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDE Partition Statistics Calculator
Analyzes partition counts for each repetition in FDE data

Usage:
  python3 fde_partition_stats.py <csv_file> --metrics <metric1> <metric2> ...
  python3 fde_partition_stats.py simhash_count_3_4.csv --metrics mean median p95 p99
  python3 fde_partition_stats.py simhash_count_3_4.csv --metrics all --output results.csv
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np


# ========================
# Metric Functions
# ========================
def calculate_mean(data):
    """평균"""
    return data.mean()

def calculate_median(data):
    """중간값"""
    return data.median()

def calculate_std(data):
    """표준편차"""
    return data.std()

def calculate_min(data):
    """최소값"""
    return data.min()

def calculate_max(data):
    """최대값"""
    return data.max()

def calculate_p50(data):
    """50th percentile"""
    return data.quantile(0.50)

def calculate_p95(data):
    """95th percentile"""
    return data.quantile(0.95)

def calculate_p99(data):
    """99th percentile"""
    return data.quantile(0.99)

def calculate_p999(data):
    """99.9th percentile"""
    return data.quantile(0.999)

def calculate_count(data):
    """데이터 개수"""
    return len(data)

def calculate_sum(data):
    """합계"""
    return data.sum()


# Metric mapping
AVAILABLE_METRICS = {
    'mean': calculate_mean,
    'avg': calculate_mean,
    'median': calculate_median,
    'med': calculate_median,
    'std': calculate_std,
    'min': calculate_min,
    'max': calculate_max,
    'p50': calculate_p50,
    'p95': calculate_p95,
    'p99': calculate_p99,
    'p999': calculate_p999,
    'count': calculate_count,
    'sum': calculate_sum,
}


# ========================
# Main Functions
# ========================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='FDE CSV 파일의 repetition별, partition별 통계를 계산합니다.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 가능한 메트릭:
  mean, avg     : 평균
  median, med   : 중간값
  std           : 표준편차
  min           : 최소값
  max           : 최대값
  p50           : 50th percentile
  p95           : 95th percentile
  p99           : 99th percentile
  p999          : 99.9th percentile
  count         : 데이터 개수
  sum           : 합계
  all           : 모든 메트릭

예시:
  python3 fde_partition_stats.py simhash_count_3_4.csv --metrics mean median p95 p99
  python3 fde_partition_stats.py simhash_count_3_4.csv --metrics all
  python3 fde_partition_stats.py simhash_count_3_4.csv --metrics mean p95 --output results.csv
        """
    )
    
    parser.add_argument('csv_file', help='분석할 CSV 파일 경로')
    parser.add_argument('--metrics', '-m', nargs='+', required=True,
                        help='계산할 메트릭 (공백으로 구분, "all"로 모든 메트릭 선택 가능)')
    parser.add_argument('--output', '-o', help='결과를 저장할 CSV 파일 경로 (선택)')
    parser.add_argument('--group-by', '-g', choices=['repetition', 'partition', 'both'], 
                        default='both', help='그룹화 방식: repetition, partition, 또는 both (기본: both)')
    parser.add_argument('--skip-na', action='store_true', 
                        help='NaN 값 제거 (기본: True)', default=True)
    
    return parser.parse_args()


def load_csv(file_path):
    """CSV 파일 로드"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ CSV 파일 로드 완료: {file_path}")
        print(f"   - 전체 행 수: {len(df)}")
        print(f"   - 전체 칼럼: {list(df.columns)}")
        print(f"   - 데이터 미리보기:")
        print(df.head())
        return df
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        sys.exit(1)


def validate_dataframe(df):
    """데이터프레임 구조 검증"""
    required_cols = ['doc_idx', 'rep_num', 'partition_idx', 'count']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ 다음 칼럼을 찾을 수 없습니다: {missing}")
        print(f"   사용 가능한 칼럼: {list(df.columns)}")
        sys.exit(1)
    print(f"✅ 데이터 구조 확인 완료")


def get_metrics_to_calculate(metrics_input):
    """계산할 메트릭 목록 결정"""
    if 'all' in metrics_input:
        return list(AVAILABLE_METRICS.keys())
    
    invalid = [m for m in metrics_input if m not in AVAILABLE_METRICS]
    if invalid:
        print(f"❌ 알 수 없는 메트릭: {invalid}")
        print(f"   사용 가능한 메트릭: {list(AVAILABLE_METRICS.keys())}")
        sys.exit(1)
    
    return metrics_input


def calculate_statistics_by_repetition(df, metrics, skip_na=True):
    """Repetition별 통계 계산"""
    print(f"\n📊 Repetition별 분석 중...")
    results = {}
    
    for rep_num in sorted(df['rep_num'].unique()):
        rep_data = df[df['rep_num'] == rep_num]['count']
        
        if skip_na:
            rep_data = rep_data.dropna()
        
        if len(rep_data) == 0:
            print(f"   ⚠️  rep_num {rep_num}: 데이터가 없어 건너뜁니다.")
            continue
        
        rep_results = {}
        for metric in metrics:
            try:
                value = AVAILABLE_METRICS[metric](rep_data)
                rep_results[metric] = value
            except Exception as e:
                print(f"   ⚠️  rep_num {rep_num}, {metric} 계산 실패: {e}")
                rep_results[metric] = None
        
        results[f'rep_{rep_num}'] = rep_results
        print(f"   - rep_num {rep_num}: {len(rep_data)} 개 데이터 처리 완료")
    
    return results


def calculate_statistics_by_partition(df, metrics, skip_na=True):
    """Partition별 통계 계산"""
    print(f"\n📊 Partition별 분석 중...")
    results = {}
    
    for partition_idx in sorted(df['partition_idx'].unique()):
        partition_data = df[df['partition_idx'] == partition_idx]['count']
        
        if skip_na:
            partition_data = partition_data.dropna()
        
        if len(partition_data) == 0:
            print(f"   ⚠️  partition_idx {partition_idx}: 데이터가 없어 건너뜁니다.")
            continue
        
        partition_results = {}
        for metric in metrics:
            try:
                value = AVAILABLE_METRICS[metric](partition_data)
                partition_results[metric] = value
            except Exception as e:
                print(f"   ⚠️  partition_idx {partition_idx}, {metric} 계산 실패: {e}")
                partition_results[metric] = None
        
        results[f'partition_{partition_idx}'] = partition_results
        print(f"   - partition_idx {partition_idx}: {len(partition_data)} 개 데이터 처리 완료")
    
    return results


def calculate_statistics_by_both(df, metrics, skip_na=True):
    """Repetition과 Partition 조합별 통계 계산"""
    print(f"\n📊 Repetition × Partition 조합별 분석 중...")
    results = {}
    
    for rep_num in sorted(df['rep_num'].unique()):
        for partition_idx in sorted(df['partition_idx'].unique()):
            mask = (df['rep_num'] == rep_num) & (df['partition_idx'] == partition_idx)
            combo_data = df[mask]['count']
            
            if skip_na:
                combo_data = combo_data.dropna()
            
            if len(combo_data) == 0:
                continue
            
            combo_results = {}
            for metric in metrics:
                try:
                    value = AVAILABLE_METRICS[metric](combo_data)
                    combo_results[metric] = value
                except Exception as e:
                    combo_results[metric] = None
            
            results[f'rep_{rep_num}_part_{partition_idx}'] = combo_results
    
    print(f"   - 총 {len(results)} 개 조합 처리 완료")
    return results


def create_results_dataframe(results):
    """결과를 DataFrame으로 변환"""
    df_results = pd.DataFrame(results).T
    df_results.index.name = 'Group'
    return df_results


def save_results(df_results, output_path):
    """결과를 CSV로 저장"""
    try:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        df_results.to_csv(output_path)
        print(f"\n✅ 결과 저장 완료: {output_path}")
    except Exception as e:
        print(f"\n❌ 결과 저장 실패: {e}")


def print_results_table(df_results):
    """결과를 테이블 형식으로 출력"""
    print("\n" + "="*80)
    print("📈 통계 결과")
    print("="*80)
    print(df_results.to_string())
    print("="*80)


def print_summary(df):
    """데이터 요약 정보 출력"""
    print("\n" + "="*80)
    print("📋 데이터 요약")
    print("="*80)
    print(f"총 행 수: {len(df)}")
    print(f"Repetition 개수: {df['rep_num'].nunique()}")
    print(f"Partition 개수: {df['partition_idx'].nunique()}")
    print(f"Repetition 값: {sorted(df['rep_num'].unique())}")
    print(f"Partition 값: {sorted(df['partition_idx'].unique())}")
    print(f"Count 통계:")
    print(f"  - 최소: {df['count'].min()}")
    print(f"  - 최대: {df['count'].max()}")
    print(f"  - 평균: {df['count'].mean():.2f}")
    print(f"  - 중간값: {df['count'].median():.2f}")
    print("="*80)


def main():
    args = parse_arguments()
    
    # 1. CSV 로드
    df = load_csv(args.csv_file)
    
    # 2. 데이터 검증
    validate_dataframe(df)
    
    # 3. 데이터 요약
    print_summary(df)
    
    # 4. 메트릭 결정
    metrics = get_metrics_to_calculate(args.metrics)
    print(f"\n✅ 계산할 메트릭: {metrics}")
    
    # 5. 통계 계산
    if args.group_by == 'repetition':
        results = calculate_statistics_by_repetition(df, metrics, args.skip_na)
    elif args.group_by == 'partition':
        results = calculate_statistics_by_partition(df, metrics, args.skip_na)
    else:  # both
        results = calculate_statistics_by_both(df, metrics, args.skip_na)
    
    # 6. 결과 정리
    df_results = create_results_dataframe(results)
    
    # 7. 결과 출력
    print_results_table(df_results)
    
    # 8. 파일 저장
    if args.output:
        save_results(df_results, args.output)
    else:
        # 기본 저장 경로
        base_name = os.path.splitext(args.csv_file)[0]
        default_output = f"{base_name}_statistics_{args.group_by}.csv"
        save_results(df_results, default_output)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python3 fde_partition_stats.py <csv_file> --metrics <metric1> <metric2> ...")
        print("도움말: python3 fde_partition_stats.py --help")
        sys.exit(1)
    
    main()