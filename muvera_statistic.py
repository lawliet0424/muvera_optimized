#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flexible Statistics Calculator for CSV files
사용법:
  python3 flexible_statistic.py <csv_file> --columns <col1> <col2> ... --metrics <metric1> <metric2> ...

예시:
  python3 flexible_statistic.py data.csv --columns Duration Latency --metrics mean median p95 p99
  python3 flexible_statistic.py data.csv --columns Score --metrics all
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import pathlib


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
        description='CSV 파일의 특정 칼럼에 대한 통계를 계산합니다.',
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
  python3 flexible_statistic.py data.csv --columns Duration Latency --metrics mean median p95 p99
  python3 flexible_statistic.py data.csv --columns Score --metrics all
  python3 flexible_statistic.py data.csv --columns Duration --metrics mean p95 p99 --output results.csv
  python3 flexible_statistic.py data.csv --columns Search Rerank --metrics mean p95 --dataset 
        """
    )
    
    parser.add_argument('csv_file', help='분석할 CSV 파일 경로')
    parser.add_argument('--columns', '-c', nargs='+', required=True, 
                        help='분석할 칼럼명 (공백으로 구분)')
    parser.add_argument('--metrics', '-m', nargs='+', required=True,
                        help='계산할 메트릭 (공백으로 구분, "all"로 모든 메트릭 선택 가능)')
    parser.add_argument('--output', '-o', help='결과를 저장할 CSV 파일 경로 (선택)')
    parser.add_argument('--dataset', '-d', help='데이터셋 이름 (결과에 포함)', default='unknown')
    parser.add_argument('--method', '-med', help='메서드 이름 (결과에 포함)', default='unknown')
    parser.add_argument('--skip-na', action='store_true', 
                        help='NaN 값 제거 (기본: True)', default=True)
    
    return parser.parse_args()


def load_csv(file_path):
    """CSV/TSV 파일 로드"""
    try:
        # 파일 확장자에 따라 구분자 결정
        if file_path.endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t')
            file_type = "TSV"
        else:
            df = pd.read_csv(file_path)
            file_type = "CSV"
        
        print(f"✅ {file_type} 파일 로드 완료: {file_path}")
        print(f"   - 전체 행 수: {len(df)}")
        print(f"   - 전체 칼럼: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        sys.exit(1)


def validate_columns(df, columns):
    """칼럼 존재 여부 확인"""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        print(f"❌ 다음 칼럼을 찾을 수 없습니다: {missing}")
        print(f"   사용 가능한 칼럼: {list(df.columns)}")
        sys.exit(1)
    print(f"✅ 칼럼 확인 완료: {columns}")


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


def calculate_statistics(df, columns, metrics, skip_na=True):
    """통계 계산"""
    results = {}
    
    for col in columns:
        print(f"\n📊 칼럼 '{col}' 분석 중...")
        data = df[col]
        
        if skip_na:
            original_count = len(data)
            data = data.dropna()
            dropped = original_count - len(data)
            if dropped > 0:
                print(f"   - NaN 값 {dropped}개 제거됨 (남은 데이터: {len(data)}개)")
        
        if len(data) == 0:
            print(f"   ⚠️  데이터가 없어 건너뜁니다.")
            continue
        
        col_results = {}
        for metric in metrics:
            try:
                value = AVAILABLE_METRICS[metric](data)
                col_results[metric] = value
                print(f"   - {metric}: {value:.4f}")
            except Exception as e:
                print(f"   ⚠️  {metric} 계산 실패: {e}")
                col_results[metric] = None
        
        results[col] = col_results
    
    return results


def create_results_dataframe(results, dataset_name='unknown'):
    """결과를 DataFrame으로 변환"""
    # 구조: rows=칼럼명, columns=메트릭
    df_results = pd.DataFrame(results).T
    df_results.index.name = 'Column'
    
    # 데이터셋 이름을 첫 번째 컬럼으로 추가
    df_results.insert(0, 'Dataset', dataset_name)
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


def main():
    args = parse_arguments()

    # 사전 준비 : 디렉토리 위치
    DIRECTORY = os.path.join(pathlib.Path(__file__).parent.absolute(), "cache_muvera", "query_search", args.dataset, args.method)
    
    # 1. CSV 로드
    csv_file = os.path.join(DIRECTORY, "latency.tsv")
    df = load_csv(csv_file)
    
    # 2. 칼럼 검증
    validate_columns(df, args.columns)
    
    # 3. 메트릭 결정
    metrics = get_metrics_to_calculate(args.metrics)
    print(f"✅ 계산할 메트릭: {metrics}")
    
    # 4. 통계 계산
    results = calculate_statistics(df, args.columns, metrics, args.skip_na)
    
    # 5. 결과 정리
    df_results = create_results_dataframe(results, args.dataset)
    
    # 6. 결과 출력
    print_results_table(df_results)
    
    # 7. 파일 저장 (옵션)
    if args.output:
        save_results(df_results, args.output)
    else:
        # 기본 저장 경로
        default_output = os.path.join(DIRECTORY, "statistics.csv")
        save_results(df_results, default_output)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python3 flexible_statistic.py <csv_file> --columns <col1> <col2> ... --metrics <metric1> <metric2> ...")
        print("도움말: python3 flexible_statistic.py --help")
        sys.exit(1)
    
    main()