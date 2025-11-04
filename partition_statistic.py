#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Partition Masking and Utilization Statistics Calculator
1) doc_idx별로 유효 partition 마스킹 생성 (count > 0인 partition)
2) 마스킹 정보를 바탕으로 partition별 활용도 통계 계산

Usage:
  python3 partition_masking_stats.py <csv_file> --rep-num 0
  python3 partition_masking_stats.py partition_counter.csv --rep-num 0 --output results.csv
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Partition 마스킹 및 활용도 통계 계산',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('csv_file', help='분석할 CSV 파일 경로')
    parser.add_argument('--rep-num', '-r', type=int, 
                        help='분석할 repetition 번호 (지정하지 않으면 전체 repetition 분석)')
    parser.add_argument('--output', '-o', help='결과를 저장할 CSV 파일 경로 (선택)')
    parser.add_argument('--output-mask', help='마스킹 결과를 저장할 CSV 파일 경로 (선택)')
    parser.add_argument('--output-rep-stats', help='Repetition별 통계를 저장할 CSV 파일 경로 (선택)')
    
    return parser.parse_args()


def load_csv(file_path):
    """CSV 파일 로드"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ CSV 파일 로드 완료: {file_path}")
        print(f"   - 전체 행 수: {len(df)}")
        print(f"   - 전체 칼럼: {list(df.columns)}")
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


def create_partition_masking(df, rep_num):
    """
    Step 1: doc_idx별로 partition 마스킹 생성
    count > 0인 partition은 1, 그렇지 않으면 0
    
    Returns:
        masking_df: DataFrame with columns [doc_idx, rep_num, partition_masking, active_partition_count]
    """
    print(f"\n📊 Step 1: doc_idx별 Partition 마스킹 생성 중 (rep_num={rep_num})...")
    
    # 해당 repetition만 필터링
    df_filtered = df[df['rep_num'] == rep_num].copy()
    
    if len(df_filtered) == 0:
        print(f"❌ rep_num={rep_num}에 해당하는 데이터가 없습니다.")
        return None, None, None
    
    # partition 개수 확인
    num_partitions = df_filtered['partition_idx'].nunique()
    partition_indices = sorted(df_filtered['partition_idx'].unique())
    print(f"   - Partition 개수: {num_partitions}")
    print(f"   - Partition 인덱스: {partition_indices}")
    
    # doc_idx 목록
    doc_indices = sorted(df_filtered['doc_idx'].unique())
    print(f"   - Document 개수: {len(doc_indices)}")
    
    masking_results = []
    
    for doc_idx in doc_indices:
        # 해당 doc_idx의 데이터
        doc_data = df_filtered[df_filtered['doc_idx'] == doc_idx].sort_values('partition_idx')
        
        # 마스킹 생성: count > 0이면 1, 아니면 0
        mask = (doc_data['count'] > 0).astype(int).tolist()
        mask_str = ''.join(map(str, mask))
        
        # 활성 partition 개수
        active_count = sum(mask)
        
        masking_results.append({
            'doc_idx': doc_idx,
            'rep_num': rep_num,
            'partition_masking': mask_str,
            'active_partition_count': active_count
        })
        
        if doc_idx < 5:  # 처음 5개만 출력
            print(f"   - doc_idx {doc_idx}: {mask_str} (활성 파티션: {active_count}개)")
    
    masking_df = pd.DataFrame(masking_results)
    print(f"✅ 마스킹 생성 완료: {len(masking_df)}개 문서")
    
    return masking_df, df_filtered, partition_indices


def calculate_partition_utilization(df_filtered, masking_df, partition_indices, rep_num):
    """
    Step 2: 마스킹 정보를 바탕으로 partition별 활용도 통계 계산
    
    Returns:
        utilization_df: DataFrame with partition utilization statistics
    """
    print(f"\n📊 Step 2: Partition별 활용도 통계 계산 중...")
    
    utilization_results = []
    
    for partition_idx in partition_indices:
        # 해당 partition의 모든 데이터
        partition_data = df_filtered[df_filtered['partition_idx'] == partition_idx].copy()
        
        # 마스킹에서 해당 partition 위치의 값 추출
        mask_values = []
        for _, row in masking_df.iterrows():
            mask_str = row['partition_masking']
            if partition_idx < len(mask_str):
                mask_values.append(int(mask_str[partition_idx]))
            else:
                mask_values.append(0)
        
        # 활용된 문서 개수 (mask=1인 개수)
        utilized_count = sum(mask_values)
        total_docs = len(masking_df)
        utilization_rate = (utilized_count / total_docs * 100) if total_docs > 0 else 0
        
        # count 값 통계 (마스킹된 문서만 대상)
        masked_counts = []
        for idx, (_, row) in enumerate(masking_df.iterrows()):
            doc_idx = row['doc_idx']
            mask_str = row['partition_masking']
            
            if partition_idx < len(mask_str) and mask_str[partition_idx] == '1':
                # 마스킹이 1인 경우만 count 값 수집
                count_value = partition_data[partition_data['doc_idx'] == doc_idx]['count'].values
                if len(count_value) > 0:
                    masked_counts.append(count_value[0])
        
        # 통계 계산
        if len(masked_counts) > 0:
            count_sum = sum(masked_counts)
            count_mean = np.mean(masked_counts)
            count_median = np.median(masked_counts)
            count_std = np.std(masked_counts)
            count_min = min(masked_counts)
            count_max = max(masked_counts)
        else:
            count_sum = count_mean = count_median = count_std = count_min = count_max = 0
        
        utilization_results.append({
            'rep_num': rep_num,
            'partition_idx': partition_idx,
            'utilized_docs': utilized_count,
            'total_docs': total_docs,
            'utilization_rate(%)': utilization_rate,
            'count_sum': count_sum,
            'count_mean': count_mean,
            'count_median': count_median,
            'count_std': count_std,
            'count_min': count_min,
            'count_max': count_max
        })
    
    utilization_df = pd.DataFrame(utilization_results)
    print(f"✅ 활용도 통계 계산 완료: {len(utilization_df)}개 파티션")
    
    return utilization_df


def print_masking_summary(masking_df):
    """마스킹 요약 정보 출력"""
    print("\n" + "="*80)
    print("📋 마스킹 요약")
    print("="*80)
    print(f"총 문서 수: {len(masking_df)}")
    print(f"활성 파티션 개수 통계:")
    print(f"  - 평균: {masking_df['active_partition_count'].mean():.2f}")
    print(f"  - 중간값: {masking_df['active_partition_count'].median():.2f}")
    print(f"  - 최소: {masking_df['active_partition_count'].min()}")
    print(f"  - 최대: {masking_df['active_partition_count'].max()}")
    print(f"  - 표준편차: {masking_df['active_partition_count'].std():.2f}")
    print("="*80)


def print_utilization_table(utilization_df):
    """활용도 결과 테이블 출력"""
    print("\n" + "="*80)
    print("📈 Partition별 활용도 통계")
    print("="*80)
    
    # 포맷팅된 출력
    pd.options.display.float_format = '{:.2f}'.format
    print(utilization_df.to_string(index=False))
    
    print("\n" + "-"*80)
    print("📊 전체 요약:")
    print(f"  - 평균 활용률: {utilization_df['utilization_rate(%)'].mean():.2f}%")
    print(f"  - 가장 많이 활용된 파티션: {utilization_df.loc[utilization_df['utilization_rate(%)'].idxmax(), 'partition_idx']} "
          f"({utilization_df['utilization_rate(%)'].max():.2f}%)")
    print(f"  - 가장 적게 활용된 파티션: {utilization_df.loc[utilization_df['utilization_rate(%)'].idxmin(), 'partition_idx']} "
          f"({utilization_df['utilization_rate(%)'].min():.2f}%)")
    print(f"  - 전체 count 합계: {utilization_df['count_sum'].sum():.0f}")
    print("="*80)


def calculate_repetition_statistics(all_utilization_df):
    """
    Repetition별 통계 계산
    각 repetition의 partition 활용도를 집계
    """
    print(f"\n📊 Repetition별 통계 계산 중...")
    
    rep_stats = []
    
    for rep_num in sorted(all_utilization_df['rep_num'].unique()):
        rep_data = all_utilization_df[all_utilization_df['rep_num'] == rep_num]
        
        rep_stats.append({
            'rep_num': rep_num,
            'avg_utilization_rate(%)': rep_data['utilization_rate(%)'].mean(),
            'std_utilization_rate(%)': rep_data['utilization_rate(%)'].std(),
            'min_utilization_rate(%)': rep_data['utilization_rate(%)'].min(),
            'max_utilization_rate(%)': rep_data['utilization_rate(%)'].max(),
            'total_count_sum': rep_data['count_sum'].sum(),
            'avg_count_mean': rep_data['count_mean'].mean(),
            'avg_utilized_docs': rep_data['utilized_docs'].mean(),
            'partitions': len(rep_data)
        })
    
    rep_stats_df = pd.DataFrame(rep_stats)
    print(f"✅ Repetition별 통계 계산 완료: {len(rep_stats_df)}개 repetition")
    
    return rep_stats_df


def print_repetition_statistics(rep_stats_df):
    """Repetition별 통계 테이블 출력"""
    print("\n" + "="*80)
    print("📈 Repetition별 통계")
    print("="*80)
    
    pd.options.display.float_format = '{:.2f}'.format
    print(rep_stats_df.to_string(index=False))
    
    print("\n" + "-"*80)
    print("📊 Repetition 간 비교:")
    print(f"  - 평균 활용률이 가장 높은 repetition: {rep_stats_df.loc[rep_stats_df['avg_utilization_rate(%)'].idxmax(), 'rep_num']} "
          f"({rep_stats_df['avg_utilization_rate(%)'].max():.2f}%)")
    print(f"  - 평균 활용률이 가장 낮은 repetition: {rep_stats_df.loc[rep_stats_df['avg_utilization_rate(%)'].idxmin(), 'rep_num']} "
          f"({rep_stats_df['avg_utilization_rate(%)'].min():.2f}%)")
    print(f"  - 전체 count 합계가 가장 높은 repetition: {rep_stats_df.loc[rep_stats_df['total_count_sum'].idxmax(), 'rep_num']} "
          f"({rep_stats_df['total_count_sum'].max():.0f})")
    print("="*80)


def save_results(df, output_path, description):
    """결과를 CSV로 저장"""
    try:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✅ {description} 저장 완료: {output_path}")
    except Exception as e:
        print(f"\n❌ {description} 저장 실패: {e}")


def main():
    args = parse_arguments()
    
    # 1. CSV 로드
    df = load_csv(args.csv_file)
    
    # 2. 데이터 검증
    validate_dataframe(df)
    
    # 3. Repetition 목록 확인
    all_rep_nums = sorted(df['rep_num'].unique())
    print(f"\n📋 사용 가능한 Repetition: {all_rep_nums}")
    
    # 4. 분석할 repetition 결정
    if args.rep_num is not None:
        # 특정 repetition만 분석
        rep_nums_to_analyze = [args.rep_num]
        print(f"✅ 단일 Repetition 분석 모드: rep_num={args.rep_num}")
    else:
        # 전체 repetition 분석
        rep_nums_to_analyze = all_rep_nums
        print(f"✅ 전체 Repetition 분석 모드: {len(rep_nums_to_analyze)}개 repetition")
    
    # 5. 각 repetition에 대해 분석 수행
    all_masking_dfs = []
    all_utilization_dfs = []
    
    for rep_num in rep_nums_to_analyze:
        print(f"\n{'='*80}")
        print(f"🔍 Repetition {rep_num} 분석 시작")
        print(f"{'='*80}")
        
        # Step 1: Partition 마스킹 생성
        masking_df, df_filtered, partition_indices = create_partition_masking(df, rep_num)
        
        if masking_df is None:
            print(f"⚠️  rep_num={rep_num} 건너뜀")
            continue
        
        # 마스킹 요약 출력
        print_masking_summary(masking_df)
        
        # Step 2: Partition 활용도 통계 계산
        utilization_df = calculate_partition_utilization(df_filtered, masking_df, partition_indices, rep_num)
        
        # 결과 출력
        print_utilization_table(utilization_df)
        
        # 결과 저장
        all_masking_dfs.append(masking_df)
        all_utilization_dfs.append(utilization_df)
    
    # 6. 전체 결과 통합
    combined_masking_df = pd.concat(all_masking_dfs, ignore_index=True)
    combined_utilization_df = pd.concat(all_utilization_dfs, ignore_index=True)
    
    # 7. Repetition별 통계 계산 (여러 repetition을 분석한 경우)
    if len(rep_nums_to_analyze) > 1:
        rep_stats_df = calculate_repetition_statistics(combined_utilization_df)
        print_repetition_statistics(rep_stats_df)
        
        # Repetition별 통계 저장
        if args.output_rep_stats:
            save_results(rep_stats_df, args.output_rep_stats, "Repetition별 통계")
        else:
            base_name = os.path.splitext(args.csv_file)[0]
            default_rep_stats_output = f"{base_name}_repetition_statistics.csv"
            save_results(rep_stats_df, default_rep_stats_output, "Repetition별 통계")
    
    # 8. 결과 파일 저장
    if args.output_mask:
        save_results(combined_masking_df, args.output_mask, "마스킹 결과")
    else:
        base_name = os.path.splitext(args.csv_file)[0]
        if len(rep_nums_to_analyze) == 1:
            default_mask_output = f"{base_name}_masking_rep{rep_nums_to_analyze[0]}.csv"
        else:
            default_mask_output = f"{base_name}_masking_all.csv"
        save_results(combined_masking_df, default_mask_output, "마스킹 결과")
    
    if args.output:
        save_results(combined_utilization_df, args.output, "활용도 통계")
    else:
        base_name = os.path.splitext(args.csv_file)[0]
        if len(rep_nums_to_analyze) == 1:
            default_util_output = f"{base_name}_utilization_rep{rep_nums_to_analyze[0]}.csv"
        else:
            default_util_output = f"{base_name}_utilization_all.csv"
        save_results(combined_utilization_df, default_util_output, "활용도 통계")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python3 partition_masking_stats.py <csv_file> --rep-num <rep_number>")
        print("도움말: python3 partition_masking_stats.py --help")
        sys.exit(1)
    
    main()