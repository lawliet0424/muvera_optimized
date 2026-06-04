import sys

# 기대 컬럼: doc_idx rep_idx token_idx partition_index doc_length
def parse_row(line: str):
    line = line.strip()
    if not line:
        return None
    parts = line.split()  # 공백/탭 모두 처리
    if parts[0] == "doc_idx":  # header
        return None
    if len(parts) < 5:
        raise ValueError(f"Bad line (expected 5 cols): {line}")

    doc_idx = int(parts[0])
    rep_idx = int(parts[1])
    token_idx = int(parts[2])
    part = int(float(parts[3]))      # 혹시 0.0 같은 게 섞여도 안전
    doc_len = int(float(parts[4]))
    return (doc_idx, rep_idx, token_idx, part, doc_len)

def main(cpu_path: str, gpu_path: str, out_path: str = "partition_mismatch.txt", max_write: int = 200000):
    compared = 0
    mismatched = 0
    key_mismatch = 0
    len_mismatch = 0

    with open(cpu_path, "r", encoding="utf-8") as fc, \
         open(gpu_path, "r", encoding="utf-8") as fg, \
         open(out_path, "w", encoding="utf-8") as out:

        while True:
            lc = fc.readline()
            lg = fg.readline()

            if not lc and not lg:
                break
            if not lc or not lg:
                # 한쪽이 먼저 끝남 => dump 범위 자체가 다름
                print("[ERROR] File length differs (one file ended earlier).")
                break

            rc = parse_row(lc)
            rg = parse_row(lg)
            if rc is None or rg is None:
                continue

            compared += 1

            kc = rc[:3]
            kg = rg[:3]
            if kc != kg:
                key_mismatch += 1
                # 키 mismatch는 보통 정렬/출력 순서가 다르거나 한쪽이 누락된 것
                if key_mismatch <= 20:
                    out.write(f"KEY_MISMATCH cpu={kc} gpu={kg}\n")
                continue

            if rc[4] != rg[4]:
                len_mismatch += 1
                if len_mismatch <= 20:
                    out.write(f"DOCLEN_MISMATCH {kc} cpu_len={rc[4]} gpu_len={rg[4]}\n")

            if rc[3] != rg[3]:
                mismatched += 1
                if mismatched <= max_write:
                    out.write(f"{kc[0]} {kc[1]} {kc[2]} CPU={rc[3]} GPU={rg[3]} doclen={rc[4]}\n")

            if compared % 5_000_000 == 0:
                print(f"... compared {compared:,} rows, mismatched {mismatched:,}, key_mismatch {key_mismatch:,}")

    print("Total compared:", f"{compared:,}")
    print("Partition mismatches:", f"{mismatched:,}")
    print("Key mismatches:", f"{key_mismatch:,}")
    print("Doc_length mismatches:", f"{len_mismatch:,}")
    print("Saved:", out_path)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_partition_dump_stream.py cpu.txt gpu.txt [output.txt]")
        sys.exit(1)

    cpu = sys.argv[1]
    gpu = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "partition_mismatch.txt"
    main(cpu, gpu, out)
