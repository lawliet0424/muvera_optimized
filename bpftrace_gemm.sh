#!/bin/bash

OBLAS_SO=$(python3 print_openblas_path.py)
LIBC=/usr/lib/x86_64-linux-gnu/libc.so.6
#nm -D "$OBLAS_SO" | grep -Ei 'cblas_sgemm|sgemm_' || true
PID=$(pgrep -n python3)   # 또는 pgrep -n your_script_name # cblas_sgemm64_  cblas_sgemmt64_
# print(@incopy); print(@itcopy);  print(@oncopy);  print(@otcopy);

sudo bpftrace -e '
uprobe:'"$OBLAS_SO"':cblas_sgemm64_ { @depth[pid] = @depth[pid] + 1; }
uprobe:'"$OBLAS_SO"':cblas_sgemm64_ { @depth[pid] = @depth[pid] - 1; }
uprobe:'"$OBLAS_SO"':sgemm_incopy_* { @incopy = count(); }
uprobe:'"$OBLAS_SO"':sgemm_itcopy_* { @itcopy = count(); }
uprobe:'"$OBLAS_SO"':sgemm_oncopy_* { @oncopy = count(); }
uprobe:'"$OBLAS_SO"':sgemm_otcopy_* { @otcopy = count(); }

uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:malloc         /@depth[pid]>0/ { @bytes += arg0; @calls += 1; }
uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:calloc         /@depth[pid]>0/ { @bytes += arg0*arg1; @calls += 1; }
uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:realloc        /@depth[pid]>0/ { @calls += 1; }   // 크기 계산 어려우니 호출수만
uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:posix_memalign /@depth[pid]>0/ { @bytes += arg1; @calls += 1; }
uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:aligned_alloc  /@depth[pid]>0/ { @bytes += arg0*arg1; @calls += 1; }
uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:valloc         /@depth[pid]>0/ { @bytes += arg0; @calls += 1; }
uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:mmap           /@depth[pid]>0/ { @bytes += arg1; @calls += 1; }
uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:mmap64         /@depth[pid]>0/ { @bytes += arg1; @calls += 1; }

interval:ms:100 {
printf("GEMM-alloc bytes=%ld calls=%ld\n", @bytes, @calls);
print(@incopy);
print(@oncopy);
}
' -p $PID

