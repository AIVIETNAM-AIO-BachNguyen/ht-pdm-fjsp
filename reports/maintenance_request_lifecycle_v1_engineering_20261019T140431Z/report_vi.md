# Báo cáo bước 1–3: vòng đời yêu cầu bảo trì dùng chung

## Material Passport
Dữ liệu mô phỏng tổng hợp; engineering trên Mac ngày 19/10/2026. Kế thừa
commit `2e190435bde46e99ac608ea19a3413a9bcc3a08c`; nhánh bàn giao
`codex/maintenance-request-lifecycle-v1`. Không training, không chạy full,
không SSH hoặc rsync. Kết quả smoke chỉ kiểm tra kỹ thuật.

## Phần đã triển khai

1. Tạo checkout riêng từ bản maintenance benchmark đã xác minh. Không sửa
   các thay đổi trong workspace gốc hoặc thay ngữ nghĩa môi trường cũ.
2. Khóa hợp đồng vận hành và protocol trước smoke trong
   `docs/maintenance_request_lifecycle_v1_plan.md`.
3. Thêm `maintenance_requests.py`: mỗi máy tối đa một yêu cầu; danh sách chờ
   chung; request ID theo máy và số thứ tự; giữ thời điểm tạo; không rút yêu cầu.
   Máy chờ PM vẫn chạy và suy giảm, hỏng thì cùng yêu cầu chuyển CM. Technician
   chỉ được gắn lúc bắt đầu sửa; sửa không ngắt; phục hồi lúc hoàn tất.
4. Đếm riêng thời gian chờ PM, chờ CM và thời gian sửa. Báo cả yêu cầu còn chờ,
   tuổi chờ và công việc đang sửa tại H và cuối tail, không chỉ yêu cầu hoàn tất.
5. Bổ sung hai requester × ba dispatcher: threshold/resource-aware, mỗi loại
   với FCFS/risk matching/rollout. Policy chỉ nhận cấu hình và trạng thái hiện
   tại; rollout dùng shock giả định riêng, không dùng future tape thật.

## Kiểm tra kỹ thuật

- 23 test mới PASS, gồm tính tay transition/cost, yêu cầu PM chuyển CM,
  hoàn tất sửa, danh sách chờ chung, matcher đối chiếu exhaustive machine
  proposals, duplicate/invalid action không đổi state, request conservation,
  source seed audit, overwrite refusal, FAILED manifest khi timeout,
  dirty-source full guard và phát hiện log shock bị sửa.
- 127 test hồi quy liên quan PASS trên checkout mới.
- Toàn bộ test repository: **988 PASS trong314,59s**. Một warning hiển thị
  tqdm/rich từ test cũ, không có test thất bại.
- Smoke CLI trên macOS ARM, Python3.12.13/Numpy2.5.3: COMPLETED, 36 episode,
  720 transition vật lý, 2.256 forecast transition được thực hiện khi chọn
  action. Replay mọi physical transition và tái dựng toàn bộ controller/
  forecast bank của smoke: PASS; zero invalid assignment.
- Smoke runner mất khoảng 0,38s trong môi trường đã có dependencies;
  đây không phải thời gian dự đoán full trên lab.

## Ví dụ quan sát được trong log

Cell `interchangeable_scale2`, threshold/FCFS, rawseed61100000,
machine index2, request sequence1 (index máy bắt đầu từ0):

| Mốc tick | Sự kiện |
|---|---|
| 2 | Tạo yêu cầu PM, máy tiếp tục chạy |
| 3 | Máy hỏng, cùng yêu cầu chuyển CM; created_at vẫn2 |
| 4 | Technician index0 bắt đầu sửa; thời gian chờ2tick |
| 8 | Sửa hoàn tất, máy được phục hồi và yêu cầu đóng |

Trong 36 episode smoke có tổng376 machine-tick chờ PM,108 chờ CM,
152 lần upgrade PM→CM,429 yêu cầu hoàn tất. Mọi episode đều có chờ PM;
26 episode có chờ CM. Có yêu cầu còn chờ ở cuối tail trong17episode,
nên tail hữu hạn không được diễn giải là đã giải quyết hết backlog.
Các tổng này bao gồm các phương pháp dùng chung physical seed; chúng là
coverage kỹ thuật, không phải các mẫu độc lập để suy luận thống kê.

## Giới hạn và bước tiếp

Gửi yêu cầu sớm miễn phí và dispatcher được trì hoãn tùy ý. Vì vậy gửi sớm
có thể chỉ mở rộng các lựa chọn của dispatcher; chưa có tradeoff nội tại
để tuyên bố đã tối ưu thời điểm yêu cầu. Không thêm penalty để tạo lợi ích.
Hai requester được đánh giá như hai quy tắc vận hành cụ thể.

Rollout nhìn một cửa sổ hữu hạn, dùng transition model đã biết và baseline
continuation trong cửa sổ, không có terminal value sau cửa sổ. Đây chưa phải
oracle tối ưu dài hạn; nó có thể thua và chịu bias tại biên forecast. Common
tail xử lý cutoff đánh giá H, không tự sửa bias của forecast ngắn. Muốn kết
luận về giá trị phối hợp dài hạn cần đối chứng mạnh hơn trên bài nhỏ.

Chưa xếp hạng phương pháp từ smoke. Diagnostic đầy đủ đã chuẩn bị:
9 physical profile ×6controller ×64physicalseed =3.456episode,
221.184 physical tick (H48+tail16), CPU, timeout cứng4giờ.
Không training hoặc checkpoint; stdout có tqdm; artifact lưu thư mục mới.
CI bootstrap nếu có chỉ mang tính khám phá, không bảo đảm familywise.

Sau khi người dùng chạy full trên lab và kéo kết quả về, cần kiểm tra
manifest/audit/hash, đọc tất cả profile và paired contrasts, rồi xác định
phần quyết định còn cần tối ưu. Chưa phát triển thêm MARL trong bước này.

## Bản bàn giao

Commit `7390109a976a2f0cf55a67b6205d150623a3c0c5`; nhánh `codex/maintenance-request-lifecycle-v1`.
Smoke được chạy trước commit; hash các source và protocol khớp Git blob của
commit này. Receipt lưu trong `source_seal_verification.json`.
