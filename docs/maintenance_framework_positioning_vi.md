# Bối cảnh nghiên cứu mang sang lab

Đây là ghi chú bàn giao từ trao đổi và lần đọc bài trước đó, không phải một
literature review mới hoặc xác nhận tính mới đầy đủ.

Người dùng ưu tiên setting rõ để phát triển framework, chấp nhận RL tập trung
hoặc MARL. Hai nhóm quyết định dài hạn là bảo trì và lập lại lịch sản xuất,
nhưng bản nền hiện tại chỉ triển khai maintenance/resource coordination.
Inspection, giảm tải và thay dụng cụ chưa có action riêng. FJSP là hướng tích
hợp sau khi controller/environment/evaluation interface ổn định.

Bài MIRA đã được người dùng cung cấp: Future Internet 2026, 18(8), 430,
[DOI 10.3390/fi18080430](https://doi.org/10.3390/fi18080430). Bản PDF gốc ở Mac
không được đưa vào Git. Lần đọc trước nhận diện các role maintenance,
scheduling và communication; không nên đồng nhất các role đó với mỗi máy hoặc
mỗi technician là một agent. MIRA phối hợp health prediction, graph/RL và
feasibility shield trong một framework. Dữ liệu health thật không tự xác nhận
chất lượng vận hành trên nhà máy; phần quyết định được đánh giá trong simulator.

Không dùng riêng kết hợp MARL + hard feasibility + communication để tuyên bố
novelty. Shield có thể giải thích việc zero violations; cần phân biệt đóng góp
của learned policy và enforcement. Đọc cả availability, failure, tardiness và
seed variability thay vì chỉ cost tổng. Bản đọc trước không xác nhận MIRA đã
hoặc chưa có đầy đủ persistent request identity/created_at/upgrade contract ở
code công khai; không suy từ thiếu mô tả thành chứng minh đối thủ không có.

Hướng đáng phát triển của mình: hợp đồng request lifecycle rõ, completion-time
repair, nguồn lực technician không đồng nhất, feasibility quan sát được,
controller thay thế được và evaluation cơ chế/chi phí đầy đủ. Đây là phạm vi
kỹ thuật cho framework v1; cần tra primary literature và đối chiếu implementation
trước khi biến thành contribution claim cho bài báo.

Các literature notes/proposal cũ trong docs/ và docs/research/ được giữ để không
mất lịch sử. Chúng có thể chứa nhận định đã được thu hẹp hoặc thay thế. Không
coi mọi kết luận cũ là trạng thái hiện tại, không tự tạo citation hoặc claim
trong paper từ riêng ghi chú bàn giao này.
