# Capacity-priority machine-agent PPO v1: kết quả N4/K2

**KHÔNG ĐẠT DEV; giữ kết quả âm và chưa mở confirmation/test.** 12/16 tiêu chí điểm đạt; mục tiêu nghiên cứu tổng thể chưa đạt.

## Giả thuyết và phương pháp

Cho máy ít lựa chọn phục vụ nhanh chọn technician trước có thể giữ cơ hội cho máy khó phục vụ; các máy sau nhìn reservation thật để chọn hoặc WAIT. Xếp máy đang có cạnh hợp lệ theo c_m=Σ_j 1/D_mj tăng dần, tie ID; thứ tự tính từ trạng thái hiện tại một lần mỗi tick. Máy không active đi cuối. Thứ tự không được coi là tối ưu; policy có thể WAIT và để máy sau được phục vụ. Mọi service start thực thi đồng thời.

Actor raw104→64→64→3 dùng chung giữa bốn máy, nhận đầy đủ state71 cùng own-machine/edges/plan/reserved/processed/ID/position. Mỗi máy chọn WAIT hoặc technician chưa được giữ chỗ. Canonical vector theo machine ID có một trace cho mỗi matching; scalar joint PPO dùng tổng log conditionals. Critic chỉ nhận state và không chạy khi select. Có19.908 tham số ở mỗi nhánh lõi, cùng tensor khởi tạo riêng và cùng sensing/budget.

Head actor cuối khởi tạo0; prior đếm completion matching bằng log DP C(n,k)=C(n−1,k)+kC(n−1,k−1). Nhờ đó mỗi matching khả thi ở bốn nhánh phối hợp có cùng xác suất ban đầu,1/21 khi đủ N4/K2. Chỉ áp dụng complete-bipartite active/free support của benchmark; không tuyên bố cho arbitrary skill masks. Actor/decoder/core observation không liệt kê matching. Việc đếm21/81 chỉ dùng trong reference độc lập.

Các ablation là fixed ID order, reverse capacity, mask-only input, independent machine physical-q và independent proposal-p. Independent requests chỉ nhìn trạng thái gốc, broker chọn R/D cao nhất trên từng technician, người thua WAIT. Hai density modes đều báo cáo. Nhánh độc lập có phân phối matching ban đầu khác do bỏ phối hợp; không diễn giải gain này thành riêng lợi ích học message. Baseline thêm raw technician-first và PPO/MAPPO với khác biệt architecture/capacity được công bố trong protocol.

Giữ PPO scalar joint, MC/state advantage chuẩn hóa chung, Adam3e−4, clip0.2, entropy0.01, global clip0.5 và reward−cost/20. Toàn bộ mô hình học từ random initialization, không teacher/labels/distillation/warm-start. Generic autoregression/variable ordering/de-biasing đã có prior art: [allocation NeurIPS2024](https://arxiv.org/abs/2409.18735), [PARCO NeurIPS2025](https://arxiv.org/abs/2409.03811), [MRV](https://aima.cs.berkeley.edu/4th-ed/pdfs/newchap05.pdf). Chưa tuyên bố mới ở các ý tưởng chung đó.

## Thiết kế đã khóa

33 fits, ba training seeds mới ×11 phương pháp,491.520 quyết định/fit; tổng16.220.160 quyết định,337.920 train episodes và253.440 Adam updates.32 DEV seeds, năm checkpoint gồm bước0. PRIMARY dùng cố định491.520; không thay bằng checkpoint/seed tốt nhất. H48 policy và12tail chung; health tại cutoff48. Không scientific test hay N lớn ở vòng này.

## Kết quả fixed final

| Nhánh | Cost/máy H60 | Availability H48 | Pending/máy H48 | Failure/máy H48 |
|---|---:|---:|---:|---:|
| Capacity order | 265.716146 | 0.669705 | 0.005208 | 0.328125 |
| Fixed ID order | 263.898438 | 0.665365 | 0.000000 | 0.187500 |
| Reverse capacity | 268.367188 | 0.665853 | 0.002604 | 0.312500 |
| Mask only | 266.156250 | 0.669759 | 0.010417 | 0.346354 |
| Independent physical q | 267.682292 | 0.669596 | 0.005208 | 0.471354 |
| Independent proposal p | 267.664062 | 0.668837 | 0.005208 | 0.445312 |
| Raw technician order | 265.585938 | 0.668077 | 0.010417 | 0.377604 |
| PPO | 262.755208 | 0.668457 | 0.000000 | 0.101562 |
| PPO normalized | 262.773438 | 0.667318 | 0.000000 | 0.156250 |
| MAPPO local | 272.791667 | 0.668566 | 0.005208 | 0.742188 |
| MAPPO full | 265.385417 | 0.667643 | 0.000000 | 0.346354 |

| Tiêu chí | Giá trị điểm | Đạt |
|---|---:|---|
| gain_vs_fixed_order_ppo | -0.00688791 | Không |
| gain_vs_reverse_capacity_ppo | 0.00987841 | Có |
| gain_vs_mask_only_order_ppo | 0.00165356 | Có |
| gain_vs_independent_machine_q_ppo | 0.00734507 | Có |
| gain_vs_independent_machine_p_ppo | 0.00727747 | Có |
| gain_vs_raw_tech_ppo | -0.00049027 | Có |
| gain_vs_ppo | -0.01126881 | Không |
| gain_vs_normalized_ppo | -0.01119865 | Không |
| gain_vs_mappo | 0.02593745 | Có |
| gain_vs_full_mappo | -0.00124622 | Không |
| availability_delta_vs_ppo | 0.00124783 | Có |
| normalized_pending_delta_vs_ppo | 0.00520833 | Có |
| normalized_failures_delta_vs_ppo | 0.22656250 | Có |
| availability_delta_vs_normalized_ppo | 0.00238715 | Có |
| normalized_pending_delta_vs_normalized_ppo | 0.00520833 | Có |
| normalized_failures_delta_vs_normalized_ppo | 0.17187500 | Có |

Gain vs fixed order từng seed: 360000000: -0.8225%, 360000001: -1.1157%, 360000002: -0.1300%.
P95 actor+decoder: 1.929319ms,432 mẫu khi pool đã đóng; chỉ chứng minh latency ở N4 trên lab.
Không đạt: gain_vs_fixed_order_ppo, gain_vs_ppo, gain_vs_normalized_ppo, gain_vs_full_mappo. Extra không đạt: every_replica_main_gain_positive.

## Bằng chứng cơ chế và giới hạn

Trong4.608 quyết định greedy candidate, có 2144 trạng thái có ít nhất hai máy active và worker rảnh; thứ tự active capacity khác ID ở 836 trạng thái. Đây là mô tả cơ hội thay thứ tự trên trajectory candidate, không phải nhãn tối ưu hay bằng chứng giảm cost.

Ablation huấn luyện cùng thông tin/dung lượng/ngân sách mới là bằng chứng chính. Không lấy zero conflict, order đổi hoặc prior đúng làm kết luận thuật toán tốt. Lợi ích so local MAPPO ít thông tin hơn cũng không thay cho đối chứng chính/full MAPPO. Nếu thiếu bất kỳ gate, giữ nguyên verdict và không tự động đổi order formula/LR/prior/budget để cứu.

Kiểm chứng độc lập dựng33 bộ tensor khởi tạo, seed roles/tapes/budgets/fixed endpoint; replay1.088episode/65.280 primitive transitions/52.224policy decisions. NumPy dựng actor và đếm matching completions trực tiếp theo prefix, tách khỏi DP của production; independent q cộng đủ81latent proposals. Tính lại16gates và432raw timings. Các historical minibatch gradients không lưu đầy đủ, nên không tuyên bố replay từng optimizer update. Audit PASS xác nhận số liệu, không chứng minh CI, scalability hay goal completion.

Nguồn: `/home/bachng/ht-pdm-fjsp/artifacts/technician_capacity_order_ppo_v1_development_20261017T052155929138Z`.
Audit: `/home/bachng/ht-pdm-fjsp/artifacts/technician_capacity_order_verification_v1_20261017T054326430768Z`.

![Chất lượng và sức khỏe](/home/bachng/ht-pdm-fjsp/artifacts/technician_capacity_order_report_v1_20261017T054359884665Z/quality_health.png)
![Learning và phối hợp](/home/bachng/ht-pdm-fjsp/artifacts/technician_capacity_order_report_v1_20261017T054359884665Z/learning_coordination.png)

## Đánh giá sau kiểm chứng

Vòng này không đạt12/16 tiêu chí điểm; bốn tiêu chí không đạt là gain so fixed ID order, NI với PPO gốc, NI với PPO normalized và superiority với full-information MAPPO. Ba replica đều kém fixed order: −0,823%, −1,116%, −0,130%. Không mở confirmation/test/N lớn, không đổi công thức capacity hoặc tăng budget để cứu phiên bản.

Có một tín hiệu đúng phạm vi: candidate tốt hơn independent physical-q0,735% trên DEV và có zero rejected proposals, trong khi independent có3,615 đề xuất bị loại/episode. Nhưng fixed ID order cũng phối hợp và rẻ hơn candidate1,817708/máy. Bỏ toàn bộ coordination có thể làm kém hơn, còn ưu tiên máy capacity thấp chưa tạo được lợi ích riêng. Không dùng tín hiệu thứ nhất che kết quả thứ hai.

Order không phải thành phần không hoạt động:836/4.608state (18,14%) có active capacity order khác active ID order;2.144state có ít nhất hai máy active và worker rảnh. Đó là mô tả trên trajectory candidate, chưa phải ước lượng hiệu ứng nhân quả của từng lần đổi thứ tự.

Cost candidate−fixed theo H60/máy khớp maintenance+0,247396, unavailability−2,062500, failure+3,632813 và waiting0, tổng+1,817708. Availability tốt hơn không bù được penalty failure. Không suy từ decomposition này rằng thêm penalty duration, sửa WAIT hay đổi encoder sẽ giải quyết.

Bộ kiểm chứng PASS14,7566s, gồm tất cả33tensor khởi tạo,365raw/82.424derived seeds,33 ngân sách,337.920 train episode records/30.720paired physical tapes,5.280DEVnodes,1.088finalepisodes/65.280physical transitions/52.224policy decisions, toàn bộ actor bằng NumPy/counting độc lập và432raw timings.47 test duy nhất đã đạt (21 policy/learner và26 reference), gồm toàn bộ64mask complete-bipartite ở N4. Historical optimizer trajectories không được replay độc lập.

Một đối chứng có cost thấp hơn chưa được phép tự động trở thành “phương pháp thành công”. Cần một giả thuyết đóng góp khác, protocol riêng và dữ liệu mới nếu tiếp tục nghiên cứu. Generic machine-agent autoregression đã có trong repo và prior art; kết quả này không đổi điều đó.
