## Material Passport

- Origin: ARS experiment-agent, inline.
- Verification: ANALYZED; artifact hashes verified, runner physical replay PASS.

# Vòng 1: joint đạt gate thứ cấp, assignment chưa đạt

Run: /Users/bachng/Coding/Reinforcement Learning/marl/artifacts/maintenance_lookahead_boundary_v1_full_mac_cpu_20261012T060421Z

896 episode, 53.760 physical tick, 896 common landmarks, macOS CPU, 191,60s. Tất cả 23 SHA256/size khớp. Joint endpoint giảm cost 3,6224%, CI đồng thời 99,5833% [2,0551%;5,3272%], availability +1,3875 điểm %, pending không đổi. Joint đạt cả ba gate đã khóa. Assignment endpoint chỉ +0,3508%, CI [0,0017%;1,0962%], chưa đạt cận dưới 1%. Giữ original scientific status NOT_CONFIRMED vì primary của protocol là assignment; không đổi verdict thành joint-primary sau xem kết quả. Hướng joint đã được người dùng chọn riêng, nên cần fresh confirmation với baseline kinh tế mạnh hơn trước khi chốt.

Giới hạn: known-model planning, một fixed physical profile, không chứng minh RL learnability hay MARL cần thiết. Eleven fallacy checks: aggregation kept per method (Simpson/ecological), fixed post-screen anchor disclosed (Berkson/regression to mean), no selected-state gate (collider/base rate), complete losses retained (survivorship), locked twelve-interval family (look elsewhere/forking paths), episode policy comparison separated from mechanism and policy-induced occupancy (causality/reverse causality). 11/11 checked.
